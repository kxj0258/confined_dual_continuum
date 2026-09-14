"""独立求解/渲染进程。队列有界，控制进程仅使用非阻塞收发。"""
import csv
import json
import time
from pathlib import Path
from queue import Empty
import numpy as np
from confined_dual_continuum.adapters.executor_observer_backend import DualTrackingBackend
from confined_dual_continuum.coordination.observer_tracking import solve_observer
from confined_dual_continuum.domain.camera import CameraModel
from confined_dual_continuum.robot.configuration import load_robot_model
from confined_dual_continuum.adapters.executor_display import add_tracking_geometry, configure_camera
from confined_dual_continuum.task.catalog_trajectory import CatalogTrajectory


def solver_worker(scene, robot_path, config, requests, results, ready, stop):
    backend = None
    try:
        robot = load_robot_model(Path(robot_path))
        backend = DualTrackingBackend(Path(scene), robot)
        backend.reset()
        ready.set()
        avoidance_latched = False
        while not stop.is_set():
            try:
                request = requests.get(timeout=.05)
            except Empty:
                continue
            index, submitted, snapshot, target, ex_rate = request
            deadline = submitted+config['budget_ms']/1000
            cancelled = lambda: stop.is_set() or time.perf_counter() >= deadline
            started = time.perf_counter()
            reply = dict(index=index, submitted=submitted, target=target,
                         status='deadline', rate=None, constraints=None)
            timings = {'queue': started-submitted}
            stage_name, stage_started = None, started
            def enter_stage(name):
                nonlocal stage_name, stage_started
                if cancelled():
                    raise TimeoutError('请求预算耗尽。')
                now = time.perf_counter()
                if stage_name is not None:
                    timings[stage_name] = now-stage_started
                stage_name, stage_started = name, now
            try:
                if not cancelled():
                    enter_stage('restore')
                    backend.restore(snapshot)
                    enter_stage('feedback')
                    state = backend.read_state()
                    enter_stage('geometry')
                    geometry = backend.geometry(state.executor.tip_pose.position, .001)
                    gaps = [g.distance_m for g in geometry.clearances
                            if g.arm == 'observer' and g.other_arm == 'executor']
                    reply.update(minimum_interarm_clearance_m=min(gaps) if gaps else None,
                                 occluded=bool(geometry.occluded))
                    enter_stage('camera_diagnostics')
                    point = geometry.camera_world[:3, :3].T@(
                        state.executor.tip_pose.position-geometry.camera_world[:3, 3])
                    camera = CameraModel(float(backend.model.cam_fovy[backend.camera_id]),
                        config['camera_width'], config['camera_height'], .01, .15, .1, .001)
                    reply.update(target_camera_m=point.tolist(),
                                 visible=camera.visible(point, geometry.occluded))
                    if not cancelled():
                        rate, status, constraints = solve_observer(robot, state, geometry,
                            target, ex_rate, config['control_dt_s'], config, cancelled,
                            enter_stage, avoidance_latched)
                        reply.update(rate=rate, status=status, constraints=constraints)
                        if status == 'tracking' and constraints is not None:
                            avoidance_latched = bool(constraints[2].get(
                                'avoidance_latched', avoidance_latched))
            except TimeoutError:
                reply.update(status='deadline', rate=None)
            except Exception as exc:
                reply.update(status='solver_exception', error=f'{type(exc).__name__}: {exc}')
            if stage_name is not None:
                timings[stage_name] = time.perf_counter()-stage_started
            reply['stage_wall_s'] = timings
            reply['deadline_stage'] = (stage_name or 'queue') if reply['status'] == 'deadline' else None
            reply['request_wall_s'] = time.perf_counter()-submitted
            reply['wall_s'] = time.perf_counter()-started
            # 一次请求只产生一次响应；没有求解器切换、松弛重求或积分提交。
            results.put_nowait(reply)
    finally:
        if backend is not None:
            backend.close()


def video_worker(scene, robot_path, config, output, frames, ready, stop):
    """运行期间重建收到的状态并编码；不推进物理。丢帧以保持帧补齐时间轴。"""
    backend = writer = global_writer = viewer = image_window = None
    report = dict(saved=False, global_saved=False, frames=0, repeated_frames=0)
    output = Path(output)
    try:
        backend = DualTrackingBackend(Path(scene), load_robot_model(Path(robot_path)))
        backend.reset()
        if config['enabled']:
            import imageio.v2 as imageio
            writer = imageio.get_writer(output/'observer.mp4', fps=config['fps'])
            global_config = config.get('global_follow', {})
            if global_config.get('enabled', True):
                global_writer = imageio.get_writer(
                    output / str(global_config.get('filename', 'global.mp4')),
                    fps=config['fps'])
        trajectory = CatalogTrajectory(config['trajectory_definition'], config['time_scale'])
        base = backend._site_transform('mobile_base_frame')
        def reference(t):
            return base[:3, 3]+base[:3, :3]@trajectory.sample(t)[0]
        reference_path = np.array([reference(t) for t in np.linspace(0., trajectory.duration_s, 301)])
        actual_path = []
        if config['viewer']:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(backend.model, backend.data)
            configure_camera(viewer.cam, base[:3, 3])
        if config['camera_window']:
            from confined_dual_continuum.diagnostics.observer_image_window import ObserverImageWindow
            image_window = ObserverImageWindow(config['width'], config['height'])
        # 在开始控制之前完成 OpenGL 初始化。
        if writer is not None or global_writer is not None or image_window is not None:
            backend.capture_observer_frame(config['width'], config['height'])
        ready.set()
        previous = None
        previous_global = None
        next_index = 0
        with (output/'video_frames.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            rows = csv.writer(stream)
            rows.writerow(['frame', 'video_time_s', 'source_time_s', 'repeated'])
            previous_time = 0.
            while True:
                if image_window is not None:
                    image_window.poll()
                try:
                    snapshot = frames.get(timeout=.02)
                except Empty:
                    if stop.is_set():
                        break
                    continue
                if snapshot is None:
                    break
                index = int(round(snapshot['time']*config['fps']))
                if index < next_index:
                    continue
                backend.restore(snapshot)
                actual = backend._site_transform('executor_tip')[:3, 3].copy()
                actual_path.append(actual)
                if viewer is not None and viewer.is_running():
                    with viewer.lock():
                        viewer.user_scn.ngeom = 0
                        add_tracking_geometry(backend._mj, viewer.user_scn,
                            reference(snapshot['time']), actual, reference_path, actual_path)
                    viewer.sync()
                rgb = None
                if writer is not None or (image_window is not None and image_window.root is not None):
                    rgb = backend.capture_observer_frame(config['width'], config['height'])
                global_rgb = None
                if global_writer is not None:
                    global_config = config.get('global_follow', {})
                    camera_args = dict(
                        look_at_offset_m=np.asarray(
                            global_config.get('look_at_offset_m', [0.0, 0.0, 0.0]),
                            dtype=float),
                        distance_m=float(global_config.get('distance_m', 0.32)),
                        azimuth_deg=float(global_config.get('azimuth_deg', 135.0)),
                        elevation_deg=float(global_config.get('elevation_deg', -20.0)),
                        overlay=lambda scene: add_tracking_geometry(
                            backend._mj, scene, reference(snapshot['time']), actual,
                            reference_path, actual_path))
                    if 'local_direction_m' in global_config:
                        camera_args['local_direction_m'] = np.asarray(
                            global_config['local_direction_m'], dtype=float)
                    global_rgb = backend.capture_global_follow_frame(
                        config['width'], config['height'], **camera_args)
                if image_window is not None and rgb is not None:
                    image_window.show(rgb, snapshot['time'])
                while next_index < index:
                    if writer is not None:
                        writer.append_data(previous if previous is not None else rgb)
                    if global_writer is not None:
                        global_writer.append_data(
                            previous_global if previous_global is not None else global_rgb)
                    rows.writerow([next_index, next_index/config['fps'],
                                   previous_time if previous is not None else snapshot['time'], 1])
                    next_index += 1
                    report['repeated_frames'] += 1
                if writer is not None:
                    writer.append_data(rgb)
                if global_writer is not None:
                    global_writer.append_data(global_rgb)
                rows.writerow([next_index, next_index/config['fps'], snapshot['time'], 0])
                next_index += 1
                previous, previous_time = rgb, snapshot['time']
                previous_global = global_rgb
                report['frames'] = next_index
        if writer is not None:
            writer.close()
            writer = None
            report['saved'] = report['frames'] > 0
        if global_writer is not None:
            global_writer.close()
            global_writer = None
            report['global_saved'] = report['frames'] > 0
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        try:
            if writer is not None:
                writer.close()
            if global_writer is not None:
                global_writer.close()
        except Exception as exc:
            report['close_error'] = str(exc)
        if viewer is not None:
            viewer.close()
        if image_window is not None:
            image_window.close()
        if backend is not None:
            backend.close()
        (output/'video_summary.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
