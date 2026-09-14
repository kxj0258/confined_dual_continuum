"""固定基座实验装配与记录，仅由 cdc-run 显式分派。"""
from contextlib import ExitStack
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from uuid import uuid4
import hashlib
import json
import shutil
import subprocess
import time
import numpy as np
import yaml
from confined_dual_continuum.adapters.tracking_backend import TrackingBackend, build_tracking_scene
from confined_dual_continuum.adapters.observation_samples import load_observation_samples
from confined_dual_continuum.application.fixed_tracking import FixedTrackingController, TrackingConfig
from confined_dual_continuum.domain.camera import CameraModel
from confined_dual_continuum.domain.geometry import quaternion_rotation
from confined_dual_continuum.robot.configuration import load_robot_model
from confined_dual_continuum.task.timed_trajectory import TimedTrajectory


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def _snapshot(output, config_path, values, no_obstacles):
    snapshot = output/'snapshot'
    snapshot.mkdir()
    robot = (config_path.parent/values['robot_config']).resolve()
    source = (config_path.parent/values['mujoco_xml']).resolve()
    shutil.copy2(config_path, snapshot/'source-config.yaml')
    shutil.copy2(robot, snapshot/'robot.yaml')
    shutil.copytree(source.parent.parent, snapshot/'assets')
    package = Path(__file__).resolve().parent
    shutil.copytree(package, snapshot/'src'/package.name,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    frozen = deepcopy(values)
    frozen.update(robot_config='robot.yaml', mujoco_xml=f'assets/{source.parent.name}/{source.name}',
                  output_directory='../replay')
    if values.get('observation_samples') is not None:
        # 保留原始证据路径，摘要另记录文件哈希和实际使用的精简样本库。
        frozen['observation_samples'] = str((config_path.parent/values['observation_samples']).resolve())
    if no_obstacles:
        frozen['obstacles'] = []
    (snapshot/'config.yaml').write_text(yaml.safe_dump(frozen, allow_unicode=True, sort_keys=False), encoding='utf-8')
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=package.parents[1],
                              capture_output=True, text=True, check=False)
    status = subprocess.run(['git', 'status', '--porcelain'], cwd=package.parents[1],
                            capture_output=True, text=True, check=False)
    versions = {}
    for name in ('numpy', 'PyYAML', 'mujoco', 'imageio', 'imageio-ffmpeg'):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    hashes = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in snapshot.rglob('*') if p.is_file()}
    _write_json(output/'manifest.json', {'git_revision': revision.stdout.strip() if revision.returncode == 0 else None,
        'git_dirty': bool(status.stdout) if status.returncode == 0 else None, 'versions': versions,
        'snapshot_sha256': hashes, 'no_obstacles': no_obstacles})
    return snapshot/'robot.yaml', snapshot/'assets'/source.parent.name/source.name


def run_tracking(config_path, values, *, output_parent=None, viewer_enabled=False,
                 video_enabled=True, no_obstacles=False):
    """仿真时钟不读取从臂状态；墙钟慢仅影响运行耗时，不改变轨迹时间。"""
    dt = float(values['control_period_s'])
    trajectory = TimedTrajectory(**values['trajectory'])
    config = TrackingConfig(**values['control'])
    fps = float(values['video']['fps'])
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(fps) or not 0 < fps <= 1/dt:
        raise ValueError('控制周期或视频帧率无效。')
    steps = int(round(trajectory.duration_s/dt))
    if steps < 1 or not np.isclose(steps*dt, trajectory.duration_s, atol=1e-10, rtol=0):
        raise ValueError('轨迹总时间必须是控制周期的整数倍。')
    parent = Path(output_parent).resolve() if output_parent else (config_path.parent/values['output_directory']).resolve()
    output = parent/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    print(f'结果目录：{output}', flush=True)
    summary = {'protocol_completed': False, 'termination_reason': 'initializing',
               'observer_never_gates_executor': True, 'output_directory': str(output)}
    errors, visible_count, lost_run, longest_lost = [], 0, 0, 0
    statuses, contacts, rows = Counter(), 0, 0
    min_gap, min_interarm, maximum_residual, maximum_base_drift = None, None, 0., 0.
    backend = None
    wall_start = time.perf_counter()
    try:
        robot_path, source = _snapshot(output, config_path, values, no_obstacles)
        scene = output/'scene.xml'
        build_tracking_scene(source, scene, [] if no_obstacles else values['obstacles'])
        robot = load_robot_model(robot_path)
        backend = TrackingBackend(scene, robot)
        state = backend.reset()
        camera = CameraModel(float(backend.model.cam_fovy[backend.camera_id]), **values['camera'])
        if not camera.near_m+camera.target_radius_m < config.observation_distance_m < camera.far_m-camera.target_radius_m:
            raise ValueError('观察距离必须位于相机有效深度内。')
        candidates = ()
        if values.get('observation_samples') is not None:
            sample_path = (config_path.parent/values['observation_samples']).resolve()
            candidates, sample_hash = load_observation_samples(sample_path, config.joint_margin_rad,
                                                               config.maximum_curvature_per_m)
            _write_json(output/'observation_candidates.json', {
                'source': str(sample_path), 'source_sha256': sample_hash,
                'candidates': [{'sample': c.sample, 'bending': c.bending.tolist(),
                                'tip_root': c.tip_root.tolist()} for c in candidates]})
        controller = FixedTrackingController(robot, camera, config, candidates)
        controller.reset(state)
        base_position = state.base_pose.position.copy()
        base_rotation = quaternion_rotation(state.base_pose.quaternion_wxyz)
        camera_config = {'fovy_deg': camera.fovy_deg, **values['camera']}
        _write_json(output/'camera.json', camera_config)
        with ExitStack() as stack:
            stream = stack.enter_context((output/'trace.jsonl').open('w', encoding='utf-8'))
            viewer, writer, global_writer = None, None, None
            if viewer_enabled:
                import mujoco.viewer
                viewer = stack.enter_context(mujoco.viewer.launch_passive(backend.model, backend.data))
                with viewer.lock():
                    viewer.cam.lookat[:] = base_position+[0., 0., .09]
                    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = .35, 135., -25.
            if video_enabled:
                import imageio.v2 as imageio
                writer = imageio.get_writer(output/'observer.mp4', fps=fps)
                stack.callback(writer.close)
                global_writer = imageio.get_writer(output/'global.mp4', fps=fps)
                stack.callback(global_writer.close)
            frames = 0
            summary['termination_reason'] = 'trajectory_elapsed'
            for step in range(steps+1):
                if viewer is not None and not viewer.is_running():
                    summary['termination_reason'] = 'viewer_closed'
                    break
                t = step*dt
                state = backend.read_state()
                point_base, velocity_base = trajectory.sample(t)
                target = base_position+base_rotation@point_base
                geometry = backend.geometry(state.executor.tip_pose.position, camera.target_radius_m)
                final_sample = step == steps
                result = (controller.observe(state, geometry, target, t) if final_sample else
                          controller.step(state, geometry, target, base_rotation@velocity_base, dt, t))
                record = result.diagnostics
                contact_count = backend.interarm_contact_count()
                contacts += int(contact_count > 0)
                drift = max(float(np.linalg.norm(state.base_pose.position-base_position)),
                            float(np.linalg.norm(quaternion_rotation(state.base_pose.quaternion_wxyz)-base_rotation)))
                maximum_base_drift = max(maximum_base_drift, drift)
                record.update(sample_time_s=t, command_applied=False,
                              record_type='terminal_state' if final_sample else 'control',
                              base_drift=drift, interarm_contact_count=contact_count,
                              measurement=backend.capture_response_measurement())
                # 最终时刻只采集状态，不发送额外控制周期。
                if not final_sample and controller.executor_fault is None and drift < 1e-10:
                    backend.apply_command(result.command)
                    record['command_applied'] = True
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
                if step % 100 == 0:
                    stream.flush()
                errors.append(record['executor_error_m'])
                visible_count += int(record['visible'])
                lost_run = 0 if record['visible'] else lost_run+1
                longest_lost = max(longest_lost, lost_run)
                statuses[record['observer_status']] += 1
                for key in ('executor_compatibility_residual', 'observer_compatibility_residual'):
                    maximum_residual = max(maximum_residual, record[key])
                gap, interarm = record['minimum_clearance_m'], record['minimum_interarm_clearance_m']
                if gap is not None:
                    min_gap = gap if min_gap is None else min(min_gap, gap)
                if interarm is not None:
                    min_interarm = interarm if min_interarm is None else min(min_interarm, interarm)
                rows += 1
                if writer is not None and t+1e-10 >= frames/fps:
                    try:
                        writer.append_data(backend.capture_observer_frame(camera.width, camera.height))
                        global_writer.append_data(backend.capture_global_follow_frame(camera.width, camera.height,
                            look_at_offset_m=np.array([0., 0., .09]), distance_m=.35,
                            azimuth_deg=135., elevation_deg=-25.))
                        frames += 1
                    except Exception as exc:
                        # 视频为旁路记录，渲染故障不改变主臂任务。
                        summary['video_error'] = f'{type(exc).__name__}: {exc}'
                        writer = None
                if viewer is not None:
                    viewer.sync()
                if drift >= 1e-10:
                    summary['termination_reason'] = 'fixed_base_contract_violated'
                    break
                if controller.executor_fault:
                    summary['termination_reason'] = controller.executor_fault
                    break
                if final_sample:
                    summary['protocol_completed'] = True
                    break
                backend.step(dt)
                if viewer is not None:
                    remaining = wall_start+(step+1)*dt-time.perf_counter()
                    if remaining > 0:
                        time.sleep(min(remaining, dt))
    except KeyboardInterrupt:
        summary['termination_reason'] = 'user_interrupted'
    except Exception as exc:
        summary['termination_reason'] = 'exception'
        summary['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if backend is not None:
            backend.stop()
            backend.close()
        summary.update(samples=rows, wall_time_s=time.perf_counter()-wall_start,
            executor_rmse_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            executor_maximum_error_m=max(errors) if errors else None,
            executor_final_error_m=errors[-1] if errors else None,
            visible_fraction=visible_count/rows if rows else None,
            longest_invisible_s=max(0, longest_lost-1)*dt,
            observer_status_counts=dict(statuses), interarm_contact_samples=contacts,
            minimum_clearance_m=min_gap, minimum_interarm_clearance_m=min_interarm,
            maximum_compatibility_residual=maximum_residual, maximum_base_drift=maximum_base_drift,
            note='protocol_completed 只表示主臂时钟运行完毕；不表示跟踪、可见或安全验收通过。')
        _write_json(output/'summary.json', summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
