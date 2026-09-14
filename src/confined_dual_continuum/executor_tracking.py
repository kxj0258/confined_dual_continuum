"""主臂独立轨迹入口：50 Hz 控制，独立误差窗口，默认保存视频。"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
from uuid import uuid4
import hashlib
import json
import shutil
import subprocess
import time
import sys
import numpy as np
import yaml
from confined_dual_continuum.adapters.executor_backend import ExecutorBackend, build_executor_scene
from confined_dual_continuum.adapters.executor_display import add_tracking_geometry, configure_camera, export_video
from confined_dual_continuum.application.executor_tracking import ExecutorTrackingController
from confined_dual_continuum.diagnostics.executor_records import RunRecords, COLUMNS
from confined_dual_continuum.diagnostics.executor_error_plot import save_error_plot
from confined_dual_continuum.robot.configuration import load_robot_model
from confined_dual_continuum.task.catalog_trajectory import CatalogTrajectory
from confined_dual_continuum.trajectory_catalog import load_approved


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def snapshot(output, config_path, values, robot_path, source):
    folder = output/'snapshot'
    folder.mkdir()
    shutil.copy2(config_path, folder/'source-config.yaml')
    shutil.copy2(robot_path, folder/'robot.yaml')
    shutil.copy2(source, folder/'source.xml')
    (folder/'effective-config.yaml').write_text(yaml.safe_dump(values, allow_unicode=True, sort_keys=False), encoding='utf-8')
    package = Path(__file__).resolve().parent
    shutil.copytree(package, folder/'src'/package.name,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=package.parents[1],
                              capture_output=True, text=True, check=False)
    status = subprocess.run(['git', 'status', '--porcelain'], cwd=package.parents[1],
                            capture_output=True, text=True, check=False)
    write_json(output/'manifest.json', {
        'git_revision': revision.stdout.strip() if revision.returncode == 0 else None,
        'git_dirty': bool(status.stdout) if status.returncode == 0 else None,
        'source_xml': str(source), 'robot_config': str(robot_path),
        'robot_sha256': hashlib.sha256(robot_path.read_bytes()).hexdigest(),
        'source_xml_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'scene': 'executor_only_no_interarm_collision_pairs',
        'note': '网格由 scene.xml 引用原仓库绝对路径；不复制网格，不是独立可搬移资产包。'})


def run(config_path, values, output, records, stop_event):
    started = time.perf_counter()
    summary = {'completed': False, 'termination_reason': 'initializing'}
    frames, actual_path = [], []
    simulation_wall, video_wall = 0., 0.
    reference_at = None
    backend = None
    try:
        dt = 1/float(values['control_hz'])
        scale = float(values['time_scale'])
        trajectory = CatalogTrajectory(values['trajectory_definition'], scale)
        summary.update(trajectory=values['trajectory_definition']['name'],
                       entry_s=trajectory.entry_s, motion_s=trajectory.motion_s)
        write_json(output/'trajectory-coverage.json', values['trajectory_coverage'])
        count = int(round(trajectory.duration_s/dt))
        if count < 1 or not np.isclose(count*dt, trajectory.duration_s, atol=1e-10, rtol=0):
            raise ValueError('轨迹总时间必须是控制周期整数倍。')
        robot_path = (config_path.parent/values['robot_config']).resolve()
        source = (config_path.parent/values['mujoco_xml']).resolve()
        snapshot(output, config_path, values, robot_path, source)
        scene = output/'scene.xml'
        build_executor_scene(source, scene)
        robot = load_robot_model(robot_path)
        backend = ExecutorBackend(scene, video_fps=values['video']['fps'] if values['video']['enabled'] else None)
        substeps = dt/backend.timestep
        if not np.isclose(substeps, round(substeps), atol=1e-10, rtol=0):
            raise ValueError('控制周期必须是 XML 物理步长整数倍。')
        controller = ExecutorTrackingController(robot, backend.root_transform[:3, :3],
            gain=float(values['gain']), damping=float(values['damping']),
            joint_margin_rad=np.deg2rad(values['joint_margin_deg']),
            joint_guard_enabled=values['joint_guard_enabled'])
        feedback = backend.read_feedback()
        controller.reset(feedback)
        summary.update(
            configured_joint_guard_enabled=bool(values['joint_guard_enabled']),
            effective_joint_guard_enabled=controller.joint_guard_enabled,
            joint_margin_rad=controller.joint_margin,
            initial_minimum_joint_margin_rad=feedback.minimum_joint_margin_rad,
        )
        base = backend.base_transform
        initial = base[:3, :3].T@(feedback.tip_world_m-base[:3, 3])
        if not np.allclose(initial, trajectory.entry.points[0], atol=1e-6, rtol=0):
            raise ValueError('仿真初始末端与已检查的入场起点不一致。')
        def reference_at(t):
            p, v = trajectory.sample(t)
            return base[:3, 3]+base[:3, :3]@p, base[:3, :3]@v
        reference_path = np.array([reference_at(t)[0] for t in np.linspace(0, trajectory.duration_s, 201)])
        records.append(0., reference_at(0.)[0], feedback)
        actual_path.append(feedback.tip_world_m.copy())
        if values['video']['enabled']:
            frames.append(backend.snapshot())
        summary.update(control_hz=values['control_hz'], physics_timestep_s=backend.timestep,
            mode=values['mode'], requested_duration_s=trajectory.duration_s,
            public_limits=dict(enabled=robot.limits.enabled, rate_enabled=robot.limits.rate_enabled,
                maximum_rate_mps=robot.limits.maximum_rate_mps.tolist(),
                maximum_curvature_per_m=robot.limits.maximum_curvature_per_m))
        with ExitStack() as stack:
            viewer = None
            if values['viewer']['enabled']:
                import mujoco.viewer
                viewer = stack.enter_context(mujoco.viewer.launch_passive(backend.model, backend.data))
                with viewer.lock():
                    configure_camera(viewer.cam, base[:3, 3])
            simulation_start = time.perf_counter()
            next_display = simulation_start
            reason = 'trajectory_elapsed'
            records.set_status('运行中')
            try:
                for index in range(count):
                    if stop_event.is_set():
                        reason = 'user_stopped'
                        break
                    if viewer is not None and not viewer.is_running():
                        reason = 'viewer_closed'
                        break
                    t = index*dt
                    target, velocity = reference_at(t)
                    clock = time.perf_counter()
                    result = controller.step(feedback, target, velocity, dt)
                    control_wall = time.perf_counter()-clock
                    if result.status != 'tracking':
                        reason = result.status
                        break
                    clock = time.perf_counter()
                    new_frames = backend.step(result.command, dt)
                    physics_wall = time.perf_counter()-clock
                    clock = time.perf_counter()
                    feedback = backend.read_feedback()
                    feedback_wall = time.perf_counter()-clock
                    frames.extend(new_frames)
                    current_time = (index+1)*dt
                    reference = reference_at(current_time)[0]
                    actual_path.append(feedback.tip_world_m.copy())
                    display_wall = 0.
                    wall = time.perf_counter()
                    if viewer is not None and wall >= next_display:
                        clock = time.perf_counter()
                        with viewer.lock():
                            viewer.user_scn.ngeom = 0
                            add_tracking_geometry(backend.mj, viewer.user_scn, reference, feedback.tip_world_m,
                                                  reference_path, actual_path)
                        viewer.sync()
                        display_wall = time.perf_counter()-clock
                        next_display = wall+1/values['viewer']['refresh_hz']
                    records.append(current_time, reference, feedback, result,
                        (control_wall, physics_wall, feedback_wall, display_wall), t, True)
                    elapsed = time.perf_counter()-simulation_start
                    records.set_status('运行中', current_time/max(elapsed, 1e-12))
                    if (controller.joint_guard_enabled
                            and feedback.minimum_joint_margin_rad <= controller.joint_margin):
                        reason = 'joint_guard'
                        break
                    if values['mode'] == 'realtime':
                        remaining = simulation_start+current_time-time.perf_counter()
                        if remaining > 0:
                            stop_event.wait(remaining)
            finally:
                simulation_wall = time.perf_counter()-simulation_start
            summary.update(completed=reason == 'trajectory_elapsed', termination_reason=reason)
            if viewer is not None and viewer.is_running():
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    add_tracking_geometry(backend.mj, viewer.user_scn, reference_at(records.rows[-1][0])[0],
                                          feedback.tip_world_m, reference_path, actual_path)
                viewer.sync()
            records.save(output)
            # 控制已经结束，独立渲染数据实例不会重放或改写 viewer 的物理状态。
            if values['video']['enabled'] and frames:
                records.set_status('控制结束，生成视频')
                clock = time.perf_counter()
                try:
                    export_video(scene, frames, output/'tracking.mp4', values['video']['fps'],
                        values['video']['width'], values['video']['height'], reference_at, reference_path,
                        records.set_status)
                    summary['video_saved'] = True
                except Exception as exc:
                    summary['video_saved'] = False
                    summary['video_error'] = f'{type(exc).__name__}: {exc}'
                finally:
                    video_wall = time.perf_counter()-clock
    except Exception as exc:
        summary.update(completed=False, termination_reason='exception', error=f'{type(exc).__name__}: {exc}')
    finally:
        records.save(output)
        try:
            summary['error_plot_saved'] = save_error_plot(records.rows, output/'tracking-error.svg')
        except Exception as exc:
            summary['error_plot_saved'] = False
            summary['error_plot_error'] = f'{type(exc).__name__}: {exc}'
        summary.update(records.metrics())
        if 'entry_s' in summary and records.rows:
            rows = np.asarray(records.rows)
            for name, mask in [('entry', rows[:, 0] <= summary['entry_s']),
                               ('motion', rows[:, 0] > summary['entry_s'])]:
                errors = rows[mask, COLUMNS.index('error_norm_m')]
                summary[name+'_errors'] = dict(samples=len(errors),
                    rmse_m=float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                    maximum_error_m=float(errors.max()) if len(errors) else None)
        summary.update(simulation_wall_s=simulation_wall, video_wall_s=video_wall,
                       total_wall_s=time.perf_counter()-started, video_frames=len(frames))
        if simulation_wall > 0:
            summary['realtime_factor'] = summary['simulation_time_s']/simulation_wall
        summary['note'] = 'completed 仅表示轨迹时钟走完；不代表误差或性能验收通过。'
        write_json(output/'summary.json', summary)
        final = '完成' if summary['completed'] else '停止：'+summary['termination_reason']
        if summary.get('video_error'):
            final += '；视频保存失败，详见 summary.json'
        records.set_status(final, summary['realtime_factor'])
        print(json.dumps({'output': str(output), **summary}, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/experiments/executor_tracking.yaml'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mode', choices=('max', 'realtime'))
    parser.add_argument('--time-scale', type=float)
    parser.add_argument('--trajectory', help='轨迹目录中的 YAML 名称（不含扩展名）')
    parser.add_argument('--list-trajectories', action='store_true')
    parser.add_argument('--no-viewer', action='store_true')
    parser.add_argument('--no-error-window', action='store_true')
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--headless', action='store_true', help='关闭两个窗口，视频仍遵循配置或 --no-video')
    args = parser.parse_args()
    path = args.config.resolve()
    values = yaml.safe_load(path.read_text(encoding='utf-8'))
    directory = (path.parent/values['trajectory_directory']).resolve()
    available = {p.stem: p for p in sorted(directory.glob('*.yaml'))}
    if args.list_trajectories:
        for name, candidate in available.items():
            try:
                definition, _ = load_approved(candidate, directory)
                print(f"{name}: {definition['kind']}, 入场 {definition['entry_s']} s, 主轨迹 {definition['motion_s']} s")
            except ValueError as exc:
                print(f'{name}: 不可运行：{exc}')
        return
    selected = args.trajectory
    if selected is None:
        if not sys.stdin.isatty():
            parser.error('非交互运行请指定 --trajectory；可用名称：'+', '.join(available))
        names = list(available)
        for index, name in enumerate(names, 1):
            print(f'{index}. {name}')
        selected = input('选择轨迹编号或名称：').strip()
        if selected.isdigit() and 1 <= int(selected) <= len(names):
            selected = names[int(selected)-1]
    if selected not in available:
        parser.error('未知轨迹：'+str(selected))
    try:
        definition, coverage = load_approved(available[selected], directory)
    except ValueError as exc:
        parser.error(str(exc))
    values['trajectory_definition'] = definition
    values['trajectory_coverage'] = coverage
    if args.mode:
        values['mode'] = args.mode
    if args.time_scale is not None:
        values['time_scale'] = args.time_scale
    if args.no_viewer or args.headless:
        values['viewer']['enabled'] = False
    if args.no_error_window or args.headless:
        values['error_window']['enabled'] = False
    if args.no_video:
        values['video']['enabled'] = False
    numbers = [values['control_hz'], values['time_scale'], values['gain'], values['damping'],
               values['joint_margin_deg'], values['viewer']['refresh_hz'], values['error_window']['refresh_hz'],
               values['video']['fps'], values['video']['width'], values['video']['height']]
    if values['mode'] not in ('max', 'realtime') or not all(np.isfinite(x) and x > 0 for x in numbers):
        parser.error('频率、增益、时间倍率、图像尺寸和运行模式无效。')
    if any(type(values[name]['enabled']) is not bool for name in ('viewer', 'error_window', 'video')):
        parser.error('窗口/视频 enabled 必须为布尔值。')
    if any(int(values['video'][name]) != values['video'][name] for name in ('width', 'height')):
        parser.error('视频宽高必须为整数。')
    parent = args.output.resolve() if args.output else (path.parent/values['output_directory']).resolve()
    output = parent/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    print(f'结果目录：{output}', flush=True)
    records, stop_event, result = RunRecords(), Event(), {}
    def work():
        result.update(run(path, values, output, records, stop_event))
    worker = Thread(target=work, name='executor-simulation', daemon=False)
    try:
        if values['error_window']['enabled']:
            from confined_dual_continuum.diagnostics.executor_monitor import ErrorMonitor
            monitor = ErrorMonitor(records, stop_event, refresh_hz=values['error_window']['refresh_hz'])
            monitor.run(worker.start)
        else:
            worker.start()
            while worker.is_alive():
                worker.join(.2)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        if worker.ident is not None:
            worker.join()
    if result.get('termination_reason') == 'exception':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
