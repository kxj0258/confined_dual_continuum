"""20 Hz 单向双臂跟踪：独立从臂求解进程与运行中相机录像。"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty, Full
import shutil
import time
from datetime import datetime, timezone
from uuid import uuid4
from threading import Event, Thread
from collections import Counter
import numpy as np
import yaml
from confined_dual_continuum.actuation.compatible_tendon_control import CompatibleTendonController
from confined_dual_continuum.adapters.executor_observer_backend import DualTrackingBackend, build_scene
from confined_dual_continuum.adapters.observer_workers import solver_worker, video_worker
from confined_dual_continuum.application.executor_tracking import ExecutorTrackingController
from confined_dual_continuum.diagnostics.executor_records import RunRecords
from confined_dual_continuum.diagnostics.executor_error_plot import (
    save_centerline_distance_plot, save_error_plot, save_interarm_plot)
from confined_dual_continuum.domain.state import ArmCommand, RobotCommand
from confined_dual_continuum.robot.configuration import load_robot_model
from confined_dual_continuum.task.catalog_trajectory import CatalogTrajectory
from confined_dual_continuum.trajectory_catalog import load_approved


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def hold(model, target):
    return ArmCommand(np.zeros(model.bending_size), np.zeros(model.tendon_count), target.copy())


def wait_ready(process, ready, timeout=60., stop_event=None):
    deadline = time.perf_counter()+timeout
    while not ready.wait(.05):
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError('用户停止初始化。')
        if not process.is_alive():
            raise RuntimeError(f'{process.name} 初始化失败。')
        if time.perf_counter() >= deadline:
            raise RuntimeError(f'{process.name} 初始化超时。')


def run(config_path, values, definition, coverage, output, records=None, stop_event=None):
    dt = 1.0 / values['control_hz']
    robot_path = (config_path.parent/values['robot_config']).resolve()
    source = (config_path.parent/values['mujoco_xml']).resolve()
    scene = output/'scene.xml'
    build_scene(source, scene)
    shutil.copy2(config_path, output/'source-config.yaml')
    shutil.copy2(robot_path, output/'robot.yaml')
    write_json(output/'trajectory.json', definition)
    write_json(output/'trajectory-coverage.json', coverage)
    (output/'effective-config.yaml').write_text(yaml.safe_dump(values, allow_unicode=True), encoding='utf-8')
    robot = load_robot_model(robot_path)
    trajectory = CatalogTrajectory(definition, values['time_scale'])
    count = int(round(trajectory.duration_s/dt))
    if not np.isclose(count*dt, trajectory.duration_s, atol=1e-10, rtol=0):
        raise ValueError('轨迹时长必须为20 ms整数倍。')
    # spawn 子进程导入 NumPy 前限制 BLAS 线程，避免后台占满 CPU。
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    context = mp.get_context('spawn')
    requests, results = context.Queue(1), context.Queue(1)
    solver_stop, solver_ready = context.Event(), context.Event()
    observer_config = dict(values['observer'], control_dt_s=dt,
                           camera_width=values['video']['width'],
                           camera_height=values['video']['height'])
    solver = context.Process(target=solver_worker, name='observer-solver', args=(
        str(scene), str(robot_path), observer_config, requests, results, solver_ready, solver_stop))
    video_stop, video_ready = context.Event(), context.Event()
    frames = context.Queue(values['video']['queue_size'])
    video = None
    backend = None
    records = records if records is not None else RunRecords()
    stop_event = stop_event if stop_event is not None else Event()
    observer_rows = []
    summary = dict(completed=False, termination_reason='initializing',
        control_hz=values['control_hz'], observer_control_hz=values['control_hz'],
        video_dropped_snapshots=0, deadline_misses=0,
        interarm_contact_response=False)
    started = None
    try:
        backend = DualTrackingBackend(scene, robot)
        backend.reset()
        stride = 1/(values['video']['fps']*backend.model.opt.timestep)
        if not np.isclose(stride, round(stride), atol=1e-9) or stride < 1:
            raise ValueError('视频周期必须为物理步长整数倍。')
        base = backend._site_transform('mobile_base_frame')
        controller = ExecutorTrackingController(robot, backend._site_transform('executor_base_site')[:3, :3],
            gain=values['gain'], damping=values['damping'],
            joint_margin_rad=np.deg2rad(values['joint_margin_deg']),
            joint_guard_enabled=values['joint_guard_enabled'])
        feedback = backend.executor_feedback()
        controller.reset(feedback)
        initial = base[:3, :3].T@(feedback.tip_world_m-base[:3, 3])
        if not np.allclose(initial, trajectory.entry.points[0], atol=1e-6, rtol=0):
            raise ValueError('双臂场景主臂初始位置与轨迹入场起点不一致。')
        actuator = CompatibleTendonController(robot.bending_model('observer'), robot.limits)
        target = np.zeros(actuator.model.tendon_count)
        actuator.reset(target)
        def reference(t):
            p, v = trajectory.sample(t)
            return base[:3, 3]+base[:3, :3]@p, base[:3, :3]@v
        solver.start()
        try:
            wait_ready(solver, solver_ready, stop_event=stop_event)
        except RuntimeError as exc:
            summary['solver_error'] = str(exc)
            solver_stop.set()
            if solver.is_alive():
                solver.terminate()
            solver.join(1.)
        if any(values['video'][k] for k in ('enabled', 'viewer', 'camera_window')):
            display_config = dict(values['video'], trajectory_definition=definition,
                                  time_scale=values['time_scale'])
            video = context.Process(target=video_worker, name='observer-video', args=(
                str(scene), str(robot_path), display_config, str(output), frames, video_ready, video_stop))
            video.start()
            # 录像初始化失败单独记录，不妨碍主臂执行。
            try:
                wait_ready(video, video_ready, stop_event=stop_event)
            except RuntimeError as exc:
                summary['video_error'] = str(exc)
        def frame(snapshot):
            if video is None or not video.is_alive() or not video_ready.is_set():
                return
            try:
                frames.put_nowait(snapshot)
            except Full:
                summary['video_dropped_snapshots'] += 1
        frame(backend.snapshot())
        records.append(0., reference(0.)[0], feedback)
        busy, next_attempt = False, 0.
        pending_prediction = None
        failed_position = failed_qpos = None
        started = time.perf_counter()
        summary['termination_reason'] = 'trajectory_elapsed'
        for index in range(count):
            if stop_event.is_set():
                summary['termination_reason'] = 'user_stopped'
                break
            cycle_started = time.perf_counter()
            t = index*dt
            comparison_prediction = pending_prediction
            ex = controller.step(feedback, *reference(t), dt)
            if ex.status != 'tracking':
                summary['termination_reason'] = ex.status
                break
            ob = hold(actuator.model, target)
            status, reply = 'waiting', None
            try:
                reply = results.get_nowait()
                busy = False
            except Empty:
                pass
            if reply is not None:
                status = reply['status']
                if (reply['index'] != index-1
                        or time.perf_counter()-reply['submitted'] > .075):
                    status = 'stale_result'
                elif reply['rate'] is not None:
                    if (not np.allclose(reply['target'], target, atol=1e-12, rtol=0)
                            or (values['observer']['joint_guard_enabled']
                                and backend.margin('observer') <= np.deg2rad(values['observer']['joint_margin_deg']))):
                        status = 'current_guard'
                    else:
                        try:
                            actual = backend.data.ten_length[backend._tendon_ids['observer']]-backend._neutral['observer']
                            result = actuator.step(reply['rate'], actual, dt, allow_recovery=False)
                            constraint_payload = reply['constraints']
                            a, b = constraint_payload[:2]
                            if result.feasible and np.all(a@result.bending_rate >= b-1e-7):
                                ob = ArmCommand(result.bending_rate, result.tendon_rate_mps, result.target_m)
                                target = result.target_m.copy()
                                payload_diagnostics = (constraint_payload[2]
                                    if isinstance(constraint_payload, (tuple, list))
                                    and len(constraint_payload) > 2
                                    and isinstance(constraint_payload[2], dict)
                                    else {})
                                if 'predicted_min_gap_m' in payload_diagnostics:
                                    pending_prediction = {
                                        'source_index': int(reply['index']),
                                        'predicted_gap_m': float(payload_diagnostics['predicted_min_gap_m']),
                                    }
                            else:
                                actuator.cancel_step(result, dt)
                                status = 'final_command_rejected'
                        except Exception as exc:
                            # 从臂异常仅撤销从臂积分，绝不改变已生成的主臂命令。
                            actuator.reset(target)
                            status = 'observer_exception'
                            reply['error'] = f'{type(exc).__name__}: {exc}'
                if status != 'tracking':
                    next_attempt = t+values['observer']['failure_cooldown_s']
                    failed_position = feedback.tip_world_m.copy()
                    failed_qpos = backend.data.qpos.copy()
                else:
                    failed_position = failed_qpos = None
            changed = (values['observer']['failure_cooldown_s'] <= 0.0
                or failed_position is None
                or np.linalg.norm(feedback.tip_world_m-failed_position) >= .0005
                or np.max(np.abs(backend.data.qpos-failed_qpos)) >= .002)
            if not solver.is_alive():
                status = 'worker_stopped'
            elif not busy and t+1e-12 >= next_attempt and changed:
                # 本周期目标尚未发送，下一周期只有实际提交后的目标才能接受候选结果。
                try:
                    requests.put_nowait((index, time.perf_counter(), backend.snapshot(), target.copy(),
                                         ex.command.bending_rate.copy()))
                    busy = True
                except Full:
                    pass
            elif not busy and reply is None:
                status = 'failure_cooldown'
            control_wall = time.perf_counter()-cycle_started
            physics_started = time.perf_counter()
            backend.advance(RobotCommand(np.zeros(6), ex.command, ob), dt,
                frame if video is not None else None, int(round(stride)))
            physics_wall = time.perf_counter()-physics_started
            feedback_started = time.perf_counter()
            feedback = backend.executor_feedback()
            feedback_wall = time.perf_counter()-feedback_started
            records.append(t+dt, reference(t+dt)[0], feedback, ex,
                           (control_wall, physics_wall, feedback_wall, 0.), t, True)
            records.set_status('运行中', (t+dt)/max(time.perf_counter()-started, 1e-12))
            constraint_payload = reply.get('constraints') if reply else None
            observer_diagnostics = (
                constraint_payload[2]
                if isinstance(constraint_payload, (tuple, list))
                and len(constraint_payload) > 2
                and isinstance(constraint_payload[2], dict)
                else {})
            actual_gap_m = (reply.get('minimum_interarm_clearance_m')
                            if reply else None)
            prediction_error_m = None
            prediction_source_index = None
            if (actual_gap_m is not None and comparison_prediction is not None
                    and reply is not None
                    and int(reply['index']) == comparison_prediction['source_index'] + 1):
                prediction_source_index = comparison_prediction['source_index']
                prediction_error_m = float(actual_gap_m - comparison_prediction['predicted_gap_m'])
            observer_rows.append(dict(time_s=t, status=status,
                solver_status=reply['status'] if reply else None,
                deadline_stage=reply.get('deadline_stage') if reply else None,
                stage_wall_s=reply.get('stage_wall_s') if reply else None,
                request_wall_s=reply.get('request_wall_s') if reply else None,
                source_index=reply['index'] if reply else None,
                solve_wall_s=reply.get('wall_s') if reply else None,
                minimum_interarm_clearance_m=reply.get('minimum_interarm_clearance_m') if reply else None,
                visible=reply.get('visible') if reply else None,
                occluded=reply.get('occluded') if reply else None,
                target_camera_m=reply.get('target_camera_m') if reply else None,
                error=reply.get('error') if reply else None,
                tendon_target_m=target.tolist(), tendon_rate_mps=ob.tendon_rate_mps.tolist(),
                compatibility_residual=float(np.linalg.norm(actuator.model.residual(ob.tendon_rate_mps))),
                prediction_source_index=prediction_source_index,
                prediction_source_time_s=(prediction_source_index * dt
                                          if prediction_source_index is not None else None),
                prediction_target_time_s=((prediction_source_index + 1) * dt
                                          if prediction_source_index is not None else None),
                prediction_actual_gap_m=actual_gap_m,
                prediction_error_m=prediction_error_m,
                **observer_diagnostics))
            records.set_interarm_clearance(
                t, reply.get('minimum_interarm_clearance_m') if reply else None)
            if (controller.joint_guard_enabled
                    and feedback.minimum_joint_margin_rad <= controller.joint_margin):
                summary['termination_reason'] = 'joint_guard'
                break
            remaining = started+(index+1)*dt-time.perf_counter()
            if remaining > 0:
                stop_event.wait(remaining)
            else:
                summary['deadline_misses'] += 1
        summary['completed'] = summary['termination_reason'] == 'trajectory_elapsed'
    except (KeyboardInterrupt, InterruptedError):
        summary['termination_reason'] = 'user_stopped'
    except Exception as exc:
        summary.update(termination_reason='exception', error=f'{type(exc).__name__}: {exc}')
    finally:
        records.set_status('控制结束，保存记录与视频')
        summary['simulation_wall_s'] = time.perf_counter()-started if started is not None else 0.
        if backend is not None:
            backend.stop()
            backend.close()
        solver_stop.set()
        if solver.pid is not None:
            solver.join(1.)
            if solver.is_alive():
                solver.terminate()
                solver.join(1.)
        if video is not None and video.pid is not None:
            # 主臂已结束，可以等待排空视频队列；控制期间从不等待编码。
            try:
                if backend is not None and video.is_alive():
                    frames.put(backend.snapshot(), timeout=1.)
                frames.put(None, timeout=1.)
            except Full:
                video_stop.set()
            video.join(30.)
            if video.is_alive():
                video.terminate()
                video.join(1.)
                summary['video_error'] = '视频收尾超时，文件可能不完整。'
        for queue in (requests, results, frames):
            queue.cancel_join_thread()
            queue.close()
        records.save(output)
        with (output/'observer.jsonl').open('w', encoding='utf-8') as stream:
            for row in observer_rows:
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
        try:
            summary['error_plot_saved'] = save_error_plot(records.rows, output/'tracking-error.svg')
        except Exception as exc:
            summary['error_plot_error'] = str(exc)
        try:
            summary['interarm_plot_saved'] = save_interarm_plot(
                records.interarm_snapshot(), output/'interarm-clearance.svg',
                trigger_m=values['observer']['avoidance_trigger_m'],
                release_m=values['observer']['avoidance_release_m'])
        except Exception as exc:
            summary['interarm_plot_error'] = str(exc)
        try:
            summary['centerline_distance_plot_saved'] = save_centerline_distance_plot(
                records.interarm_snapshot(), output/'interarm-centerline-distance.svg',
                trigger_m=values['observer']['avoidance_trigger_m'],
                release_m=values['observer']['avoidance_release_m'],
                diameter_m=values['observer']['diameter_m'])
        except Exception as exc:
            summary['centerline_distance_plot_error'] = str(exc)
        summary.update(records.metrics())
        summary['observer_status_counts'] = dict(Counter(row['status'] for row in observer_rows))
        summary['observer_solver_status_counts'] = dict(Counter(
            row['solver_status'] for row in observer_rows if row['solver_status'] is not None))
        summary['observer_deadline_stage_counts'] = dict(Counter(
            row['deadline_stage'] for row in observer_rows if row['deadline_stage'] is not None))
        summary['observer_accepted_commands'] = sum(row['status'] == 'tracking' for row in observer_rows)
        summary['observer_moving_commands'] = sum(
            any(abs(v) > 1e-12 for v in row['tendon_rate_mps']) for row in observer_rows)
        stage_names = {name for row in observer_rows for name in (row['stage_wall_s'] or {})}
        summary['observer_stage_wall_s'] = {}
        for name in sorted(stage_names):
            durations = np.array([row['stage_wall_s'][name] for row in observer_rows
                                  if row['stage_wall_s'] is not None and name in row['stage_wall_s']])
            summary['observer_stage_wall_s'][name] = dict(samples=len(durations),
                mean=float(np.mean(durations)), p95=float(np.percentile(durations, 95)),
                maximum=float(np.max(durations)))
        if summary['simulation_wall_s']:
            summary['realtime_factor'] = summary['simulation_time_s']/summary['simulation_wall_s']
        if (output/'video_summary.json').exists():
            summary['video'] = json.loads((output/'video_summary.json').read_text(encoding='utf-8'))
        write_json(output/'summary.json', summary)
        records.set_status('完成' if summary['completed'] else '停止：'+summary['termination_reason'],
                           summary.get('realtime_factor', 0.))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/experiments/executor_observer_tracking.yaml'))
    parser.add_argument('--trajectory')
    parser.add_argument('--list-trajectories', action='store_true')
    parser.add_argument('--time-scale', type=float)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--no-viewer', action='store_true')
    parser.add_argument('--no-error-window', action='store_true')
    parser.add_argument('--no-camera-window', action='store_true')
    parser.add_argument('--no-video', action='store_true')
    args = parser.parse_args()
    path = args.config.resolve()
    values = yaml.safe_load(path.read_text(encoding='utf-8'))
    directory = (path.parent/values['trajectory_directory']).resolve()
    available = {p.stem: p for p in sorted(directory.glob('*.yaml'))}
    if args.list_trajectories:
        print('\n'.join(available))
        return
    selected = args.trajectory
    if selected is None:
        import sys
        if not sys.stdin.isatty():
            parser.error('非交互运行请指定 --trajectory。')
        names = list(available)
        for i, name in enumerate(names, 1):
            print(f'{i}. {name}')
        selected = input('选择轨迹编号或名称：').strip()
        if selected.isdigit() and 1 <= int(selected) <= len(names):
            selected = names[int(selected)-1]
    if selected not in available:
        parser.error('未知轨迹。')
    definition, coverage = load_approved(available[selected], directory)
    if args.time_scale is not None:
        values['time_scale'] = args.time_scale
    if args.headless or args.no_viewer:
        values['video']['viewer'] = False
    if args.headless or args.no_camera_window:
        values['video']['camera_window'] = False
    if args.headless or args.no_error_window:
        values['error_window']['enabled'] = False
    if args.no_video:
        values['video']['enabled'] = False
    numeric = [values[k] for k in ('time_scale', 'gain', 'damping', 'joint_margin_deg')]
    observer_numeric = [value for key, value in values['observer'].items()
                        if key not in ('failure_cooldown_s', 'joint_guard_enabled')]
    numeric += observer_numeric
    numeric += [values['video'][k] for k in ('fps', 'width', 'height', 'queue_size')]
    numeric.append(values['error_window']['refresh_hz'])
    if (values['control_hz'] != 20
            or not all(np.isfinite(v) and v > 0 for v in numeric)
            or not np.isfinite(values['observer']['failure_cooldown_s'])
            or values['observer']['failure_cooldown_s'] < 0):
        parser.error('双臂固定20 Hz，其余数值参数必须为正有限数。')
    if values['observer']['budget_ms'] >= 50:
        parser.error('从臂预算必须小于50 ms。')
    for value in (values['observer']['iterations'], *(values['video'][k] for k in ('width', 'height', 'queue_size'))):
        if type(value) is not int:
            parser.error('迭代次数、分辨率和队列长度必须为整数。')
    if (any(type(values['video'][k]) is not bool for k in ('enabled', 'viewer', 'camera_window'))
            or type(values['error_window']['enabled']) is not bool
            or type(values['joint_guard_enabled']) is not bool
            or type(values['observer']['joint_guard_enabled']) is not bool):
        parser.error('视频及窗口开关必须为布尔值。')
    parent = args.output.resolve() if args.output else (path.parent/values['output_directory']).resolve()
    output = parent/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    print(f'输出目录：{output}', flush=True)
    records, stop_event, summary = RunRecords(), Event(), {}
    def work():
        try:
            summary.update(run(path, values, definition, coverage, output, records, stop_event))
        except Exception as exc:
            summary.update(completed=False, termination_reason='exception', error=str(exc))
            records.set_status('初始化失败：'+str(exc))
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    worker = Thread(target=work, name='dual-arm-simulation', daemon=False)
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
    if summary.get('termination_reason') == 'exception':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
