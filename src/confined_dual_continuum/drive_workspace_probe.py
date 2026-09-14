"""固定基座的加速度轨迹记录及稳定工作空间采集；不设跟踪误差合格阈值。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import csv
import json
import hashlib
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import yaml

from confined_dual_continuum.actuation.compatible_tendon_control import CompatibleTendonController
from confined_dual_continuum.adapters.mujoco_backend import MujocoRobotBackend
from confined_dual_continuum.adapters.robot_only_scene import robot_only_scene
from confined_dual_continuum.diagnostics.drive_metrics import DriveEvents, TrapezoidProfile, tracking_statistics
from confined_dual_continuum.domain.state import ArmCommand, RobotCommand
from confined_dual_continuum.response_probe import _snapshot, _json
from confined_dual_continuum.robot.configuration import load_robot_model
from confined_dual_continuum.workspace_probe import references, compare


@contextmanager
def collision_scope(backend, arm, scope):
    """两种口径都排除臂间接触；机械口径另外隔离环境，保留自身碰撞。"""
    if scope not in ("scene", "mechanical"):
        raise ValueError("未知空间口径")
    types = backend.model.geom_contype.copy()
    affinity = backend.model.geom_conaffinity.copy()
    try:
        # 环境位 1、Executor 位 2、Observer 位 4；双臂不接受对方位。
        groups = {a: set(backend._collision_geoms[a]) for a in ("executor", "observer")}
        for i in range(backend.model.ngeom):
            if types[i] or affinity[i]:
                bit = 2 if i in groups["executor"] else 4 if i in groups["observer"] else 1
                backend.model.geom_contype[i] = bit
                backend.model.geom_conaffinity[i] = 7 if bit == 1 else bit | 1
        if scope == "mechanical":
            active = set(backend._collision_geoms[arm])
            for i in range(backend.model.ngeom):
                if i not in active:
                    backend.model.geom_contype[i] = 0
                    backend.model.geom_conaffinity[i] = 0
        yield
    finally:
        backend.model.geom_contype[:] = types
        backend.model.geom_conaffinity[:] = affinity


def workspace_targets(config):
    """零位/逐轴基准，加 Halton 内部点及一、二、三段曲率边界分层。"""
    yield from references(dict(config, combination_samples=0))
    maximum = max(config["amplitudes_per_m"])
    shift = np.random.default_rng(config["seed"]).uniform(0., 1., 6)

    def radical_inverse(index, base):
        value, fraction = 0., 1.
        while index:
            index, digit = divmod(index, base)
            fraction /= base
            value += digit * fraction
        return value

    # 固定种子的平移保持可复现；每段半径用 sqrt(u) 得到面积均匀圆盘。
    strata = [((), config["combination_samples"])]
    strata += [((i,), config["single_boundary_samples_per_group"]) for i in range(3)]
    strata += [(pair, config["double_boundary_samples_per_group"]) for pair in ((0, 1), (0, 2), (1, 2))]
    strata += [((0, 1, 2), config["boundary_samples"])]
    index = 1
    for boundary, count in strata:
        for _ in range(count):
            u = (np.array([radical_inverse(index, base) for base in (2, 3, 5, 7, 11, 13)]) + shift) % 1.
            index += 1
            radii = maximum * np.sqrt(u[:3])
            radii[list(boundary)] = maximum
            angles = 2*np.pi*u[3:]
            yield np.column_stack((radii*np.cos(angles), radii*np.sin(angles))).ravel()


def dynamic_targets(config):
    for axis in config["axes"]:
        q = np.zeros(6)
        q[axis] = config["amplitude_per_m"]
        yield q
    # 同一方向用于全部速度与两臂；避免方向随机变化混淆速度对比。
    rng = np.random.default_rng(config["seed"])
    for _ in range(config["combination_samples"]):
        angles = rng.uniform(0., 2*np.pi, 3)
        yield (config["amplitude_per_m"] * np.column_stack((np.cos(angles), np.sin(angles)))).ravel()


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _quiet(measurement, previous, dt, config, arms=("executor", "observer")):
    blockers = []
    for arm in arms:
        m = measurement[arm]
        if np.max(np.abs(m["joint_velocity_radps"])) > config["joint_speed_tolerance_radps"]:
            blockers.append(arm + ":joint_speed")
        if np.max(np.abs(m["tendon_velocity_mps"])) > config["tendon_speed_tolerance_mps"]:
            blockers.append(arm + ":tendon_speed")
        velocity = (np.asarray(m["T_world_tip"])[:3, 3] - np.asarray(previous[arm]["T_world_tip"])[:3, 3]) / dt
        if np.linalg.norm(velocity) > config["tip_speed_tolerance_mps"]:
            blockers.append(arm + ":tip_speed")
    return blockers


def _self_contact(backend, arm):
    active = set(backend._collision_geoms[arm])
    return any(int(c.geom1) in active and int(c.geom2) in active and c.dist <= 0
               for c in backend.data.contact[:backend.data.ncon])


def _scene_stop(backend, measurement, config):
    """原生胶囊距离与中心线减半径的判据一致；避免采样热路径反复求 Python 最近点。"""
    for arm in ("executor", "observer"):
        for a in backend._collision_geoms[arm]:
            for b in backend._environment_geoms:
                if backend._mj.mj_geomDistance(backend.model, backend.data, a, b, .05, None) <= config["minimum_clearance_m"]:
                    return "environment_collision_margin"
    return None


def run_path(backend, robot, arm, scope, waypoints, speed, config, trace, *,
             command_profile="step", acceleration_time=None, dwell_s=0., joint_margin_rad=0., viewer=None):
    """固定基座；每段停止并实际稳定后继续，记录误差但不按误差拒绝样本。"""
    with viewer.lock() if viewer else nullcontext():
        state = backend.reset()
    arms = ("executor", "observer")
    models = {a: robot.bending_model(a) for a in arms}
    actuators = {a: CompatibleTendonController(models[a], robot.limits) for a in arms}
    targets = {}
    for a in arms:
        actuators[a].reset(getattr(state, a).tendon_displacement_m)
        targets[a] = models[a].project(getattr(state, a).tendon_displacement_m)
    measurement = backend.capture_response_measurement()
    dt = config["control_period_s"]
    records, legs = [], []
    previous_requested = np.zeros(models[arm].tendon_count)
    previous_sent = previous_requested.copy()
    previous_actual = np.asarray(measurement[arm]["tendon_velocity_mps"])
    stop = None
    joint_ids = [backend._id(backend._mj.mjtObj.mjOBJ_JOINT, name)
                 for name in measurement[arm]["joint_names"]]
    for leg, desired in enumerate(waypoints):
        events = DriveEvents(config["stable_hold_s"], dwell_s)
        started = measurement["time_s"]
        start_q = models[arm].estimate(targets[arm])
        delta_q = desired - start_q
        delta_l = models[arm].to_tendon(delta_q)
        distance = float(np.max(np.abs(delta_l)))
        profile = None
        if command_profile == "trapezoid":
            if acceleration_time is None or not np.isfinite(acceleration_time) or acceleration_time <= 0:
                raise ValueError("动态轨迹的加速时间必须为正有限数")
            # 短行程同时降低峰值及加速度，保持加速、减速各为指定时间。
            effective_speed = min(speed, distance / acceleration_time) if distance else speed
            profile = TrapezoidProfile(distance, effective_speed, effective_speed / acceleration_time)
        motion_s = profile.duration_s if profile else dt
        if not profile and (robot.limits.enabled or robot.limits.rate_enabled):
            motion_s = max(dt, float(np.max(np.abs(delta_l) / robot.limits.maximum_rate_mps)))
        # 按实际行程给足运动时间；超时只限制发完后等待稳定的时间。
        duration_limit = motion_s + config["maximum_settle_s"] + config["stable_hold_s"] + dwell_s
        status, blockers = "settle_timeout", []
        leg_records = []
        minimum_margin = None
        for step in range(int(np.ceil(duration_limit / dt))):
            frame_started = time.perf_counter()
            if viewer and not viewer.is_running():
                raise KeyboardInterrupt("viewer 已关闭，保存已完成样本")
            stop = _scene_stop(backend, measurement, config) if scope == "scene" else None
            if stop:
                status = stop
                break
            elapsed = step * dt
            if profile:
                position, _, _, _ = profile.sample(elapsed + dt)
                _, nominal_v, nominal_a, phase = profile.sample(elapsed + dt/2)
                desired_now = start_q + delta_q * (position / distance) if distance else desired
                direction = delta_l / distance if distance else np.zeros_like(delta_l)
            else:
                desired_now = desired
                nominal_v = nominal_a = 0.
                direction = np.zeros_like(delta_l)
                phase = "step" if step == 0 else "hold"
            rates, results = {}, {}
            for a in arms:
                target = desired_now if a == arm else np.zeros(6)
                rates[a] = (target - models[a].estimate(targets[a])) / dt
                results[a] = actuators[a].step(rates[a], getattr(state, a).tendon_displacement_m, dt,
                                               allow_recovery=(a == arm))
            if not all(r.feasible for r in results.values()):
                for a, r in results.items():
                    actuators[a].cancel_step(r, dt)
                stop = status = "infeasible:" + ";".join(a+":"+r.constraint_status for a, r in results.items() if not r.feasible)
                break
            commands = {a: ArmCommand(r.bending_rate, r.tendon_rate_mps, r.target_m) for a, r in results.items()}
            with viewer.lock() if viewer else nullcontext():
                backend.apply_command(RobotCommand(np.zeros(6), commands["executor"], commands["observer"]))
                backend.step(dt)
            state = backend.read_state()
            previous, measurement = measurement, backend.capture_response_measurement()
            targets = {a: r.target_m for a, r in results.items()}
            active = measurement[arm]
            stop = _scene_stop(backend, measurement, config) if scope == "scene" else None
            if _self_contact(backend, arm):
                stop = "self_collision"
            margins = [min(p-backend.model.jnt_range[j, 0], backend.model.jnt_range[j, 1]-p)
                       for j, p in zip(joint_ids, active["joint_position_rad"]) if backend.model.jnt_limited[j]]
            margin = min(margins) if margins else None
            if margin is not None:
                minimum_margin = margin if minimum_margin is None else min(minimum_margin, margin)
                if joint_margin_rad and margin <= joint_margin_rad:
                    stop = "joint_margin"
            if any(active["joint_at_limit"]):
                stop = "joint_limit"
            sent = elapsed + dt + 1e-12 >= motion_s and max(
                float(np.max(np.abs(models[a].estimate(targets[a]) - (desired if a == arm else np.zeros(6)))))
                for a in arms) <= config["target_tolerance_per_m"]
            blockers = _quiet(measurement, previous, dt, config, (arm,))
            events.update(measurement["time_s"]-started, dt, sent, not blockers, not stop)
            r = results[arm]
            requested = models[arm].to_tendon(rates[arm])
            sent_velocity = r.tendon_rate_mps
            # 与发送的区间平均速度对齐；加速度均用相邻区间平均速度差分。
            actual = (np.asarray(active["actual_tendon_m"]) - np.asarray(previous[arm]["actual_tendon_m"])) / dt
            row = {"time_s": measurement["time_s"], "velocity_time_s": measurement["time_s"]-dt/2,
                   "acceleration_time_s": measurement["time_s"]-dt,
                   "leg": leg, "phase": phase,
                   "requested_velocity_mps": requested.tolist(), "sent_velocity_mps": sent_velocity.tolist(),
                   "actual_velocity_mps": actual.tolist(),
                   "requested_acceleration_mps2": ((requested-previous_requested)/dt).tolist(),
                   "sent_acceleration_mps2": ((sent_velocity-previous_sent)/dt).tolist(),
                   "actual_acceleration_mps2": ((actual-previous_actual)/dt).tolist(),
                   "analytic_velocity_mps": (direction*nominal_v).tolist(),
                   "analytic_acceleration_mps2": (direction*nominal_a).tolist(),
                   "sensor_velocity_mps": active["tendon_velocity_mps"],
                   "sensor_acceleration_mps2": ((np.asarray(active["tendon_velocity_mps"])-
                       np.asarray(previous[arm]["tendon_velocity_mps"]))/dt).tolist(),
                   "length_error_m": float(np.max(np.abs(targets[arm]-active["actual_tendon_m"]))),
                   "joint_margin_rad": margin, "force_peak_n": float(np.max(np.abs(active["actuator_force_n"]))),
                   "scale": r.common_scale, "constraint": r.constraint_status}
            previous_requested, previous_sent, previous_actual = requested.copy(), sent_velocity.copy(), actual.copy()
            records.append(row)
            leg_records.append(row)
            trace.write(json.dumps({**row, "reference_bending_per_m": desired.tolist(),
                                   "measurement": measurement, "events": events.as_dict(),
                                   "quiet_blockers": blockers, "stop": stop}, allow_nan=False)+"\n")
            if viewer:
                viewer.sync()
                time.sleep(max(0., dt-(time.perf_counter()-frame_started)))
            if stop:
                status = stop
                break
            if events.ready:
                status = "stable"
                break
        legs.append({"leg": leg, "status": status, "duration_s": measurement["time_s"]-started,
                     **events.as_dict(), "quiet_blockers": blockers,
                     "final_length_error_m": float(np.max(np.abs(targets[arm]-measurement[arm]["actual_tendon_m"]))),
                     "reference_bending_per_m": desired.tolist(), "minimum_joint_margin_rad": minimum_margin,
                     "planned_motion_s": motion_s,
                     "planned_peak_speed_mps": profile.peak_speed_mps if profile else None,
                     "planned_acceleration_time_s": profile.acceleration_time_s if profile else None,
                     "planned_acceleration_mps2": profile.acceleration_mps2 if profile else None,
                     "planned_cruise_s": profile.cruise_time_s if profile else None,
                     "tracking": tracking_statistics(leg_records)})
        if stop or status != "stable":
            break
    backend.stop()
    active = measurement[arm]
    root_tip = np.linalg.inv(np.asarray(active["T_world_root"])) @ np.asarray(active["T_world_tip"])
    comparison = compare(robot, dict(active, arm=arm), waypoints[len(legs)-1])
    return {"status": legs[-1]["status"], "legs": legs,
            "path_completed": len(legs) == len(waypoints) and all(l["status"] == "stable" for l in legs),
            "command_profile": command_profile, "T_root_tip": root_tip.tolist(),
            "T_world_tip": active["T_world_tip"], "measurement": measurement, "comparison": comparison,
            "measurement_source": "mujoco_geometric_tendon_length_not_motor_encoder",
            "acceleration_source": "finite_difference_of_interval_mean_geometric_tendon_velocity",
            "peak_length_error_m": max((r["length_error_m"] for r in records), default=None),
            "peak_actual_speed_mps": max((max(abs(v) for v in r["actual_velocity_mps"]) for r in records), default=None),
            "peak_actual_acceleration_mps2": max((max(abs(v) for v in r["actual_acceleration_mps2"]) for r in records), default=None),
            "peak_force_n": max((r["force_peak_n"] for r in records), default=None),
            "tracking": tracking_statistics(records)}


def summary(output, rows, config, completed):
    groups = []
    for scope in config["scopes"]:
        for arm in config["arms"]:
            subset = [r for r in rows if r["kind"] == "workspace" and r["scope"] == scope and r["arm"] == arm]
            stable = [r for r in subset if r["status"] == "stable"]
            xyz = np.array([np.asarray(r["T_root_tip"])[:3, 3] for r in stable])
            groups.append({"scope": scope, "arm": arm, "sampled": len(subset), "stable": len(stable),
                           "status_counts": {s: sum(r["status"] == s for r in subset) for s in sorted({r["status"] for r in subset})},
                           "bounds_root_m": {"min": xyz.min(axis=0).tolist(), "max": xyz.max(axis=0).tolist()} if stable else None})
    _json(output/"summary.json", {"completed": completed, "workspace": groups,
          "protocol_version": config["protocol_version"], "scene": config["scene"],
          "criteria": config["criteria"], "sampling": config["workspace"], "dynamic": config["dynamic"],
          "scope": "仅双臂与基座，固定基座、隔离双臂碰撞。记录稳定状态和误差，不作精度或速度合格判定。"})
    flat = []
    for r in rows:
        t = np.asarray(r["T_root_tip"])
        item = {k: r[k] for k in ("kind", "scope", "arm", "sample", "speed_mps", "acceleration_mps2", "status",
                "path_completed", "command_profile", "peak_length_error_m", "peak_actual_speed_mps",
                "peak_actual_acceleration_mps2", "peak_force_n")}
        item.update(x_m=t[0,3], y_m=t[1,3], z_m=t[2,3])
        item.update({f"r{i}{j}": t[i,j] for i in range(3) for j in range(3)})
        for name in ("command", "estimated"):
            for key in ("tip_error_m", "orientation_error_deg", "centerline_rms_m", "centerline_max_m"):
                item[name+"_"+key] = r["comparison"][name][key]
        flat.append(item)
    _write_csv(output/"samples.csv", flat)
    _write_csv(output/"stable_workspace.csv", [r for r in flat if r["kind"] == "workspace" and r["status"] == "stable"])


def validate(config):
    expected = {"control_period_s", "maximum_settle_s", "stable_hold_s", "target_tolerance_per_m",
                "joint_speed_tolerance_radps", "tendon_speed_tolerance_mps", "tip_speed_tolerance_mps",
                "minimum_clearance_m"}
    if set(config["criteria"]) != expected:
        raise ValueError("criteria 必须使用协议 5 字段；旧版长度、相对速度验收阈值已移除")
    if not config["arms"] or len(set(config["arms"])) != len(config["arms"]) or any(a not in ("executor", "observer") for a in config["arms"]):
        raise ValueError("arms 无效")
    if not config["scopes"] or len(set(config["scopes"])) != len(config["scopes"]) or any(s not in ("scene", "mechanical") for s in config["scopes"]):
        raise ValueError("scopes 无效")
    for k, v in config["criteria"].items():
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"{k} 必须为正有限数")
    w, d = config["workspace"], config["dynamic"]
    for values in (w["amplitudes_per_m"], d["speeds_mps"],
                   [d["amplitude_per_m"], d["acceleration_time_s"], d["dwell_s"], d["joint_margin_deg"]]):
        if not values or any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError("幅值和速度必须为正有限数")
    for k in ("combination_samples", "boundary_samples", "single_boundary_samples_per_group", "double_boundary_samples_per_group"):
        if type(w[k]) is not int or w[k] < 0:
            raise ValueError("采样数必须为非负整数")
    if not d["axes"] or len(set(d["axes"])) != len(d["axes"]) or any(type(i) is not int or not 0 <= i < 6 for i in d["axes"]):
        raise ValueError("动态测试轴无效")
    if type(d["combination_samples"]) is not int or d["combination_samples"] < 0:
        raise ValueError("动态组合采样数必须为非负整数")


def _worker_init(robot_path, xml_path, feedback):
    global _worker_robot, _worker_backend
    _worker_robot = load_robot_model(robot_path)
    _worker_backend = MujocoRobotBackend(xml_path, _worker_robot, require_wiping_board=False, **feedback)


def _worker_job(job):
    return _run_job(_worker_backend, _worker_robot, job)


def _run_job(backend, robot, job, viewer=None):
    index, kind, scope, arm, speed, points, criteria, output, dynamic = job
    with collision_scope(backend, arm, scope), (Path(output)/f"{index:04d}_trace.jsonl").open("w", encoding="utf-8") as trace:
        row = run_path(backend, robot, arm, scope, points, speed, criteria, trace,
                       command_profile="trapezoid" if kind == "dynamic" else "step",
                       acceleration_time=dynamic["acceleration_time_s"] if kind == "dynamic" else None,
                       dwell_s=dynamic["dwell_s"] if kind != "workspace" else 0.,
                       joint_margin_rad=np.radians(dynamic["joint_margin_deg"]) if kind != "workspace" else 0., viewer=viewer)
    row.update(kind=kind, scope=scope, arm=arm, sample=index, speed_mps=speed,
               acceleration_mps2=speed / dynamic["acceleration_time_s"] if kind == "dynamic" else None,
               acceleration_time_s=dynamic["acceleration_time_s"] if kind == "dynamic" else None)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/drive_workspace_probe.yaml")
    parser.add_argument("--mode", choices=("all", "dynamic", "workspace", "response"), default="all")
    parser.add_argument("--workers", type=int, default=1, help="独立仿真实例数，默认 1")
    parser.add_argument("--speed-mps", type=float, action="append", help="仅动态模式：覆盖绳速列表，可多次传入")
    parser.add_argument("--resume", type=Path, help="恢复已有未完成目录；仅跳过完整写入的样本")
    parser.add_argument("--viewer", action="store_true", help="主进程实时显示，要求 workers=1；关闭窗口停止采集")
    parser.add_argument("--hold-s", type=float, help="dynamic/response：稳定确认后继续静止保持的秒数")
    parser.add_argument("--acceleration-time-s", type=float, help="仅 dynamic：加速和减速各自持续的时间，秒")
    args = parser.parse_args()
    path = Path(args.config).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["protocol_version"] = 5
    config["scene"] = "robot_only"
    if args.hold_s is not None:
        if args.mode not in ("dynamic", "response"):
            parser.error("--hold-s 仅用于 dynamic/response")
        config["dynamic"]["dwell_s"] = args.hold_s
    if args.acceleration_time_s is not None:
        if args.mode != "dynamic":
            parser.error("--acceleration-time-s 仅用于 dynamic")
        config["dynamic"]["acceleration_time_s"] = args.acceleration_time_s
    if args.viewer and args.workers != 1:
        parser.error("--viewer 要求 --workers 1")
    if args.speed_mps is not None:
        if args.mode != "dynamic":
            raise ValueError("speed-mps 仅用于独立 dynamic 模式")
        config["dynamic"]["speeds_mps"] = args.speed_mps
    validate(config)
    config["requested_mode"] = args.mode
    if args.workers < 1 or args.workers > 8:
        raise ValueError("workers 必须为 1～8")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid4().hex[:8]
    rows = []
    if args.resume:
        output = args.resume.resolve()
        original = yaml.safe_load((output/"snapshot"/"config.yaml").read_text(encoding="utf-8"))
        if original.get("protocol_version") != config["protocol_version"]:
            raise ValueError("采集协议已改变，不能接续旧版结果；请新建运行")
        for key in ("arms", "scopes", "feedback", "workspace"):
            if original[key] != config[key]:
                raise ValueError(f"恢复时 {key} 不能改变")
        if args.mode != "workspace":
            raise ValueError("恢复当前仅支持独立 workspace 采集")
        for key, value in original["criteria"].items():
            if config["criteria"][key] != value:
                raise ValueError(f"恢复时判据 {key} 不能改变")
        rows = [json.loads(line) for line in (output/"samples.jsonl").read_text(encoding="utf-8").splitlines()]
        if any(r["sample"] != i or r["kind"] != "workspace" for i, r in enumerate(rows)):
            raise ValueError("已有样本不是连续的 workspace 记录")
        for current, frozen in ((path.parent/config["robot_config"], output/"snapshot"/"robot.yaml"),
                                (path.parent/config["mujoco_xml"], output/"snapshot"/"assets"/"mujoco"/Path(config["mujoco_xml"]).name)):
            if hashlib.sha256(current.read_bytes()).digest() != hashlib.sha256(frozen.read_bytes()).digest():
                raise ValueError("恢复时机器人参数和 XML 必须与原快照完全相同")
        snapshot_root = output/"continuations"/stamp
        snapshot_root.mkdir(parents=True)
    else:
        output = (path.parent/config["output_directory"]).resolve()/stamp
        output.mkdir(parents=True)
        snapshot_root = output
    robot_path, xml_path = _snapshot(snapshot_root, path, config)
    robot_path = snapshot_root/"snapshot"/"robot.yaml"
    xml_path = snapshot_root/"snapshot"/"assets"/xml_path.parent.name/xml_path.name
    # 与 playground 共用场景裁剪；原 XML 留在快照中供溯源及续采对比。
    scene_path = snapshot_root/"dual_arm_only.xml"
    robot_only_scene(xml_path, scene_path)
    _json(snapshot_root/"scene.json", {"scene": "robot_only", "source": str(xml_path),
          "generated": scene_path.name, "sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest()})
    xml_path = scene_path
    robot = load_robot_model(robot_path)
    backend = MujocoRobotBackend(xml_path, robot, require_wiping_board=False, **config.get("feedback", {}))
    completed = False
    print(f"结果目录：{output}", flush=True)
    try:
        jobs = []
        if args.mode in ("all", "dynamic"):
            for arm in config["arms"]:
                for speed in config["dynamic"]["speeds_mps"]:
                    for q in dynamic_targets(config["dynamic"]):
                        jobs.append(("dynamic", "mechanical", arm, speed, [q, -q, np.zeros(6)]))
        if args.mode == "response":
            for arm in config["arms"]:
                for q in dynamic_targets(config["dynamic"]):
                    jobs.append(("response", "mechanical", arm, None, [q, -q, np.zeros(6)]))
        if args.mode in ("all", "workspace"):
            for scope in config["scopes"]:
                for arm in config["arms"]:
                    for q in workspace_targets(config["workspace"]):
                        jobs.append(("workspace", scope, arm, None, [q]))
        if len(rows) > len(jobs):
            raise ValueError("已有记录超过采样计划")
        for row, (kind, scope, arm, speed, points) in zip(rows, jobs):
            if (row['kind'],row['scope'],row['arm'],row['speed_mps']) != (kind,scope,arm,speed) or row['legs'][0]['reference_bending_per_m'] != points[0].tolist():
                raise ValueError("已有记录与恢复采样计划不匹配")
        pending = [(i, *job, config["criteria"], str(output), config["dynamic"])
                   for i, job in enumerate(jobs) if i >= len(rows)]
        if args.viewer:
            import mujoco.viewer
        display = mujoco.viewer.launch_passive(backend.model, backend.data) if args.viewer else nullcontext(None)
        workers = (ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init,
                   initargs=(robot_path, xml_path, config.get("feedback", {}))) if args.workers > 1 else nullcontext(None))
        with (output/"samples.jsonl").open("a" if args.resume else "w", encoding="utf-8") as stream, display as viewer, workers as pool:
            if viewer:
                with viewer.lock():
                    viewer.cam.lookat[:] = backend.data.site_xpos[backend._site_ids["executor_base_site"]]
                    viewer.cam.distance = .35
                    viewer.cam.azimuth = 135
                    viewer.cam.elevation = -15
            results = pool.map(_worker_job, pending) if pool else (_run_job(backend, robot, job, viewer) for job in pending)
            for row in results:
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
                stream.flush()
                drive = f"{row['speed_mps']*1000:g} mm/s" if row['speed_mps'] is not None else "step"
                print(f"{len(rows)}/{len(jobs)} {row['kind']} {row['scope']} {row['arm']} {drive} {row['status']}", flush=True)
        completed = True
    except KeyboardInterrupt:
        print("采集已中止；完整样本已保留，当前样本 trace 可能未完成。", flush=True)
    finally:
        backend.stop()
        backend.close()
        summary(output, rows, config, completed)
    if rows:
        from confined_dual_continuum.diagnostics.drive_report import create_report
        print(f"图形报告：{create_report([output], output/'report')}", flush=True)


if __name__ == "__main__":
    main()
