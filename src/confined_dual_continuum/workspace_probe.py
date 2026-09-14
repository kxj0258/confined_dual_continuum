"""固定基座六维相容驱动采集；只有显式调用 main 才推进动力学。"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import yaml

from confined_dual_continuum.actuation.compatible_tendon_control import CompatibleTendonController
from confined_dual_continuum.adapters.mujoco_backend import MujocoRobotBackend
from confined_dual_continuum.coordination.interarm_safety import InterArmSafety
from confined_dual_continuum.domain.state import ArmCommand, RobotCommand
from confined_dual_continuum.kinematics.pcc import forward_kinematics, material_point
from confined_dual_continuum.robot.configuration import load_robot_model
from confined_dual_continuum.response_probe import _snapshot, _json


def references(config):
    """零位、逐轴分级正负加载，再采样每段曲率圆盘内的组合。"""
    yield np.zeros(6)
    for amplitude in config["amplitudes_per_m"]:
        for axis in range(6):
            for sign in (-1, 1):
                value = np.zeros(6)
                value[axis] = sign * amplitude
                yield value
    rng = np.random.default_rng(config["seed"])
    maximum = max(config["amplitudes_per_m"])
    for _ in range(config["combination_samples"]):
        angles = rng.uniform(0, 2*np.pi, 3)
        radii = maximum*np.sqrt(rng.uniform(0, 1, 3))
        yield np.column_stack((radii*np.cos(angles), radii*np.sin(angles))).ravel()


def compare(robot, measurement, reference):
    """在固定侧臂根系比较；中心线按实际杆长对应材料位置。"""
    root = np.asarray(measurement["T_world_root"])
    tip = np.linalg.inv(root) @ np.asarray(measurement["T_world_tip"])
    physical = (np.asarray(measurement["physical_centerline_world_m"])-root[:3, 3]) @ root[:3, :3]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(physical, axis=0), axis=1))]
    if not np.isclose(arc[-1], sum(robot.flexure_lengths_m)+sum(robot.distal_offsets_m), atol=1e-8):
        raise ValueError("PCC 与离散中心线总材料长度不一致，不能直接对应比较。")
    # 每个刚性杆内也采样，避免只比较关节而漏掉弧线与直杆之间的差异。
    samples = np.linspace(0., arc[-1], 121)
    actual = np.column_stack([np.interp(samples, arc, physical[:, i]) for i in range(3)])
    model = robot.bending_model(measurement["arm"])
    estimated = model.estimate(np.asarray(measurement["actual_tendon_m"]))
    result = {"actual_tip_root_m": tip[:3, 3].tolist(),
              "estimated_bending_per_m": estimated.tolist(),
              "actual_centerline_root_m": actual.tolist(), "material_arc_m": samples.tolist(),
              "compatibility_residual_m": float(np.linalg.norm(model.residual(np.asarray(measurement["actual_tendon_m"]))))}
    for name, bending in (("command", reference), ("estimated", estimated)):
        pcc = forward_kinematics(bending, robot.flexure_lengths_m, robot.distal_offsets_m)
        points = np.asarray([material_point(bending, robot.flexure_lengths_m, robot.distal_offsets_m, s) for s in samples])
        distances = np.linalg.norm(points-actual, axis=1)
        cosine = (np.trace(pcc.tip_transform[:3, :3].T @ tip[:3, :3])-1)/2
        result[name] = {"tip_root_m": pcc.tip_transform[:3, 3].tolist(),
                        "T_root_tip": pcc.tip_transform.tolist(), "centerline_root_m": points.tolist(),
                        "tip_error_m": float(np.linalg.norm(pcc.tip_transform[:3, 3]-tip[:3, 3])),
                        "orientation_error_deg": float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))),
                        "centerline_rms_m": float(np.sqrt(np.mean(distances**2))),
                        "centerline_max_m": float(np.max(distances))}
    return result


def _unsafe(backend, measurement, config):
    """采集的物理安全判据独立于控制反馈来源。"""
    arms = ("executor", "observer")
    lines = [np.asarray(measurement[a]["physical_centerline_world_m"]) for a in arms]
    if InterArmSafety().evaluate(*lines).safety_stop:
        return "inter_arm_collision_margin"
    if min(backend.nearest_clearance(a, allow_tool_contact=False).distance_m for a in arms) <= config["minimum_clearance_m"]:
        return "environment_collision_margin"
    return None


def collect_sample(backend, robot, arm, reference, config, trace):
    state = backend.reset()
    arms = ("executor", "observer")
    models = {a: robot.bending_model(a) for a in arms}
    actuators = {a: CompatibleTendonController(models[a], robot.limits) for a in arms}
    targets = {a: np.zeros(models[a].tendon_count) for a in arms}
    for a in arms:
        actuators[a].reset(getattr(state, a).tendon_displacement_m)
    dt = config["control_period_s"]
    stable_time = 0.
    saturated = False
    status = "timeout"
    measurement = backend.capture_response_measurement()
    for step in range(int(np.ceil(config["maximum_sample_s"]/dt))):
        reason = _unsafe(backend, measurement, config)
        if reason:
            status = reason
            break
        results = {}
        for a in arms:
            desired = reference if a == arm else np.zeros(6)
            rate = np.clip((desired-models[a].estimate(targets[a]))/dt,
                           -config["maximum_bending_rate_per_m_per_s"], config["maximum_bending_rate_per_m_per_s"])
            results[a] = actuators[a].step(rate, getattr(state, a).tendon_displacement_m, dt)
        if not all(r.feasible for r in results.values()):
            status = "infeasible:" + ";".join(f"{a}:{r.constraint_status}" for a, r in results.items() if not r.feasible)
            for a, result in results.items():
                actuators[a].cancel_step(result, dt)
            break
        saturated |= any(r.saturated for r in results.values())
        commands = {a: ArmCommand(r.bending_rate, r.tendon_rate_mps, r.target_m) for a, r in results.items()}
        backend.apply_command(RobotCommand(np.zeros(6), commands["executor"], commands["observer"]))
        backend.step(dt)
        targets = {a: r.target_m for a, r in results.items()}
        state = backend.read_state()
        previous = measurement
        measurement = backend.capture_response_measurement()
        reason = _unsafe(backend, measurement, config)
        reached = max(float(np.max(np.abs(models[a].estimate(targets[a])-(reference if a == arm else np.zeros(6))))) for a in arms) <= config["target_tolerance_per_m"]
        quiet = all(max(abs(v) for v in measurement[a]["joint_velocity_radps"]) <= config["joint_speed_tolerance_radps"] and
                    max(abs(v) for v in measurement[a]["tendon_velocity_mps"]) <= config["tendon_speed_tolerance_mps"] and
                    np.linalg.norm(np.asarray(measurement[a]["T_world_tip"])[:3, 3]-np.asarray(previous[a]["T_world_tip"])[:3, 3])/dt <= config["tip_speed_tolerance_mps"]
                    for a in arms)
        stable_time = stable_time+dt if reached and quiet and not reason else 0.
        trace.write(json.dumps({"step": step, "measurement": measurement,
                               "target_reached": reached, "quiet": bool(quiet),
                               "constraints": {a: {"status": r.constraint_status, "scale": r.common_scale} for a, r in results.items()}}, allow_nan=False)+"\n")
        if reason:
            status = reason
            break
        if stable_time >= config["stable_hold_s"]:
            status = "stable"
            break
    backend.stop()
    measurement[arm]["arm"] = arm
    return {"arm": arm, "reference_bending_per_m": reference.tolist(), "status": status,
            "joint_at_limit": any(measurement[arm]["joint_at_limit"]),
            "saturated_during_loading": saturated, "duration_s": float(backend.data.time),
            "measurement": measurement, "comparison": compare(robot, measurement[arm], reference)}


def summarize(output, rows):
    """保留全部终态；统计与点云仅采用稳定样本，分臂报告。"""
    summary = {}
    flat = []
    for row in rows:
        item = {k: row[k] for k in ("sample", "arm", "status", "duration_s", "saturated_during_loading", "joint_at_limit")}
        item["reference_norm_per_m"] = float(np.linalg.norm(row["reference_bending_per_m"]))
        for name in ("command", "estimated"):
            for metric in ("tip_error_m", "orientation_error_deg", "centerline_rms_m", "centerline_max_m"):
                item[f"{name}_{metric}"] = row["comparison"][name][metric]
        flat.append(item)
    with (output/"samples.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        if flat:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    for arm in sorted({r["arm"] for r in rows}):
        subset = [r for r in rows if r["arm"] == arm]
        stable = [r for r in subset if r["status"] == "stable"]
        counts = {s: sum(r["status"] == s for r in subset) for s in sorted({r["status"] for r in subset})}
        metrics = {}
        for name in ("command", "estimated"):
            for metric in ("tip_error_m", "orientation_error_deg", "centerline_rms_m", "centerline_max_m"):
                values = [r["comparison"][name][metric] for r in stable]
                metrics[f"{name}_{metric}"] = ({"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95)), "max": float(np.max(values))} if values else None)
        summary[arm] = {"status_counts": counts, "stable_metrics": metrics}
        with (output/f"{arm}_workspace.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sample"]+[f"{name}_{axis}_m" for name in ("mujoco", "command_pcc", "estimated_pcc") for axis in "xyz"])
            for r in stable:
                c = r["comparison"]
                writer.writerow([r["sample"], *c["actual_tip_root_m"], *c["command"]["tip_root_m"], *c["estimated"]["tip_root_m"]])
        _plots(output, arm, stable)
    _json(output/"summary.json", {"scope": "固定基座、另一臂零位、当前场景与约束下的采样可达集合；不是完整工作空间", "arms": summary})


def _plots(output, arm, rows):
    """标准 SVG 散点图，无绘图库或浏览器依赖。"""
    def chart(series, xlabel, ylabel, filename):
        points = [p for _, _, values in series for p in values]
        if not points:
            return
        low = np.min(points, axis=0)
        span = np.maximum(np.ptp(points, axis=0), 1e-6)
        parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="720" height="520" viewBox="0 0 720 520">', '<rect width="720" height="520" fill="white"/>']
        for i in range(6):
            x, y = 65+110*i, 430-70*i
            parts.append(f'<text x="{x}" y="455" font-size="12">{low[0]+span[0]*i/5:.3g}</text><text x="5" y="{y}" font-size="12">{low[1]+span[1]*i/5:.3g}</text>')
        parts.append(f'<path d="M65 70 V430 H615" fill="none" stroke="black"/><text x="290" y="495">{xlabel}</text><text x="65" y="22">{arm}: {ylabel}</text>')
        for index, (name, color, values) in enumerate(series):
            parts.append(f'<text x="{65+index*195}" y="50" fill="{color}">{name}</text>')
            for point in values:
                px, py = (np.asarray(point)-low)/span
                parts.append(f'<circle cx="{65+550*px}" cy="{430-350*py}" r="3" fill="{color}" fill-opacity="0.6"/>')
        (output/filename).write_text("".join(parts)+"</svg>", encoding="utf-8")
    for a, b in ((0, 1), (0, 2), (1, 2)):
        series = []
        for name, color in (("mujoco", "#222222"), ("command", "#d97706"), ("estimated", "#2563eb")):
            points = [r["comparison"]["actual_tip_root_m"] if name == "mujoco" else r["comparison"][name]["tip_root_m"] for r in rows]
            series.append((name, color, [(p[a]*1000, p[b]*1000) for p in points]))
        chart(series, f'{"XYZ"[a]} (mm)', f'{"XYZ"[b]} (mm)', f'{arm}_{"xyz"[a]}{"xyz"[b]}.svg')
    for metric, scale, unit in (("tip_error_m", 1000, "mm"), ("orientation_error_deg", 1, "deg"), ("centerline_rms_m", 1000, "mm")):
        chart([(name, color, [(float(np.linalg.norm(r["reference_bending_per_m"])), r["comparison"][name][metric]*scale) for r in rows]) for name, color in (("command", "#d97706"), ("estimated", "#2563eb"))],
              "command curvature norm (1/m)", f"{metric} ({unit})", f"{arm}_{metric}.svg")


def main():
    parser = argparse.ArgumentParser(description="六维驱动工作空间采集及 PCC/离散模型对比，不打开 viewer")
    parser.add_argument("--config", default="configs/experiments/workspace_probe.yaml")
    args = parser.parse_args()
    path = Path(args.config).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    probe = config["sampling"]
    positive = ("control_period_s", "maximum_sample_s", "stable_hold_s", "maximum_bending_rate_per_m_per_s", "target_tolerance_per_m", "joint_speed_tolerance_radps", "tendon_speed_tolerance_mps", "tip_speed_tolerance_mps", "minimum_clearance_m")
    if any(not np.isfinite(probe[k]) or probe[k] <= 0 for k in positive):
        raise ValueError("采集时间、速率、容差和间距必须为有限正值。")
    if not probe["amplitudes_per_m"] or any(not np.isfinite(v) or v <= 0 for v in probe["amplitudes_per_m"]):
        raise ValueError("采样幅值必须为有限正值。")
    if type(probe["combination_samples"]) is not int or probe["combination_samples"] < 0:
        raise ValueError("组合样本数必须为非负整数。")
    if probe["stable_hold_s"] > probe["maximum_sample_s"]:
        raise ValueError("稳定保持时间不能超过样本超时。")
    arms = config["arms"]
    if not arms or len(set(arms)) != len(arms) or any(a not in ("executor", "observer") for a in arms):
        raise ValueError("采集机械臂配置无效。")
    output = (path.parent/config["output_directory"]).resolve()/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    robot_path, xml_path = _snapshot(output, path, config)
    robot = load_robot_model(robot_path)
    backend = MujocoRobotBackend(xml_path, robot, **config.get("feedback", {}))
    rows = []
    try:
        with (output/"samples.jsonl").open("w", encoding="utf-8") as stream:
            for arm in arms:
                for index, reference in enumerate(references(probe)):
                    with (output/f"{arm}_{index:04d}_trace.jsonl").open("w", encoding="utf-8") as trace:
                        row = collect_sample(backend, robot, arm, reference, probe, trace)
                    row["sample"] = index
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
                    stream.flush()
                    print(f"{arm} sample={index} status={row['status']}", flush=True)
    finally:
        backend.stop()
        backend.close()
        summarize(output, rows)
    print(f"结果目录：{output}")


if __name__ == "__main__":
    main()
