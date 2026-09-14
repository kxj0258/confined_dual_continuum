"""独立响应诊断装配入口；显式运行才推进物理，不调用任务状态机或 viewer。"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
import yaml

from confined_dual_continuum.actuation.compatible_tendon_control import CompatibleTendonController
from confined_dual_continuum.adapters.mujoco_backend import MujocoRobotBackend
from confined_dual_continuum.application.response_protocol import ResponsePlan
from confined_dual_continuum.coordination.interarm_safety import InterArmSafety
from confined_dual_continuum.domain.state import ArmCommand, RobotCommand
from confined_dual_continuum.kinematics.pcc import material_point
from confined_dual_continuum.robot.configuration import load_robot_model


def _json(path, values):
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _snapshot(output, config_path, config):
    root = config_path.parent
    robot_path = (root/config["robot_config"]).resolve()
    xml_path = (root/config["mujoco_xml"]).resolve()
    snapshot = output/"snapshot"
    snapshot.mkdir()
    shutil.copy2(config_path, snapshot/"source-config.yaml")
    shutil.copy2(robot_path, snapshot/"robot.yaml")
    shutil.copytree(xml_path.parent.parent, snapshot/"assets")
    package = Path(__file__).resolve().parent
    shutil.copytree(package, snapshot/"src"/package.name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    frozen = dict(config, robot_config="robot.yaml", mujoco_xml=f"assets/{xml_path.parent.name}/{xml_path.name}", output_directory="../replay")
    (snapshot/"config.yaml").write_text(yaml.safe_dump(frozen, allow_unicode=True, sort_keys=False), encoding="utf-8")
    hashes = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.rglob("*") if p.is_file()}
    _json(output/"manifest.json", {"versions": {name: version(name) for name in ("numpy", "mujoco", "PyYAML")}, "snapshot_sha256": hashes})
    return robot_path, xml_path


def run_arm(backend, robot, plan, active_arm, output):
    """另一臂保持零位；两臂均执行原有约束检查。失败不自动放宽阈值。"""
    output.mkdir()
    summary = {"arm": active_arm, "protocol_completed": False, "reason": "not_started",
               "steps": 0, "duration_s": 0., "maximum_lead_m": 0.,
               "maximum_actuator_force_n": 0., "maximum_pcc_tip_error_m": 0., "plateaus": {}}
    dt = plan.control_period_s
    try:
        state = backend.reset()
        initial = backend.capture_response_measurement()
        _json(output/"initial_frames_and_state.json", initial)
        actuators = {arm: CompatibleTendonController(robot.bending_model(arm), robot.limits) for arm in ("executor", "observer")}
        targets = {arm: np.zeros(9) for arm in actuators}
        for arm, actuator in actuators.items():
            actuator.reset(getattr(state, arm).tendon_displacement_m)
        safety = InterArmSafety()
        with (output/"samples.jsonl").open("w", encoding="utf-8") as stream, (output/"timeseries.csv").open("w", newline="", encoding="utf-8-sig") as csv_stream:
            writer = None
            summary["reason"] = "running"
            for index, (stage, axis, sign, reference) in enumerate(plan.samples()):
                decision_time = float(backend.data.time)
                interarm = safety.evaluate(state.executor.centerline_world, state.observer.centerline_world)
                gaps = {arm: backend.nearest_clearance(arm, allow_tool_contact=False).distance_m for arm in actuators}
                reason = None
                if interarm.safety_stop:
                    reason = "inter_arm_critical_clearance"
                if min(gaps.values()) <= .0005:
                    reason = "environment_critical_clearance"
                if max(np.linalg.norm(getattr(state, arm).contact_force_world) for arm in actuators) > 3.:
                    reason = "maximum_contact_force"
                results, commands = {}, {}
                for arm, actuator in actuators.items():
                    # 追踪命令参考而非反馈估计，便于直接观察开环物理响应。
                    target_bending = reference if arm == active_arm else np.zeros(6)
                    current_bending_target = actuator.model.estimate(targets[arm])
                    maximum_rate = plan.amplitude_per_m/min(plan.ramp_s, plan.return_s)
                    rate = np.clip((target_bending-current_bending_target)/dt, -maximum_rate, maximum_rate)
                    if reason:
                        rate = np.zeros(6)
                    result = actuator.step(rate, getattr(state, arm).tendon_displacement_m, dt,
                                           allow_recovery=(arm == active_arm and reason is None))
                    results[arm] = result
                    commands[arm] = ArmCommand(result.bending_rate, result.tendon_rate_mps, result.target_m)
                    if not result.feasible:
                        reason = f"{arm}:{result.constraint_status}:tendon_{result.limiting_tendon_index+1}"
                accepted = reason is None
                if accepted:
                    backend.apply_command(RobotCommand(np.zeros(6), commands["executor"], commands["observer"]))
                    backend.step(dt)
                    summary["steps"] += 1
                    for arm in actuators:
                        targets[arm] = results[arm].target_m.copy()
                else:
                    for arm, actuator in actuators.items():
                        actuator.cancel_step(results[arm], dt)
                state = backend.read_state()
                measurement = backend.capture_response_measurement()
                row = {"sample": index, "decision_time_s": decision_time, "time_s": measurement["time_s"],
                       "stage": stage, "axis": axis, "sign": sign, "accepted": accepted,
                       "reason": reason or "", "active_arm": active_arm,
                       "inter_arm_gap_before_m": interarm.distance_m,
                       "environment_gap_before_m": min(gaps.values())}
                for arm, result in results.items():
                    feedback = measurement[arm]
                    actual = np.asarray(feedback["actual_tendon_m"])
                    root = np.asarray(feedback["T_world_root"])
                    tip = np.asarray(feedback["T_world_tip"])
                    tip_root = root[:3, :3].T @ (tip[:3, 3]-root[:3, 3])
                    bending = actuators[arm].model.estimate(actual)
                    pcc_error = float(np.linalg.norm(tip_root-material_point(bending, robot.flexure_lengths_m, robot.distal_offsets_m)))
                    lead = np.asarray(feedback["target_tendon_m"])-actual
                    feedback.update(estimated_bending_per_m=bending.tolist(), tip_root_m=tip_root.tolist(),
                                    centerline_world_m=getattr(state, arm).centerline_world.tolist(),
                                    pcc_tip_error_m=pcc_error,
                                    actual_compatibility_residual_m=float(np.linalg.norm(actuators[arm].model.residual(actual))))
                    row.update({f"{arm}_status": result.constraint_status, f"{arm}_limiting_tendon_index": result.limiting_tendon_index,
                                f"{arm}_lead_excess_before_m": result.lead_excess_m, f"{arm}_scale": result.common_scale,
                                f"{arm}_pcc_tip_error_m": pcc_error})
                    for name, values in (("actual_m", actual), ("target_m", feedback["target_tendon_m"]),
                                         ("velocity_mps", feedback["tendon_velocity_mps"]),
                                         ("force_n", feedback["actuator_force_n"]), ("lead_m", lead)):
                        row.update({f"{arm}_{name}_{i+1}": float(v) for i,v in enumerate(values)})
                    summary["maximum_lead_m"] = max(summary["maximum_lead_m"], float(np.max(np.abs(lead))))
                    summary["maximum_actuator_force_n"] = max(summary["maximum_actuator_force_n"], float(np.max(np.abs(feedback["actuator_force_n"]))))
                    summary["maximum_pcc_tip_error_m"] = max(summary["maximum_pcc_tip_error_m"], pcc_error)
                stream.write(json.dumps({**row, "reference_bending_per_m": reference.tolist(), "measurement": measurement}, ensure_ascii=False, allow_nan=False)+"\n")
                stream.flush()
                if writer is None:
                    writer = csv.DictWriter(csv_stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                csv_stream.flush()
                if stage == "hold" and accepted:
                    summary["plateaus"][f"axis_{axis}_sign_{sign}"] = {
                        "tip_root_m": measurement[active_arm]["tip_root_m"],
                        "estimated_bending_per_m": measurement[active_arm]["estimated_bending_per_m"],
                        "maximum_tendon_speed_mps": float(np.max(np.abs(measurement[active_arm]["tendon_velocity_mps"]))),
                    }
                if reason:
                    summary["reason"] = reason
                    break
            else:
                summary.update(protocol_completed=True, reason="protocol_finished_not_a_reachability_claim")
    except BaseException as exc:
        summary["reason"] = f"exception:{type(exc).__name__}:{exc}"
        raise
    finally:
        backend.stop()
        summary["duration_s"] = float(backend.data.time)
        _json(output/"summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="固定基座的小幅单臂相容响应诊断，不开 viewer 或视频")
    parser.add_argument("--config", default="configs/experiments/response_probe.yaml")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    plan_values = dict(config["probe"])
    plan_values["axes"] = tuple(plan_values["axes"])
    plan = ResponsePlan(**plan_values)
    arms = config["arms"]
    if not arms or len(set(arms)) != len(arms) or any(arm not in ("executor", "observer") for arm in arms):
        raise ValueError("arms 必须明确列出待诊断机械臂，不允许重复")
    output = (config_path.parent/config["output_directory"]).resolve()/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    robot_path, xml_path = _snapshot(output, config_path, config)
    robot = load_robot_model(robot_path)
    backend = MujocoRobotBackend(xml_path, robot, **config.get("feedback", {}))
    results = []
    try:
        for arm in arms:
            results.append(run_arm(backend, robot, plan, arm, output/arm))
            _json(output/"comparison.json", results)
    finally:
        backend.stop()
        backend.close()
    print(json.dumps({"output_directory": str(output), "planned_simulation_duration_s": len(arms)*plan.duration_s,
                      "results": results}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
