"""专利统一 Demo 的正式命令行入口。"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import yaml

from confined_dual_continuum.actuation.compatible_tendon_control import (
    CompatibleTendonController,
)
from confined_dual_continuum.adapters.mujoco_backend import MujocoRobotBackend
from confined_dual_continuum.application.controller import ConfinedDualContinuumController
from confined_dual_continuum.application.session import OperationSession, SessionConfig
from confined_dual_continuum.coordination.observer_target import (
    ObserverTargetConfig,
    ObserverTargetGenerator,
)
from confined_dual_continuum.robot.model import load_robot_model
from confined_dual_continuum.task.operation_director import (
    OperationDirector,
    OperationPlan,
)
from confined_dual_continuum.workspace.configuration import load_workspace


def main() -> None:
    parser = argparse.ArgumentParser(description="运行分阶段非对称双连续体专利统一 Demo")
    parser.add_argument("--config", default="configs/experiments/unified_demo.yaml")
    parser.add_argument("--output", help="覆盖配置中的诊断输出目录")
    parser.add_argument("--viewer", action="store_true", help="打开 MuJoCo 被动查看器")
    arguments = parser.parse_args()

    config_path = Path(arguments.config).resolve()
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = config_path.parent
    robot = load_robot_model((root / values["robot_config"]).resolve())
    workspace = load_workspace((root / values["workspace_config"]).resolve())
    backend = MujocoRobotBackend((root / values["mujoco_xml"]).resolve(), robot)
    observer = values["observer"]
    controller = ConfinedDualContinuumController(
        CompatibleTendonController(robot.bending_model, robot.limits),
        CompatibleTendonController(robot.bending_model, robot.limits),
        robot,
        ObserverTargetGenerator(
            ObserverTargetConfig(
                normal_offset_m=float(observer["normal_offset_m"]),
                tangential_offset_m=float(observer["tangential_offset_m"]),
                max_target_speed_mps=float(observer["maximum_target_speed_mps"]),
            )
        ),
    )
    task = values["task"]
    director = OperationDirector(
        workspace,
        OperationPlan(
            position_tolerance_m=float(task["position_tolerance_m"]),
            base_gain=float(task["base_gain"]),
            maximum_base_speed_mps=float(task["maximum_base_speed_mps"]),
            local_operation_duration_s=float(task["local_operation_duration_s"]),
            bending_tolerance_inv_m=float(task["bending_tolerance_inv_m"]),
            straight_executor_tip_offset_m=task["straight_executor_tip_offset_m"],
        ),
    )
    runtime = values["runtime"]
    session = OperationSession(backend, controller, director, SessionConfig(
        control_period_s=float(runtime["control_period_s"]),
        maximum_duration_s=float(runtime["maximum_duration_s"]),
    ))
    output = (
        Path(arguments.output).resolve()
        if arguments.output
        else (root / values["output_directory"]).resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "config.yaml")
    summary = _run(session, backend, str(output), arguments.viewer)
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


def _run(
    session: OperationSession,
    backend: MujocoRobotBackend,
    output: str,
    viewer_enabled: bool,
):
    if not viewer_enabled:
        return session.run(output)
    import mujoco.viewer

    with mujoco.viewer.launch_passive(backend.model, backend.data) as viewer:
        return session.run(output, on_step=lambda _state: viewer.sync())


if __name__ == "__main__":
    main()
