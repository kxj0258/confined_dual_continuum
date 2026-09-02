"""专利统一 Demo 的正式命令行入口。"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from confined_dual_continuum.actuation.compatible_tendon_control import (
    CompatibleTendonController,
)
from confined_dual_continuum.adapters.mujoco_backend import MujocoRobotBackend
from confined_dual_continuum.application.controller import ConfinedDualContinuumController
from confined_dual_continuum.application.session import OperationSession, SessionConfig
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
    controller = ConfinedDualContinuumController(
        CompatibleTendonController(robot.bending_model, robot.limits),
        CompatibleTendonController(robot.bending_model, robot.limits),
        robot,
    )
    task = values["task"]
    director = OperationDirector(
        workspace,
        OperationPlan(
            position_tolerance_m=float(task["position_tolerance_m"]),
            base_gain=float(task["base_gain"]),
            maximum_base_speed_mps=float(task["maximum_base_speed_mps"]),
            waypoint_duration_s=float(task["waypoint_duration_s"]),
            wiping_force_n=float(task["wiping_force_n"]),
            observer_offset_m=float(task["observer_offset_m"]),
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
    summary = _run(
        session,
        backend,
        str(output),
        arguments.viewer,
        values["viewer"],
        values["video"],
        float(runtime["control_period_s"]),
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


def _run(
    session: OperationSession,
    backend: MujocoRobotBackend,
    output: str,
    viewer_enabled: bool,
    viewer_config: dict[str, object],
    video_config: dict[str, object],
    control_period_s: float,
):
    import imageio.v2 as imageio

    video_path = Path(output) / str(video_config["filename"])
    writer = imageio.get_writer(video_path, fps=int(video_config["fps"]))
    capture_interval = max(
        1,
        round(1.0 / (int(video_config["fps"]) * control_period_s)),
    )
    step = 0

    def capture_frame() -> None:
        nonlocal step
        if step % capture_interval == 0:
            writer.append_data(backend.capture_observer_frame(
                int(video_config["width"]),
                int(video_config["height"]),
            ))
        step += 1

    if not viewer_enabled:
        try:
            return session.run(output, on_step=lambda _state: capture_frame())
        finally:
            writer.close()

    import mujoco.viewer

    try:
        with mujoco.viewer.launch_passive(backend.model, backend.data) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = viewer_config["look_at_world_m"]
                viewer.cam.distance = float(viewer_config["distance_m"])
                viewer.cam.azimuth = float(viewer_config["azimuth_deg"])
                viewer.cam.elevation = float(viewer_config["elevation_deg"])
            next_frame_time = time.perf_counter()

            def refresh(_state) -> bool:
                nonlocal next_frame_time
                if not viewer.is_running():
                    return False
                capture_frame()
                viewer.sync()
                next_frame_time += control_period_s
                delay = next_frame_time - time.perf_counter()
                if delay > 0.0:
                    time.sleep(delay)
                return True

            return session.run(output, on_step=refresh)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
