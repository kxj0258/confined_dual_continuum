"""正式 Demo 与显式消融入口；每次运行保存独立、可追溯的结果。"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import shutil
import subprocess
import time
from uuid import uuid4
import yaml
from confined_dual_continuum.actuation.compatible_tendon_control import CompatibleTendonController
from confined_dual_continuum.adapters.mujoco_backend import MujocoRobotBackend
from confined_dual_continuum.application.controller import ConfinedDualContinuumController
from confined_dual_continuum.application.session import OperationSession, SessionConfig
from confined_dual_continuum.robot.configuration import load_robot_model
from confined_dual_continuum.task.operation_director import OperationDirector, OperationPlan
from confined_dual_continuum.workspace.configuration import load_workspace

VARIANTS = {"formal", "always_active", "staged_activation", "fixed_target", "task_assist",
            "independent_clipping", "bending_compatible"}


def main():
    parser = argparse.ArgumentParser(description="分阶段双连续体验证")
    parser.add_argument("--config", default="configs/experiments/unified_demo.yaml")
    parser.add_argument("--output", help="结果父目录；每次运行自动创建独立子目录")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-video", action="store_true", help="只运行控制和记录，不创建渲染器")
    parser.add_argument("--no-obstacles", action="store_true", help="固定基座跟随任务：移除外部障碍，保留双臂碰撞")
    parser.add_argument("--ablation", help="显式运行消融配置的全部变体；非相容基线仅允许此入口")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if values.get('task_type') == 'fixed_base_tracking':
        if args.ablation:
            parser.error('固定基座跟随任务不使用四工位消融配置。')
        from confined_dual_continuum.tracking_cli import run_tracking
        run_tracking(config_path, values, output_parent=args.output,
                     viewer_enabled=args.viewer, video_enabled=not args.no_video,
                     no_obstacles=args.no_obstacles)
        return
    if args.no_obstacles:
        parser.error('--no-obstacles 仅适用于 fixed_base_tracking。')
    if args.ablation:
        ablation_path = Path(args.ablation).resolve()
        ablation = yaml.safe_load(ablation_path.read_text(encoding="utf-8"))
        variants = ablation["variants"]
        if not variants or any(v not in VARIANTS or v == "formal" for v in variants):
            raise ValueError("消融变体无效。")
    else:
        ablation_path = None
        variants = ["formal"]
    parent = Path(args.output).resolve() if args.output else (config_path.parent / values["output_directory"]).resolve()
    batch = parent / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    batch.mkdir(parents=True, exist_ok=False)
    if ablation_path:
        shutil.copy2(ablation_path, batch / "ablation.yaml")
    results = []
    for variant in variants:
        output = batch / variant
        output.mkdir()
        try:
            summary = run_experiment(config_path, deepcopy(values), output, args.viewer, not args.no_video, variant)
            results.append({"variant": variant, **asdict(summary)})
        except Exception as exc:
            # 已保存的失败汇总仍参与消融聚合，不把失败试验过滤掉。
            summary_path = output / "summary.json"
            failure = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"completed": False, "termination_reason": str(exc)}
            results.append({"variant": variant, **failure})
            (batch / "comparison.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            if not ablation_path:
                raise
        (batch / "comparison.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output_directory": str(batch), "runs": results}, ensure_ascii=False, indent=2, allow_nan=False))


def run_experiment(config_path, values, output, viewer_enabled, video_enabled, variant="formal"):
    if variant not in VARIANTS:
        raise ValueError("未知控制变体。")
    root = config_path.parent
    robot_path = (root / values["robot_config"]).resolve()
    workspace_path = (root / values["workspace_config"]).resolve()
    xml_path = (root / values["mujoco_xml"]).resolve()
    _snapshot(output, config_path, values, robot_path, workspace_path, xml_path, variant)
    robot = load_robot_model(robot_path)
    workspace = load_workspace(workspace_path)
    backend = MujocoRobotBackend(xml_path, robot, **values.get("feedback", {}))
    try:
        actuator = CompatibleTendonController
        if variant == "independent_clipping":
            from confined_dual_continuum.actuation.independent_tendon_control import IndependentTendonController
            actuator = IndependentTendonController
        controller = ConfinedDualContinuumController(
            actuator(robot.bending_model("executor"), robot.limits),
            actuator(robot.bending_model("observer"), robot.limits), robot, backend,
            staged_activation=variant != "always_active", observer_task_assist=variant != "fixed_target")
        director = OperationDirector(workspace, OperationPlan(**values["task"]))
        session = OperationSession(backend, controller, director, SessionConfig(**values["runtime"]))
        return _run(session, backend, output, viewer_enabled, values["viewer"], values["video"], video_enabled)
    finally:
        backend.stop()
        backend.close()


def _snapshot(output, config_path, values, robot_path, workspace_path, xml_path, variant):
    snapshot = output / "snapshot"
    snapshot.mkdir()
    shutil.copy2(config_path, snapshot / "source-config.yaml")
    shutil.copy2(robot_path, snapshot / "robot.yaml")
    shutil.copy2(workspace_path, snapshot / "workspace.yaml")
    shutil.copytree(xml_path.parent.parent, snapshot / "assets")
    package = Path(__file__).resolve().parent
    shutil.copytree(package, snapshot / "src" / package.name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    frozen = deepcopy(values)
    frozen.update(robot_config="robot.yaml", workspace_config="workspace.yaml",
                  mujoco_xml=f"assets/{xml_path.parent.name}/{xml_path.name}", output_directory="../replay-output")
    (snapshot / "config.yaml").write_text(yaml.safe_dump(frozen, allow_unicode=True, sort_keys=False), encoding="utf-8")
    versions = {}
    for name in ("numpy", "PyYAML", "mujoco", "imageio", "imageio-ffmpeg"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    repository = package.parents[1]
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repository, capture_output=True, text=True, check=True).stdout)
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    hashes = {str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in snapshot.rglob("*") if path.is_file()}
    (output / "manifest.json").write_text(json.dumps({"variant": variant, "git_revision": revision,
        "git_dirty": dirty, "versions": versions, "snapshot_sha256": hashes}, ensure_ascii=False, indent=2), encoding="utf-8")


def _run(session, backend, output, viewer_enabled, viewer_config, video_config, video_enabled):
    dt = session.config.control_period_s
    with ExitStack() as stack:
        writer = None
        global_writer = None
        fps = float(video_config["fps"])
        if video_enabled:
            if not 0 < fps <= 1/dt:
                raise ValueError("视频帧率必须为正且不能超过控制频率。")
            import imageio.v2 as imageio
            global_config = video_config["global_follow"]
            if global_config["enabled"] and (output / str(global_config["filename"])).resolve() == (output / str(video_config["filename"])).resolve():
                raise ValueError("全局视频与 Observer 视频不能使用同一个文件名。")
            writer = imageio.get_writer(output / str(video_config["filename"]), fps=fps)
            stack.callback(writer.close)
            if global_config["enabled"]:
                global_writer = imageio.get_writer(output / str(global_config["filename"]), fps=fps)
                stack.callback(global_writer.close)
        viewer = None
        if viewer_enabled:
            import mujoco.viewer
            viewer = stack.enter_context(mujoco.viewer.launch_passive(backend.model, backend.data))
            with viewer.lock():
                viewer.cam.lookat[:] = viewer_config["look_at_world_m"]
                viewer.cam.distance = float(viewer_config["distance_m"])
                viewer.cam.azimuth = float(viewer_config["azimuth_deg"])
                viewer.cam.elevation = float(viewer_config["elevation_deg"])
        frames, steps = 0, 0
        wall_start = time.perf_counter()
        def refresh(_state):
            nonlocal frames, steps
            steps += 1
            if writer is not None and frames/fps < steps*dt - 1e-12:
                writer.append_data(backend.capture_observer_frame(int(video_config["width"]), int(video_config["height"])))
                if global_writer is not None:
                    global_writer.append_data(backend.capture_global_follow_frame(
                        int(video_config["width"]), int(video_config["height"]),
                        look_at_offset_m=global_config["look_at_offset_m"],
                        distance_m=float(global_config["distance_m"]),
                        azimuth_deg=float(global_config["azimuth_deg"]),
                        elevation_deg=float(global_config["elevation_deg"])))
                frames += 1
            if viewer is not None:
                if not viewer.is_running():
                    return False
                viewer.sync()
                time.sleep(max(0.0, wall_start + steps*dt - time.perf_counter()))
            return True
        return session.run(str(output), on_step=refresh)


if __name__ == "__main__":
    main()
