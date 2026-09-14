"""静态坐标系检查：只做 mj_forward，不执行控制器或动力学步进。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from queue import SimpleQueue, Empty
import time

import numpy as np
import yaml


def _transform(position, rotation):
    result = np.eye(4)
    result[:3, :3] = np.asarray(rotation).reshape(3, 3)
    result[:3, 3] = position
    return result


def _frames(mj, model, data):
    """从 MuJoCo 当前派生状态读取坐标系，不能用 CAD 网格位置代替。"""
    result = {"W": ("world", np.eye(4))}
    sites = {
        "B": ("base", "mobile_base_frame"),
        "E0": ("executor", "executor_base_site"),
        "O0": ("observer", "observer_base_site"),
        "Et": ("tips", "executor_tip"),
        "Ot": ("tips", "observer_tip"),
    }
    for label, (group, name) in sites.items():
        identifier = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, name)
        if identifier < 0:
            raise ValueError(f"场景缺少 site: {name}")
        result[label] = (group, _transform(data.site_xpos[identifier], data.site_xmat[identifier]))
    camera = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "observer_camera")
    if camera < 0:
        raise ValueError("场景缺少 observer_camera")
    result["Cam"] = ("camera", _transform(data.cam_xpos[camera], data.cam_xmat[camera]))
    for arm, prefix in (("executor", "E"), ("observer", "O")):
        for segment in range(1, 4):
            name = f"{arm}_segment_{segment}_link_1"
            body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
            if body < 0:
                raise ValueError(f"场景缺少 body: {name}")
            result[f"{prefix}S{segment}"] = ("segments", _transform(data.xpos[body], data.xmat[body]))
    return result


def _report(frames):
    base_inverse = np.linalg.inv(frames["B"][1])
    print("\n坐标原点（m）；W=世界，B=移动基座，E0/O0=臂根，Et/Ot=末端，Cam=相机")
    result = {}
    for name, (_, world) in frames.items():
        relative = base_inverse @ world
        print(f"{name:>4}  W: {np.array2string(world[:3, 3], precision=6)}  B: {np.array2string(relative[:3, 3], precision=6)}")
        result[name] = {"T_world_frame": world.tolist(), "T_base_frame": relative.tolist()}
    result["T_executor_observer"] = (np.linalg.inv(frames["E0"][1]) @ frames["O0"][1]).tolist()
    result["T_observer_tip_camera"] = (np.linalg.inv(frames["Ot"][1]) @ frames["Cam"][1]).tolist()
    print("T_B_E0 =\n", np.asarray(result["E0"]["T_base_frame"]))
    print("T_B_O0 =\n", np.asarray(result["O0"]["T_base_frame"]))
    print("T_Ot_Cam =\n", np.asarray(result["T_observer_tip_camera"]))
    return result


def _draw(mj, scene, frames, visible, length, labels):
    scene.ngeom = 0
    colors = ((1., .1, .1, 1.), (.1, 1., .1, 1.), (.15, .35, 1., 1.))
    def geometry(kind, position, color, size):
        if scene.ngeom >= scene.maxgeom:
            raise ValueError("坐标系标记超过 user_scn 容量。")
        geom = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(geom, kind, np.array([size, 0., 0.]), position,
                        np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
        scene.ngeom += 1
        return geom
    for name, (group, transform) in frames.items():
        if group not in visible:
            continue
        origin = transform[:3, 3]
        marker = geometry(mj.mjtGeom.mjGEOM_SPHERE, origin, (1., 1., 1., 1.), length*.065)
        marker.label = name if labels else ""
        for index, color in enumerate(colors):
            endpoint = origin + length*transform[:3, index]
            arrow = geometry(mj.mjtGeom.mjGEOM_ARROW, origin, color, length*.035)
            mj.mjv_connector(arrow, mj.mjtGeom.mjGEOM_ARROW, length*.035, origin, endpoint)
            arrow.label = f"{name}.{'XYZ'[index]}" if labels else ""


def main():
    parser = argparse.ArgumentParser(description="静态显示坐标原点和 XYZ 轴，不运行动力学")
    parser.add_argument("--config", default="configs/experiments/unified_demo.yaml")
    parser.add_argument("--station", choices=("initial", "entry", "curved_surface", "wiping", "obstacle"), default="initial")
    parser.add_argument("--base-position", type=float, nargs=3, metavar=("X", "Y", "Z"), help="显式覆盖基座世界位置，单位 m")
    parser.add_argument("--base-quaternion", type=float, nargs=4, metavar=("W", "X", "Y", "Z"), help="显式覆盖基座世界姿态，wxyz")
    parser.add_argument("--axis-length", type=float, default=.015, help="坐标轴长度，单位 m")
    parser.add_argument("--report", type=Path, help="可选保存初始坐标变换 JSON；不覆盖已有文件")
    args = parser.parse_args()
    if not np.isfinite(args.axis_length) or args.axis_length <= 0:
        parser.error("坐标轴长度必须为有限正数")
    import mujoco as mj
    import mujoco.viewer
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = mj.MjModel.from_xml_path(str((config_path.parent/config["mujoco_xml"]).resolve()))
    data = mj.MjData(model)
    base = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "mobile_base")
    if base < 0 or model.body_mocapid[base] < 0:
        raise ValueError("检查工具要求 mobile_base 为 mocap 基座")
    mocap = int(model.body_mocapid[base])
    if args.station != "initial":
        workspace = yaml.safe_load((config_path.parent/config["workspace_config"]).resolve().read_text(encoding="utf-8"))
        data.mocap_pos[mocap] = workspace["base_stations"][args.station]
    if args.base_position is not None:
        if not np.all(np.isfinite(args.base_position)):
            parser.error("基座位置必须有限")
        data.mocap_pos[mocap] = args.base_position
    if args.base_quaternion is not None:
        quaternion = np.asarray(args.base_quaternion)
        norm = np.linalg.norm(quaternion)
        if not np.isfinite(norm) or norm <= 1e-12:
            parser.error("基座四元数必须有限且非零")
        data.mocap_quat[mocap] = quaternion/norm
    mj.mj_forward(model, data)
    report = _report(_frames(mj, model, data))
    if args.report:
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print("\n红=X，绿=Y，蓝=Z；相机实际看向其 -Z。")
    print("按 1 世界 / 2 基座 / 3 Executor根 / 4 Observer根 / 5 末端 / 6 相机 / 7 分段 切换显示。")
    print("按 T 切换标签，P 打印当前坐标。关闭窗口结束；本工具不调用 mj_step。")
    events = SimpleQueue()
    groups = dict(zip("1234567", ("world", "base", "executor", "observer", "tips", "camera", "segments")))
    visible = set(groups.values()) - {"segments"}
    labels = True
    # 键盘回调仅投递事件；所有 MuJoCo 状态和场景写入在主线程及 viewer 锁内完成。
    def key_callback(key):
        if 0 <= key < 128:
            events.put(chr(key).upper())
    with mj.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = data.mocap_pos[mocap] + [0., 0., .06]
            viewer.cam.distance = .5
            viewer.cam.azimuth = 135.
            viewer.cam.elevation = -20.
        while viewer.is_running():
            with viewer.lock():
                mj.mj_forward(model, data)
                frames = _frames(mj, model, data)
                while True:
                    try:
                        key = events.get_nowait()
                    except Empty:
                        break
                    if key in groups:
                        group = groups[key]
                        if group in visible:
                            visible.remove(group)
                        else:
                            visible.add(group)
                    elif key == "T":
                        labels = not labels
                    elif key == "P":
                        _report(frames)
                _draw(mj, viewer.user_scn, frames, visible, args.axis_length, labels)
            viewer.sync()
            time.sleep(.03)


if __name__ == "__main__":
    main()
