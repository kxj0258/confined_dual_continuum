"""将稳定的机器人合同映射到专利样机 MuJoCo XML。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from confined_dual_continuum.domain.geometry import Pose
from confined_dual_continuum.domain.state import ArmState, RobotCommand, RobotState
from confined_dual_continuum.robot.model import RobotModel
from confined_dual_continuum.workspace.query import Clearance
from confined_dual_continuum.kinematics.pcc import forward_kinematics, transform_points


class MujocoRobotBackend:
    """薄适配器：只负责反馈读取、命令映射和物理步进。"""

    def __init__(self, xml_path: str | Path, robot: RobotModel, *,
                 centerline_source: str = "pcc", tip_pose_source: str = "mujoco",
                 require_wiping_board: bool = True) -> None:
        if centerline_source not in ("pcc", "mujoco") or tip_pose_source not in ("pcc", "mujoco"):
            raise ValueError("反馈来源必须为 pcc 或 mujoco。")
        self.centerline_source = centerline_source
        self.tip_pose_source = tip_pose_source
        self._mj = _import_mujoco()
        self.model = self._mj.MjModel.from_xml_path(str(Path(xml_path).resolve()))
        self.data = self._mj.MjData(self.model)
        self.robot = robot
        self._bending_models = {arm: robot.bending_model(arm) for arm in ("executor", "observer")}
        base_body = self._id(self._mj.mjtObj.mjOBJ_BODY, "mobile_base")
        self._base_mocap = int(self.model.body_mocapid[base_body])
        if self._base_mocap < 0:
            raise ValueError("正式基座必须是理想 mocap 平台。")
        self._link_bodies = {
            arm: [self._id(self._mj.mjtObj.mjOBJ_BODY, f"{arm}_segment_{segment}_link_{link}")
                  for segment in range(1, 4) for link in range(1, 5)]
            for arm in ("executor", "observer")
        }
        self._site_ids = {
            name: self._id(self._mj.mjtObj.mjOBJ_SITE, name)
            for name in (
                "mobile_base_frame",
                "executor_base_site",
                "executor_tip",
                "observer_base_site",
                "observer_tip",
            )
        }
        self._tendon_ids = {
            arm: np.array(
                [
                    self._id(
                        self._mj.mjtObj.mjOBJ_TENDON, f"{arm}_tendon_{index}"
                    )
                    for index in range(1, self._bending_models[arm].tendon_count + 1)
                ]
            )
            for arm in ("executor", "observer")
        }
        self._actuator_ids = {
            arm: np.array(
                [
                    self._id(
                        self._mj.mjtObj.mjOBJ_ACTUATOR,
                        f"act_{arm}_tendon_{index}",
                    )
                    for index in range(1, self._bending_models[arm].tendon_count + 1)
                ]
            )
            for arm in ("executor", "observer")
        }
        self._pending: RobotCommand | None = None
        self._collision_geoms = {
            arm: [self._id(self._mj.mjtObj.mjOBJ_GEOM, f"{arm}_segment_{segment}_link_{link}_collision")
                  for segment in range(1, 4) for link in range(1, 5)]
            for arm in ("executor", "observer")
        }
        arm_geoms = set(self._collision_geoms["executor"] + self._collision_geoms["observer"])
        self._environment_geoms = [i for i in range(self.model.ngeom)
                                   if i not in arm_geoms and self.model.geom_contype[i] != 0]
        self._wiping_board = (self._id(self._mj.mjtObj.mjOBJ_GEOM, "wiping_board")
                              if require_wiping_board else -1)
        self._executor_force_sensor = self._id(
            self._mj.mjtObj.mjOBJ_SENSOR, "sensor_executor_tip_force"
        )
        self._renderer = None
        self._mj.mj_forward(self.model, self.data)
        self._neutral = {
            arm: self.data.ten_length[indices].copy()
            for arm, indices in self._tendon_ids.items()
        }
        self._targets = {arm: values.copy() for arm, values in self._neutral.items()}

    def reset(self) -> RobotState:
        self._mj.mj_resetData(self.model, self.data)
        self._mj.mj_forward(self.model, self.data)
        self._pending = None
        self._targets = {arm: values.copy() for arm, values in self._neutral.items()}
        self._write_targets()
        # 复位后的力/速度传感器也必须对应刚写入的中性目标，而非 resetData 的零 ctrl。
        self._mj.mj_forward(self.model, self.data)
        return self.read_state()

    def read_state(self) -> RobotState:
        base_transform = self._site_transform("mobile_base_frame")
        return RobotState(
            base_pose=_pose(base_transform),
            executor=self._read_arm("executor"),
            observer=self._read_arm("observer"),
        )

    def apply_command(self, command: RobotCommand) -> None:
        self._pending = command

    def step(self, dt: float) -> None:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt 必须为正。")
        timestep = float(self.model.opt.timestep)
        steps = int(round(dt / timestep))
        if steps < 1 or not np.isclose(steps * timestep, dt, rtol=0, atol=1e-12):
            raise ValueError("控制周期必须是物理步长的正整数倍。")
        if self._pending is not None:
            for arm in ("executor", "observer"):
                command = getattr(self._pending, arm)
                expected = self._targets[arm] - self._neutral[arm] + dt * command.tendon_rate_mps
                if command.tendon_target_m.shape != expected.shape or not np.allclose(
                    command.tendon_target_m, expected, rtol=0, atol=1e-10
                ):
                    raise ValueError(f"{arm} 控制器与后端肌腱积分目标不一致。")
        for _ in range(steps):
            if self._pending is not None:
                self._integrate_command(self._pending, timestep)
            self._mj.mj_step(self.model, self.data)
        # 读状态前同步末周期积分后的派生位置、肌腱长度与传感器。
        self._mj.mj_forward(self.model, self.data)

    def _read_arm(self, arm: str) -> ArmState:
        tendon = self.data.ten_length[self._tendon_ids[arm]] - self._neutral[arm]
        bending = self._bending_models[arm].estimate(tendon)
        physical_tip = self._site_transform(f"{arm}_tip")
        pcc = forward_kinematics(bending, self.robot.flexure_lengths_m, self.robot.distal_offsets_m)
        root = self._site_transform(f"{arm}_base_site")
        pcc_tip = root @ pcc.tip_transform
        centerline = (transform_points(root, pcc.centerline) if self.centerline_source == "pcc"
                      else self._physical_centerline(arm))
        return ArmState(
            tip_pose=_pose(pcc_tip if self.tip_pose_source == "pcc" else physical_tip),
            bending=bending,
            tendon_displacement_m=tendon,
            centerline_world=centerline,
            contact_force_world=self._contact_force_world(arm),
            pcc_tip_error_m=float(np.linalg.norm(pcc_tip[:3, 3] - physical_tip[:3, 3])),
        )

    def _physical_centerline(self, arm):
        return np.vstack((self.data.xpos[self._link_bodies[arm]],
                          self._site_transform(f"{arm}_tip")[:3, 3]))

    def capture_observer_frame(self, width: int, height: int) -> np.ndarray:
        """渲染 Observer 末端摄像头的 RGB 图像。"""

        return self._capture_frame(width, height, "observer_camera")

    def capture_response_measurement(self) -> dict[str, object]:
        """诊断边界读取物理反馈；不参与控制层状态推断。数组转列表便于保存。"""
        result = {"time_s": float(self.data.time),
                  "T_world_base": self._site_transform("mobile_base_frame").tolist()}
        for arm in ("executor", "observer"):
            joints = [int(self.model.body_jntadr[body]) for body in self._link_bodies[arm]]
            if any(self.model.body_jntnum[body] != 1 for body in self._link_bodies[arm]):
                raise ValueError("响应诊断要求每根离散杆具有一个转动关节。")
            qpos = [float(self.data.qpos[self.model.jnt_qposadr[joint]]) for joint in joints]
            qvel = [float(self.data.qvel[self.model.jnt_dofadr[joint]]) for joint in joints]
            velocity = []
            for index in range(1, len(self._tendon_ids[arm])+1):
                sensor = self._id(self._mj.mjtObj.mjOBJ_SENSOR, f"sensor_{arm}_tendon_{index}_velocity")
                velocity.append(float(self.data.sensordata[self.model.sensor_adr[sensor]]))
            result[arm] = {
                "T_world_root": self._site_transform(f"{arm}_base_site").tolist(),
                "T_world_tip": self._site_transform(f"{arm}_tip").tolist(),
                "physical_centerline_world_m": self._physical_centerline(arm).tolist(),
                "feedback_sources": {"centerline": self.centerline_source, "tip_pose": self.tip_pose_source},
                "actual_tendon_m": (self.data.ten_length[self._tendon_ids[arm]]-self._neutral[arm]).tolist(),
                "target_tendon_m": (self._targets[arm]-self._neutral[arm]).tolist(),
                "tendon_velocity_mps": velocity,
                "actuator_force_n": self.data.actuator_force[self._actuator_ids[arm]].tolist(),
                "joint_names": [self._mj.mj_id2name(self.model, self._mj.mjtObj.mjOBJ_JOINT, joint) for joint in joints],
                "joint_position_rad": qpos, "joint_velocity_radps": qvel,
                "joint_at_limit": [bool(self.model.jnt_limited[joint] and
                    min(position-self.model.jnt_range[joint, 0], self.model.jnt_range[joint, 1]-position) <= 1e-4)
                    for joint, position in zip(joints, qpos, strict=True)],
            }
        return result

    def capture_global_follow_frame(
        self, width: int, height: int, *, look_at_offset_m: np.ndarray,
        distance_m: float, azimuth_deg: float, elevation_deg: float,
        local_direction_m: np.ndarray | None = None, overlay=None,
    ) -> np.ndarray:
        """以基座跟随的全局视角渲染，可选叠加诊断几何。"""

        offset = np.asarray(look_at_offset_m, dtype=float)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError("全局相机偏置必须是有限三维向量。")
        if not np.all(np.isfinite([distance_m, azimuth_deg, elevation_deg])) or distance_m <= 0:
            raise ValueError("全局相机距离必须为正，角度必须有限。")
        camera = self._mj.MjvCamera()
        camera.type = self._mj.mjtCamera.mjCAMERA_FREE
        base = self._site_transform("mobile_base_frame")
        camera.lookat[:] = base[:3, 3] + offset
        camera.distance = distance_m
        if local_direction_m is None:
            camera.azimuth = azimuth_deg
            camera.elevation = elevation_deg
        else:
            direction = base[:3, :3] @ np.asarray(local_direction_m, dtype=float)
            norm = np.linalg.norm(direction)
            if direction.shape != (3,) or not np.all(np.isfinite(direction)) or norm <= 0:
                raise ValueError("全局相机基座方向必须是非零有限三维向量。")
            direction /= norm
            camera.azimuth = np.degrees(np.arctan2(direction[1], direction[0]))
            camera.elevation = np.degrees(np.arcsin(np.clip(direction[2], -1.0, 1.0)))
        return self._capture_frame(width, height, camera, overlay=overlay)

    def _capture_frame(self, width: int, height: int, camera: Any, overlay=None) -> np.ndarray:
        # 两路分辨率一致，顺序复用离屏渲染器；复制像素后才能切换相机。

        if self._renderer is None:
            self._renderer = self._mj.Renderer(
                self.model,
                height=height,
                width=width,
            )
        self._renderer.update_scene(self.data, camera=camera)
        if overlay is not None:
            overlay(self._renderer.scene)
        return self._renderer.render().copy()

    def _contact_force_world(self, arm: str) -> np.ndarray:
        # 只累计末节工具的环境接触力，避免把内部肌腱/关节反力作为擦拭力。
        tool_body = self._link_bodies[arm][-1]
        force = np.zeros(3)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            if tool_body not in (body1, body2):
                continue
            other = body2 if body1 == tool_body else body1
            if other in self._link_bodies["executor"] or other in self._link_bodies["observer"]:
                continue
            wrench = np.zeros(6)
            self._mj.mj_contactForce(self.model, self.data, index, wrench)
            # 接触坐标系的法向由 geom1 指向 geom2；正反力作用在 geom2。
            force += (1 if body2 == tool_body else -1) * contact.frame.reshape(3, 3).T @ wrench[:3]
        return force

    def _integrate_command(self, command: RobotCommand, dt: float) -> None:
        for arm, arm_command in (
            ("executor", command.executor),
            ("observer", command.observer),
        ):
            rate = np.asarray(arm_command.tendon_rate_mps, dtype=float)
            if rate.shape != self._targets[arm].shape:
                raise ValueError(f"{arm} tendon_rate_mps 维度与模型不一致。")
            self._targets[arm] += dt * rate
        self._write_targets()
        self._integrate_base(command.base_twist_world, dt)

    def _write_targets(self) -> None:
        for arm, actuator_ids in self._actuator_ids.items():
            self.data.ctrl[actuator_ids] = self._targets[arm]

    def _integrate_base(self, twist_world: np.ndarray, dt: float) -> None:
        twist = np.asarray(twist_world, dtype=float)
        self.data.mocap_pos[self._base_mocap] += dt * twist[:3]
        quaternion = self.data.mocap_quat[self._base_mocap].copy()
        delta = _rotation_vector_quaternion(dt * twist[3:])
        self.data.mocap_quat[self._base_mocap] = _quaternion_product(
            delta, quaternion
        )
        self._mj.mj_forward(self.model, self.data)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def stop(self) -> None:
        """结束会话后清除待执行速度，保留已接受的位置目标。"""
        self._pending = None

    def nearest_clearance(self, arm: str, *, allow_tool_contact: bool) -> Clearance:
        """查询真实 XML 碰撞几何；只有擦拭阶段的 Executor 工具—擦拭板允许接触。"""
        best = Clearance(0.05, np.zeros(3), np.zeros(3), "outside_query_range")
        for link, arm_geom in enumerate(self._collision_geoms[arm]):
            for environment_geom in self._environment_geoms:
                if allow_tool_contact and arm == "executor" and link == 11 and environment_geom == self._wiping_board:
                    continue
                segment = np.zeros(6)
                distance = float(self._mj.mj_geomDistance(
                    self.model, self.data, environment_geom, arm_geom, 0.05, segment))
                if distance >= best.distance_m:
                    continue
                delta = segment[3:] - segment[:3]
                norm = float(np.linalg.norm(delta))
                normal = delta / norm if norm > 1e-12 else np.zeros(3)
                if distance < 0:
                    normal = -normal
                body = self._link_bodies[arm][link]
                origin = self.data.xpos[body]
                axis = self.data.xmat[body].reshape(3, 3)[:, 2]
                arc = link * 0.01 + float(np.clip(np.dot(segment[3:] - origin, axis), 0.0, 0.01))
                name = self._mj.mj_id2name(self.model, self._mj.mjtObj.mjOBJ_GEOM, environment_geom)
                best = Clearance(distance, segment[3:].copy(), normal, str(name), arc)
        return best

    def _site_transform(self, name: str) -> np.ndarray:
        site_id = self._site_ids[name]
        transform = np.eye(4)
        transform[:3, :3] = self.data.site_xmat[site_id].reshape(3, 3)
        transform[:3, 3] = self.data.site_xpos[site_id]
        return transform

    def _id(self, object_type: Any, name: str) -> int:
        identifier = int(self._mj.mj_name2id(self.model, object_type, name))
        if identifier < 0:
            raise ValueError(f"MuJoCo XML 缺少命名对象 {name!r}。")
        return identifier


def _import_mujoco() -> Any:
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("使用 MuJoCo 后端前请安装 simulation 可选依赖。") from exc
    return mujoco


def _pose(transform: np.ndarray) -> Pose:
    return Pose(transform[:3, 3], _rotation_matrix_quaternion(transform[:3, :3]))


def _rotation_matrix_quaternion(rotation: np.ndarray) -> np.ndarray:
    """稳定地将旋转矩阵转换为 wxyz 四元数。"""

    matrix = np.asarray(rotation, dtype=float)
    w = np.sqrt(max(0.0, 1.0 + np.trace(matrix))) / 2.0
    x = np.copysign(
        np.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]))
        / 2.0,
        matrix[2, 1] - matrix[1, 2],
    )
    y = np.copysign(
        np.sqrt(max(0.0, 1.0 - matrix[0, 0] + matrix[1, 1] - matrix[2, 2]))
        / 2.0,
        matrix[0, 2] - matrix[2, 0],
    )
    z = np.copysign(
        np.sqrt(max(0.0, 1.0 - matrix[0, 0] - matrix[1, 1] + matrix[2, 2]))
        / 2.0,
        matrix[1, 0] - matrix[0, 1],
    )
    return np.array([w, x, y, z])


def _rotation_vector_quaternion(rotation_vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rotation_vector / angle
    return np.concatenate(([np.cos(angle / 2.0)], axis * np.sin(angle / 2.0)))


def _quaternion_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    result = np.array([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ])
    return result / np.linalg.norm(result)

