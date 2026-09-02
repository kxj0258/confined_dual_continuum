"""将稳定的机器人合同映射到专利样机 MuJoCo XML。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from confined_dual_continuum.domain.geometry import Pose
from confined_dual_continuum.domain.state import ArmState, RobotCommand, RobotState
from confined_dual_continuum.kinematics.pcc import forward_kinematics, transform_points
from confined_dual_continuum.robot.model import RobotModel


class MujocoRobotBackend:
    """薄适配器：只负责反馈读取、命令映射和物理步进。"""

    def __init__(self, xml_path: str | Path, robot: RobotModel) -> None:
        self._mj = _import_mujoco()
        self.model = self._mj.MjModel.from_xml_path(str(Path(xml_path).resolve()))
        self.data = self._mj.MjData(self.model)
        self.robot = robot
        self._bending_model = robot.bending_model
        self._base_joint = self._id(
            self._mj.mjtObj.mjOBJ_JOINT, "mobile_base_freejoint"
        )
        self._base_qpos = int(self.model.jnt_qposadr[self._base_joint])
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
                    for index in range(1, self._bending_model.tendon_count + 1)
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
                    for index in range(1, self._bending_model.tendon_count + 1)
                ]
            )
            for arm in ("executor", "observer")
        }
        self._pending: RobotCommand | None = None
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
        if dt <= 0.0:
            raise ValueError("dt 必须为正。")
        if self._pending is not None:
            self._integrate_command(self._pending, dt)
        steps = max(1, int(round(dt / float(self.model.opt.timestep))))
        for _ in range(steps):
            self._mj.mj_step(self.model, self.data)

    def _read_arm(self, arm: str) -> ArmState:
        tendon = self.data.ten_length[self._tendon_ids[arm]] - self._neutral[arm]
        bending = self._bending_model.estimate(tendon)
        local = forward_kinematics(bending, self.robot.segment_lengths_m)
        centerline = transform_points(
            self._site_transform(f"{arm}_base_site"), local.centerline
        )
        return ArmState(
            tip_pose=_pose(self._site_transform(f"{arm}_tip")),
            bending=bending,
            tendon_displacement_m=tendon,
            centerline_world=centerline,
        )

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
        self.data.qpos[self._base_qpos : self._base_qpos + 3] += dt * twist[:3]
        quaternion = self.data.qpos[self._base_qpos + 3 : self._base_qpos + 7].copy()
        delta = _rotation_vector_quaternion(dt * twist[3:])
        self.data.qpos[self._base_qpos + 3 : self._base_qpos + 7] = _quaternion_product(
            delta, quaternion
        )
        self._mj.mj_forward(self.model, self.data)

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
