"""跨控制层传递的稳定状态与命令合同。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from confined_dual_continuum.domain.geometry import Pose, vector


@dataclass(frozen=True)
class ArmState:
    tip_pose: Pose
    bending: np.ndarray
    tendon_displacement_m: np.ndarray
    centerline_world: np.ndarray
    contact_force_world: np.ndarray

    def __post_init__(self) -> None:
        bending = np.asarray(self.bending, dtype=float)
        tendon = np.asarray(self.tendon_displacement_m, dtype=float)
        centerline = np.asarray(self.centerline_world, dtype=float)
        force = vector(self.contact_force_world, 3, "contact_force_world")
        if bending.ndim != 1 or tendon.ndim != 1:
            raise ValueError("bending 和 tendon_displacement_m 必须是一维向量。")
        if centerline.ndim != 2 or centerline.shape[1] != 3:
            raise ValueError("centerline_world 必须具有形状 (N, 3)。")
        if not all(np.all(np.isfinite(item)) for item in (bending, tendon, centerline)):
            raise ValueError("机械臂状态必须为有限值。")
        object.__setattr__(self, "bending", bending.copy())
        object.__setattr__(self, "tendon_displacement_m", tendon.copy())
        object.__setattr__(self, "centerline_world", centerline.copy())
        object.__setattr__(self, "contact_force_world", force)


@dataclass(frozen=True)
class RobotState:
    base_pose: Pose
    executor: ArmState
    observer: ArmState


@dataclass(frozen=True)
class ArmCommand:
    bending_rate: np.ndarray
    tendon_rate_mps: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        bending = np.asarray(self.bending_rate, dtype=float)
        tendon = np.asarray(self.tendon_rate_mps, dtype=float)
        if bending.ndim != 1 or tendon.ndim != 1:
            raise ValueError("机械臂命令必须是一维向量。")
        object.__setattr__(self, "bending_rate", bending.copy())
        object.__setattr__(self, "tendon_rate_mps", tendon.copy())


@dataclass(frozen=True)
class RobotCommand:
    base_twist_world: np.ndarray
    executor: ArmCommand
    observer: ArmCommand

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_twist_world",
            vector(self.base_twist_world, 6, "base_twist_world"),
        )
