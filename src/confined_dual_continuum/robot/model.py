"""专利样机机器人参数模型。"""

from __future__ import annotations

from dataclasses import dataclass


import numpy as np


from confined_dual_continuum.actuation.bending_space import BendingSpaceModel
from confined_dual_continuum.actuation.compatible_tendon_control import TendonLimits
from confined_dual_continuum.actuation.tendon_coupling import TendonRoute, build_bending_coupling


@dataclass(frozen=True)
class RobotModel:
    """控制算法使用的物理参数，不包含任何仿真对象。"""

    flexure_lengths_m: tuple[float, ...]
    distal_offsets_m: tuple[float, ...]
    tendon_routes: dict[str, tuple[TendonRoute, ...]]
    limits: TendonLimits

    def __post_init__(self) -> None:
        if not self.flexure_lengths_m or any(
            not np.isfinite(length) or length <= 0.0 for length in self.flexure_lengths_m
        ):
            raise ValueError("所有柔性弯曲段长度必须为正。")
        if len(self.distal_offsets_m) != len(self.flexure_lengths_m) or any(
            not np.isfinite(offset) or offset < 0.0 for offset in self.distal_offsets_m
        ):
            raise ValueError("段末刚性偏置必须与柔性段一一对应且不能为负。")
        if set(self.tendon_routes) != {"executor", "observer"}:
            raise ValueError("必须分别定义 executor 和 observer 肌腱路由。")
        if any(self.limits.minimum_m.size != len(routes) for routes in self.tendon_routes.values()):
            raise ValueError("肌腱约束数量必须与路由数量一致。")

    def bending_model(self, arm: str) -> BendingSpaceModel:
        return BendingSpaceModel(
            build_bending_coupling(self.flexure_lengths_m, self.tendon_routes[arm])
        )



