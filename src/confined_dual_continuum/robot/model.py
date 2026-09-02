"""专利样机机器人参数模型。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from confined_dual_continuum.actuation.bending_space import BendingSpaceModel
from confined_dual_continuum.actuation.compatible_tendon_control import TendonLimits
from confined_dual_continuum.actuation.tendon_coupling import TendonRoute, build_bending_coupling


@dataclass(frozen=True)
class RobotModel:
    """控制算法使用的物理参数，不包含任何仿真对象。"""

    flexure_lengths_m: tuple[float, ...]
    distal_offsets_m: tuple[float, ...]
    tendon_routes: tuple[TendonRoute, ...]
    limits: TendonLimits

    def __post_init__(self) -> None:
        if not self.flexure_lengths_m or any(
            length <= 0.0 for length in self.flexure_lengths_m
        ):
            raise ValueError("所有柔性弯曲段长度必须为正。")
        if len(self.distal_offsets_m) != len(self.flexure_lengths_m) or any(
            offset < 0.0 for offset in self.distal_offsets_m
        ):
            raise ValueError("段末刚性偏置必须与柔性段一一对应且不能为负。")
        if not self.tendon_routes:
            raise ValueError("至少需要一条肌腱路由。")
        if self.limits.minimum_m.size != len(self.tendon_routes):
            raise ValueError("肌腱约束数量必须与路由数量一致。")

    @property
    def bending_model(self) -> BendingSpaceModel:
        return BendingSpaceModel(
            build_bending_coupling(self.flexure_lengths_m, self.tendon_routes)
        )


def load_robot_model(path: str | Path) -> RobotModel:
    """在配置边界读取 YAML，内部各层只接收强类型参数。"""

    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    flexure_lengths = tuple(float(value) for value in values["flexure_lengths_m"])
    distal_offsets = tuple(float(value) for value in values["distal_offsets_m"])
    routes = tuple(
        TendonRoute(
            angle_deg=float(item["angle_deg"]),
            radial_offset_m=float(item["radial_offset_m"]),
            path_segment_indices=tuple(
                int(index) for index in item["path_segment_indices"]
            ),
        )
        for item in values["tendon_routes"]
    )
    count = len(routes)
    limits = values["limits"]
    return RobotModel(
        flexure_lengths_m=flexure_lengths,
        distal_offsets_m=distal_offsets,
        tendon_routes=routes,
        limits=TendonLimits(
            minimum_m=np.full(count, float(limits["minimum_m"])),
            maximum_m=np.full(count, float(limits["maximum_m"])),
            maximum_rate_mps=np.full(count, float(limits["maximum_rate_mps"])),
            maximum_lead_m=np.full(count, float(limits["maximum_lead_m"])),
        ),
    )
