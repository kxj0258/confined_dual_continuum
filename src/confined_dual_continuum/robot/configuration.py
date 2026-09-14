"""机器人 YAML 装载边界；控制核心只导入纯参数模型。"""
from pathlib import Path
import numpy as np
import yaml
from confined_dual_continuum.robot.model import RobotModel
from confined_dual_continuum.actuation.tendon_coupling import TendonRoute
from confined_dual_continuum.actuation.compatible_tendon_control import TendonLimits

def load_robot_model(path: str | Path) -> RobotModel:
    """在配置边界读取 YAML，内部各层只接收强类型参数。"""

    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    flexure_lengths = tuple(float(value) for value in values["flexure_lengths_m"])
    distal_offsets = tuple(float(value) for value in values["distal_offsets_m"])
    routes = {arm: tuple(
        TendonRoute(
            angle_deg=float(item["angle_deg"]),
            radial_offset_m=float(item["radial_offset_m"]),
            path_segment_indices=tuple(
                int(index) for index in item["path_segment_indices"]
            ),
        )
        for item in arm_routes
    ) for arm, arm_routes in values["tendon_routes"].items()}
    count = len(routes["executor"])
    limits = values["limits"]
    if not isinstance(limits["enabled"], bool):
        raise ValueError("limits.enabled 必须为布尔值。")
    return RobotModel(
        flexure_lengths_m=flexure_lengths,
        distal_offsets_m=distal_offsets,
        tendon_routes=routes,
        limits=TendonLimits(
            enabled=limits["enabled"],
            rate_enabled=limits.get("rate_enabled", False),
            maximum_curvature_per_m=limits.get("maximum_curvature_per_m"),
            minimum_m=np.full(count, float(limits["minimum_m"])),
            maximum_m=np.full(count, float(limits["maximum_m"])),
            maximum_rate_mps=np.full(count, float(limits["maximum_rate_mps"])),
            maximum_lead_m=np.full(count, float(limits["maximum_lead_m"])),
        ),
    )

