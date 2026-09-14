"""固定基座跟随的纯几何查询结果。"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TrackingClearance:
    arm: str
    arc_m: float
    distance_m: float
    normal_world: np.ndarray
    other_arm: str | None = None
    other_arc_m: float = 0.
    source: str = ''


@dataclass(frozen=True)
class TrackingGeometry:
    camera_world: np.ndarray
    executor_root_world: np.ndarray
    observer_root_world: np.ndarray
    clearances: tuple[TrackingClearance, ...]
    occluded: bool
    joint_margin_rad: dict[str, float]


@dataclass(frozen=True)
class ObservationCandidate:
    sample: int
    bending: np.ndarray
    tip_root: np.ndarray
