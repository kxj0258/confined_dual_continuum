"""主臂独立跟踪的最小反馈合同。"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ExecutorFeedback:
    tip_world_m: np.ndarray
    tendon_displacement_m: np.ndarray
    minimum_joint_margin_rad: float
