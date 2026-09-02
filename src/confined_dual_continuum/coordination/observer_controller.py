"""Observer 辅助任务独立求解；不接受基座自由度。"""

import numpy as np

from confined_dual_continuum.kinematics.solver import solve_velocity


class ObserverController:
    def __init__(self, *, gain: float = 2.0, damping: float = 1.0e-4) -> None:
        self.gain = float(gain)
        self.damping = float(damping)

    def solve(self, current: np.ndarray, target: np.ndarray, arm_jacobian: np.ndarray) -> np.ndarray:
        velocity = self.gain * (np.asarray(target) - np.asarray(current))
        return solve_velocity(arm_jacobian, velocity, damping=self.damping)

    def solve_velocity(self, velocity_world: np.ndarray, arm_jacobian: np.ndarray) -> np.ndarray:
        """安全层可直接覆盖辅助目标，但仍只使用 Observer 自身自由度。"""

        return solve_velocity(arm_jacobian, velocity_world, damping=self.damping)
