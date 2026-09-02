"""Executor 主任务独立求解。"""

import numpy as np

from confined_dual_continuum.kinematics.solver import solve_velocity


class ExecutorController:
    def __init__(self, *, gain: float = 2.0, damping: float = 1.0e-4) -> None:
        self.gain = float(gain)
        self.damping = float(damping)

    def solve(
        self,
        current_position: np.ndarray,
        target_position: np.ndarray,
        jacobian: np.ndarray,
    ) -> np.ndarray:
        velocity = self.gain * (
            np.asarray(target_position, dtype=float)
            - np.asarray(current_position, dtype=float)
        )
        return solve_velocity(jacobian, velocity, damping=self.damping)

