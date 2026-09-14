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

    def solve_velocity(
        self,
        velocity_world: np.ndarray,
        jacobian: np.ndarray,
    ) -> np.ndarray:
        return solve_velocity(jacobian, velocity_world, damping=self.damping)

    def solve_force_position(
        self,
        current_position: np.ndarray,
        target_position: np.ndarray,
        jacobian: np.ndarray,
        surface_normal_world: np.ndarray,
        measured_force_world: np.ndarray,
        target_force_n: float,
        force_gain: float = 0.003,
    ) -> np.ndarray:
        normal = np.asarray(surface_normal_world, dtype=float)
        norm = np.linalg.norm(normal)
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError("接触面法向必须为有限非零向量。")
        normal = normal / norm
        position_error = np.asarray(target_position) - np.asarray(current_position)
        tangential_error = position_error - normal * np.dot(position_error, normal)
        measured_force = max(0.0, float(np.dot(measured_force_world, normal)))
        # 自由空间侧法向指向机械臂，增大接触力时沿反法向压向表面。
        normal_velocity = -float(np.clip(force_gain * (target_force_n - measured_force), -0.003, 0.003)) * normal
        return solve_velocity(
            jacobian,
            self.gain * tangential_error + normal_velocity,
            damping=self.damping,
        )
