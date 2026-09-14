"""非对称双臂编排：分别求解，只在最终命令层合并。"""

from dataclasses import dataclass

import numpy as np

from confined_dual_continuum.coordination.executor_controller import ExecutorController
from confined_dual_continuum.coordination.interarm_safety import (
    InterArmSafety,
    InterArmSafetyDecision,
)
from confined_dual_continuum.coordination.observer_controller import ObserverController
from confined_dual_continuum.domain.state import ArmCommand, RobotCommand, RobotState
from confined_dual_continuum.domain.geometry import quaternion_rotation
from confined_dual_continuum.task.phase_machine import PhaseActivation


@dataclass(frozen=True)
class CoordinationTargets:
    executor_position_world: np.ndarray
    observer_position_world: np.ndarray
    surface_normal_world: np.ndarray
    observer_view_direction_world: np.ndarray
    executor_force_target_n: float = 0.0


class DualArmCoordinator:
    def __init__(self) -> None:
        self.executor = ExecutorController()
        self.observer = ObserverController()
        self.safety = InterArmSafety()

    def solve(
        self,
        state: RobotState,
        targets: CoordinationTargets,
        activation: PhaseActivation,
        executor_jacobian: np.ndarray,
        observer_jacobian: np.ndarray,
        base_twist_world: np.ndarray,
        *,
        safety: InterArmSafetyDecision | None = None,
        avoidance_arm: str = "observer",
        executor_safety_jacobian: np.ndarray,
        observer_safety_jacobian: np.ndarray,
        observer_direction_jacobian: np.ndarray,
    ) -> RobotCommand:
        if avoidance_arm not in {"executor", "observer"}:
            raise ValueError("avoidance_arm 必须为 executor 或 observer")
        safety = safety or self.evaluate_safety(state)
        if activation.executor_enabled and targets.executor_force_target_n > 0.0:
            executor_rate = self.executor.solve_force_position(
                state.executor.tip_pose.position,
                targets.executor_position_world,
                executor_jacobian,
                targets.surface_normal_world,
                state.executor.contact_force_world,
                targets.executor_force_target_n,
            )
        elif activation.executor_enabled:
            executor_rate = self.executor.solve(
                state.executor.tip_pose.position,
                targets.executor_position_world,
                executor_jacobian,
            )
        else:
            executor_rate = np.zeros(executor_jacobian.shape[1])
        if activation.observer_enabled:
            observer_rate = self.observer.solve_view(
                state.observer.tip_pose.position,
                targets.observer_position_world,
                observer_jacobian,
                quaternion_rotation(state.observer.tip_pose.quaternion_wxyz)[:, 2],
                targets.observer_view_direction_world,
                observer_direction_jacobian,
            )
        else:
            observer_rate = np.zeros(observer_jacobian.shape[1])
        if safety.safety_stop:
            executor_rate = np.zeros_like(executor_rate)
            observer_rate = np.zeros_like(observer_rate)
        elif safety.avoidance_active and avoidance_arm == "observer" and activation.observer_enabled:
            observer_rate = self.observer.solve_velocity(
                safety.avoidance_velocity_world,
                observer_safety_jacobian,
            )
        elif safety.avoidance_active and avoidance_arm == "executor" and activation.executor_enabled:
            executor_rate = self.executor.solve_velocity(
                -safety.avoidance_velocity_world,
                executor_safety_jacobian,
            )
        base = np.asarray(base_twist_world, dtype=float) if activation.base_enabled and not safety.safety_stop else np.zeros(6)
        return RobotCommand(
            base_twist_world=base,
            executor=ArmCommand(executor_rate),
            observer=ArmCommand(observer_rate),
        )

    def evaluate_safety(self, state: RobotState) -> InterArmSafetyDecision:
        return self.safety.evaluate(
            state.executor.centerline_world,
            state.observer.centerline_world,
        )
