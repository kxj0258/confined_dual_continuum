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
from confined_dual_continuum.task.phase_machine import PhaseActivation


@dataclass(frozen=True)
class CoordinationTargets:
    executor_position_world: np.ndarray
    observer_position_world: np.ndarray


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
        safety: InterArmSafetyDecision | None = None,
    ) -> RobotCommand:
        safety = safety or self.evaluate_safety(state)
        executor_rate = (
            self.executor.solve(
                state.executor.tip_pose.position,
                targets.executor_position_world,
                executor_jacobian,
            )
            if activation.executor_enabled
            else np.zeros(executor_jacobian.shape[1])
        )
        if not activation.observer_enabled or safety.safety_stop:
            observer_rate = np.zeros(observer_jacobian.shape[1])
        elif safety.avoidance_active:
            observer_rate = self.observer.solve_velocity(
                safety.avoidance_velocity_world,
                observer_jacobian,
            )
        else:
            observer_rate = self.observer.solve(
                state.observer.tip_pose.position,
                targets.observer_position_world,
                observer_jacobian,
            )
        base = np.asarray(base_twist_world, dtype=float) if activation.base_enabled else np.zeros(6)
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
