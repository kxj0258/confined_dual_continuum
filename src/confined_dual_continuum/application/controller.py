"""唯一专利控制入口，按任务层→协同层→执行层顺序调用。"""

from dataclasses import dataclass

import numpy as np

from confined_dual_continuum.actuation.compatible_tendon_control import (
    CompatibleTendonController,
)
from confined_dual_continuum.coordination.dual_arm_coordinator import (
    CoordinationTargets,
    DualArmCoordinator,
)
from confined_dual_continuum.coordination.observer_target import ObserverTargetGenerator
from confined_dual_continuum.diagnostics.records import ControlDiagnostics
from confined_dual_continuum.domain.state import ArmCommand, RobotCommand, RobotState
from confined_dual_continuum.task.confined_operation import (
    ConfinedOperationSupervisor,
    OperationSignals,
)


@dataclass(frozen=True)
class ControlInput:
    state: RobotState
    signals: OperationSignals
    executor_target_world: np.ndarray
    task_point_world: np.ndarray
    surface_normal_world: np.ndarray
    surface_tangent_world: np.ndarray
    executor_jacobian: np.ndarray
    observer_jacobian: np.ndarray
    requested_base_twist_world: np.ndarray
    dt: float


@dataclass(frozen=True)
class ControlOutput:
    command: RobotCommand
    diagnostics: ControlDiagnostics


class ConfinedDualContinuumController:
    def __init__(
        self,
        executor_actuation: CompatibleTendonController,
        observer_actuation: CompatibleTendonController,
    ) -> None:
        self.supervisor = ConfinedOperationSupervisor()
        self.observer_target = ObserverTargetGenerator()
        self.coordinator = DualArmCoordinator()
        self.executor_actuation = executor_actuation
        self.observer_actuation = observer_actuation

    def step(self, request: ControlInput) -> ControlOutput:
        phase = self.supervisor.update(request.signals)
        observer_target = self.observer_target.generate(
            request.task_point_world,
            request.surface_normal_world,
            request.surface_tangent_world,
            request.dt,
        )
        bending_command = self.coordinator.solve(
            request.state,
            CoordinationTargets(
                executor_position_world=request.executor_target_world,
                observer_position_world=observer_target.position_world,
            ),
            phase.activation,
            request.executor_jacobian,
            request.observer_jacobian,
            request.requested_base_twist_world,
        )
        executor = self.executor_actuation.step(
            bending_command.executor.bending_rate,
            request.state.executor.tendon_displacement_m,
            request.dt,
        )
        observer = self.observer_actuation.step(
            bending_command.observer.bending_rate,
            request.state.observer.tendon_displacement_m,
            request.dt,
        )
        command = RobotCommand(
            base_twist_world=bending_command.base_twist_world,
            executor=ArmCommand(executor.bending_rate, executor.tendon_rate_mps),
            observer=ArmCommand(observer.bending_rate, observer.tendon_rate_mps),
        )
        distance = float(np.min(np.linalg.norm(
            request.state.observer.centerline_world[:, None, :]
            - request.state.executor.centerline_world[None, :, :],
            axis=2,
        )))
        return ControlOutput(
            command=command,
            diagnostics=ControlDiagnostics(
                phase=phase.phase.value,
                phase_step=phase.phase_step,
                transition_reason=phase.transition_reason,
                base_enabled=phase.activation.base_enabled,
                executor_enabled=phase.activation.executor_enabled,
                observer_enabled=phase.activation.observer_enabled,
                inter_arm_distance_m=distance,
                executor_compatibility_residual=executor.compatibility_residual_norm,
                observer_compatibility_residual=observer.compatibility_residual_norm,
            ),
        )

