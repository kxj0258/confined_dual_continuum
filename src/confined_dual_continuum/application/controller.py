"""唯一专利控制入口，按任务层→协同层→执行层顺序调用。"""

from dataclasses import dataclass, replace

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
from confined_dual_continuum.kinematics.pcc import position_jacobian
from confined_dual_continuum.robot.model import RobotModel
from confined_dual_continuum.task.confined_operation import (
    ConfinedOperationSupervisor,
    OperationSignals,
)
from confined_dual_continuum.task.phase_machine import ConfinedOperationPhase


@dataclass(frozen=True)
class ControlInput:
    state: RobotState
    signals: OperationSignals
    executor_target_world: np.ndarray
    task_point_world: np.ndarray
    surface_normal_world: np.ndarray
    surface_tangent_world: np.ndarray
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
        robot: RobotModel,
    ) -> None:
        self.supervisor = ConfinedOperationSupervisor()
        self.observer_target = ObserverTargetGenerator()
        self.coordinator = DualArmCoordinator()
        self.executor_actuation = executor_actuation
        self.observer_actuation = observer_actuation
        self.robot = robot

    @property
    def phase(self) -> ConfinedOperationPhase:
        return self.supervisor.machine.phase

    @property
    def phase_step(self) -> int:
        return self.supervisor.machine.phase_step

    @property
    def terminal(self) -> bool:
        return self.supervisor.machine.terminal

    def step(self, request: ControlInput) -> ControlOutput:
        safety = self.coordinator.evaluate_safety(request.state)
        signals = (
            replace(request.signals, safety_violation="inter_arm_critical_distance")
            if safety.safety_stop
            else request.signals
        )
        phase = self.supervisor.update(signals)
        observer_target = self.observer_target.generate(
            request.task_point_world,
            request.surface_normal_world,
            request.surface_tangent_world,
            request.dt,
        )
        rotation = _quaternion_rotation(request.state.base_pose.quaternion_wxyz)
        executor_jacobian = rotation @ position_jacobian(
            request.state.executor.bending,
            self.robot.flexure_lengths_m,
            self.robot.distal_offsets_m,
        )
        observer_jacobian = rotation @ position_jacobian(
            request.state.observer.bending,
            self.robot.flexure_lengths_m,
            self.robot.distal_offsets_m,
        )
        bending_command = self.coordinator.solve(
            request.state,
            CoordinationTargets(
                executor_position_world=request.executor_target_world,
                observer_position_world=observer_target.position_world,
            ),
            phase.activation,
            executor_jacobian,
            observer_jacobian,
            request.requested_base_twist_world,
            safety,
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
        return ControlOutput(
            command=command,
            diagnostics=ControlDiagnostics(
                phase=phase.phase.value,
                phase_step=phase.phase_step,
                transition_reason=phase.transition_reason,
                base_enabled=phase.activation.base_enabled,
                executor_enabled=phase.activation.executor_enabled,
                observer_enabled=phase.activation.observer_enabled,
                inter_arm_distance_m=safety.distance_m,
                executor_compatibility_residual=executor.compatibility_residual_norm,
                observer_compatibility_residual=observer.compatibility_residual_norm,
                inter_arm_avoidance_active=safety.avoidance_active,
                safety_stop=safety.safety_stop,
            ),
        )


def _quaternion_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=float)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
