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
from confined_dual_continuum.diagnostics.records import ControlDiagnostics
from confined_dual_continuum.domain.state import ArmCommand, RobotCommand, RobotState
from confined_dual_continuum.kinematics.pcc import position_jacobian, direction_jacobian, material_point
from confined_dual_continuum.domain.geometry import quaternion_rotation
from confined_dual_continuum.robot.model import RobotModel
from confined_dual_continuum.task.confined_operation import (
    ConfinedOperationSupervisor,
    OperationSignals,
)
from confined_dual_continuum.task.phase_machine import ConfinedOperationPhase, PhaseActivation
from confined_dual_continuum.workspace.query import WorkspaceQuery


@dataclass(frozen=True)
class ControlInput:
    state: RobotState
    signals: OperationSignals
    executor_target_world: np.ndarray
    observer_target_world: np.ndarray
    surface_normal_world: np.ndarray
    executor_force_target_n: float
    avoidance_arm: str
    requested_base_twist_world: np.ndarray
    dt: float
    observer_view_direction_world: np.ndarray
    waypoint_index: int


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
        environment: WorkspaceQuery,
        *,
        staged_activation: bool = True,
        observer_task_assist: bool = True,
    ) -> None:
        self.supervisor = ConfinedOperationSupervisor()
        self.coordinator = DualArmCoordinator()
        self.executor_actuation = executor_actuation
        self.observer_actuation = observer_actuation
        self.robot = robot
        self.environment = environment
        self.staged_activation = staged_activation
        self.observer_task_assist = observer_task_assist
        self._fixed_observer_target = None
        self._fixed_observer_phase = None

    def reset(self, state: RobotState) -> None:
        self.supervisor = ConfinedOperationSupervisor()
        self.coordinator = DualArmCoordinator()
        self.executor_actuation.reset(state.executor.tendon_displacement_m)
        self.observer_actuation.reset(state.observer.tendon_displacement_m)
        self._fixed_observer_target = None
        self._fixed_observer_phase = None

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
        task_observer_target = request.observer_target_world.copy()
        if not self.observer_task_assist:
            if self.phase != self._fixed_observer_phase:
                self._fixed_observer_phase = self.phase
                self._fixed_observer_target = request.observer_target_world.copy()
            request = replace(request, observer_target_world=self._fixed_observer_target)
        safety = self.coordinator.evaluate_safety(request.state)
        clearances = {
            arm: self.environment.nearest_clearance(arm, allow_tool_contact=(
                arm == "executor" and self.phase in (
                    ConfinedOperationPhase.FORCE_POSITION_WIPING,
                    ConfinedOperationPhase.OBSTACLE_TRANSIT)))
            for arm in ("executor", "observer")
        }
        signals = (
            replace(request.signals, safety_violation="inter_arm_critical_distance")
            if safety.safety_stop
            else request.signals
        )
        if min(item.distance_m for item in clearances.values()) <= 0.0005:
            signals = replace(signals, safety_violation="environment_critical_clearance")
        for clearance in clearances.values():
            predicted_gap = clearance.distance_m + request.dt * float(
                clearance.normal_world @ request.requested_base_twist_world[:3])
            if predicted_gap <= 0.0005:
                signals = replace(signals, safety_violation="base_motion_environment_clearance")
        if max(np.linalg.norm(request.state.executor.contact_force_world),
               np.linalg.norm(request.state.observer.contact_force_world)) > 3.0:
            signals = replace(signals, safety_violation="maximum_contact_force")
        previous_phase = self.phase
        phase = self.supervisor.update(signals)
        # 转换周期只保持目标；下一周期 Director 使用新阶段生成一致的目标与角色。
        activation = phase.activation if phase.phase == previous_phase else PhaseActivation(False, False, False)
        if not self.staged_activation and phase.phase == previous_phase and not self.terminal:
            activation = PhaseActivation(True, True, True)
        rotation = quaternion_rotation(request.state.base_pose.quaternion_wxyz)
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
            state=request.state,
            targets=CoordinationTargets(
                executor_position_world=request.executor_target_world,
                observer_position_world=request.observer_target_world,
                surface_normal_world=request.surface_normal_world,
                executor_force_target_n=request.executor_force_target_n,
                observer_view_direction_world=request.observer_view_direction_world,
            ),
            activation=activation,
            executor_jacobian=executor_jacobian,
            observer_jacobian=observer_jacobian,
            base_twist_world=request.requested_base_twist_world,
            safety=safety,
            avoidance_arm=request.avoidance_arm,
            executor_safety_jacobian=rotation @ position_jacobian(
                request.state.executor.bending, self.robot.flexure_lengths_m,
                self.robot.distal_offsets_m, arc_m=safety.executor_arc_m),
            observer_safety_jacobian=rotation @ position_jacobian(
                request.state.observer.bending, self.robot.flexure_lengths_m,
                self.robot.distal_offsets_m, arc_m=safety.observer_arc_m),
            observer_direction_jacobian=rotation @ direction_jacobian(
                request.state.observer.bending, self.robot.flexure_lengths_m, self.robot.distal_offsets_m),
        )
        executor = self.executor_actuation.step(
            self._environment_rate(bending_command.executor.bending_rate, clearances["executor"],
                                   request.state.executor.bending, rotation, activation.executor_enabled),
            request.state.executor.tendon_displacement_m,
            request.dt,
            allow_recovery=activation.executor_enabled,
        )
        observer = self.observer_actuation.step(
            self._environment_rate(bending_command.observer.bending_rate, clearances["observer"],
                                   request.state.observer.bending, rotation, activation.observer_enabled),
            request.state.observer.tendon_displacement_m,
            request.dt,
            allow_recovery=activation.observer_enabled,
        )
        infeasible = not (executor.feasible and observer.feasible)
        rejection = "tendon_constraints_infeasible" if infeasible else None
        # 恢复改变了请求方向，必须再次检查最终命令是否破坏环境避让约束。
        for arm, result in (("executor", executor), ("observer", observer)):
            clearance = clearances[arm]
            if clearance.distance_m < 0.008:
                jacobian = rotation @ position_jacobian(getattr(request.state, arm).bending,
                    self.robot.flexure_lengths_m, self.robot.distal_offsets_m, arc_m=clearance.arm_arc_m)
                velocity = jacobian @ result.bending_rate + bending_command.base_twist_world[:3]
                if float(clearance.normal_world @ velocity) < -1e-9:
                    rejection = "environment_final_velocity_conflict"
        # 对环境投影及两臂分别限幅后的最终速度再检查；不得靠后续覆盖破坏双臂安全。
        if safety.avoidance_active:
            direction = safety.avoidance_velocity_world
            norm = float(np.linalg.norm(direction))
            if norm > 1e-12:
                ex_j = rotation @ position_jacobian(request.state.executor.bending,
                    self.robot.flexure_lengths_m, self.robot.distal_offsets_m, arc_m=safety.executor_arc_m)
                ob_j = rotation @ position_jacobian(request.state.observer.bending,
                    self.robot.flexure_lengths_m, self.robot.distal_offsets_m, arc_m=safety.observer_arc_m)
                separation_rate = float(direction @ (ob_j @ observer.bending_rate - ex_j @ executor.bending_rate) / norm)
                if separation_rate < -1e-9:
                    rejection = "inter_arm_final_velocity_conflict"
        if rejection:
            phase = self.supervisor.update(OperationSignals(safety_violation=rejection))
            activation = PhaseActivation(False, False, False)
            self.executor_actuation.cancel_step(executor, request.dt)
            self.observer_actuation.cancel_step(observer, request.dt)
        infeasible = rejection is not None
        command = RobotCommand(
            base_twist_world=np.zeros(6) if infeasible else bending_command.base_twist_world,
            executor=ArmCommand(np.zeros_like(executor.bending_rate) if infeasible else executor.bending_rate,
                                np.zeros_like(executor.tendon_rate_mps) if infeasible else executor.tendon_rate_mps,
                                executor.target_m - request.dt * executor.tendon_rate_mps if infeasible else executor.target_m),
            observer=ArmCommand(np.zeros_like(observer.bending_rate) if infeasible else observer.bending_rate,
                                np.zeros_like(observer.tendon_rate_mps) if infeasible else observer.tendon_rate_mps,
                                observer.target_m - request.dt * observer.tendon_rate_mps if infeasible else observer.target_m),
        )
        return ControlOutput(
            command=command,
            diagnostics=ControlDiagnostics(
                phase=phase.phase.value,
                phase_step=phase.phase_step,
                transition_reason=phase.transition_reason,
                base_enabled=activation.base_enabled,
                executor_enabled=activation.executor_enabled,
                observer_enabled=activation.observer_enabled,
                inter_arm_distance_m=safety.distance_m,
                executor_compatibility_residual=executor.compatibility_residual_norm,
                observer_compatibility_residual=observer.compatibility_residual_norm,
                inter_arm_avoidance_active=safety.avoidance_active,
                safety_stop=phase.phase == ConfinedOperationPhase.SAFETY_STOP,
                executor_contact_force_n=float(
                    np.linalg.norm(request.state.executor.contact_force_world)
                ),
                avoidance_arm=request.avoidance_arm,
                target_phase=previous_phase.value,
                waypoint_index=request.waypoint_index,
                executor_tracking_error_m=float(np.linalg.norm(request.executor_target_world-request.state.executor.tip_pose.position)),
                observer_tracking_error_m=float(np.linalg.norm(task_observer_target-request.state.observer.tip_pose.position)),
                observer_view_error_rad=float(np.arccos(np.clip(
                    quaternion_rotation(request.state.observer.tip_pose.quaternion_wxyz)[:, 2] @ request.observer_view_direction_world, -1, 1))),
                executor_force_error_n=request.executor_force_target_n-float(request.state.executor.contact_force_world @ request.surface_normal_world),
                environment_clearance_m=min(item.distance_m for item in clearances.values()),
                executor_actual_residual=executor.actual_residual_norm,
                observer_actual_residual=observer.actual_residual_norm,
                executor_lead_m=executor.maximum_lead_m,
                observer_lead_m=observer.maximum_lead_m,
                executor_scale=executor.common_scale,
                observer_scale=observer.common_scale,
                constraints_feasible=executor.feasible and observer.feasible,
                executor_pcc_tip_error_m=self._pcc_error(request.state.executor, rotation),
                observer_pcc_tip_error_m=self._pcc_error(request.state.observer, rotation),
                executor_constraint_status=executor.constraint_status,
                observer_constraint_status=observer.constraint_status,
                executor_limiting_tendon_index=executor.limiting_tendon_index,
                observer_limiting_tendon_index=observer.limiting_tendon_index,
                executor_lead_excess_m=executor.lead_excess_m,
                observer_lead_excess_m=observer.lead_excess_m,
            ),
        )

    def _environment_rate(self, rate, clearance, bending, rotation, enabled):
        if not enabled or clearance.distance_m >= 0.008:
            return rate
        jacobian = rotation @ position_jacobian(
            bending, self.robot.flexure_lengths_m, self.robot.distal_offsets_m,
            arc_m=clearance.arm_arc_m)
        gradient = clearance.normal_world @ jacobian
        norm_squared = float(gradient @ gradient)
        # 局部单约束投影；可行速度不再朝环境继续接近，所有输出仍经相容限幅。
        required = 2.0 * max(0.003 - clearance.distance_m, 0.0)
        deficit = required - float(gradient @ rate)
        if deficit > 0:
            if norm_squared <= 1e-16:
                return np.zeros_like(rate)
            return rate + deficit / norm_squared * gradient
        return rate

    def _pcc_error(self, state, rotation):
        if state.pcc_tip_error_m is not None:
            return state.pcc_tip_error_m
        expected = state.centerline_world[0] + rotation @ material_point(
            state.bending, self.robot.flexure_lengths_m, self.robot.distal_offsets_m)
        return float(np.linalg.norm(expected - state.tip_pose.position))
