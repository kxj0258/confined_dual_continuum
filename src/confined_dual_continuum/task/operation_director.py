"""把任务阶段转换为几何目标和阶段完成信号。"""

from dataclasses import dataclass

import numpy as np

from confined_dual_continuum.domain.state import RobotState
from confined_dual_continuum.task.confined_operation import OperationSignals
from confined_dual_continuum.task.phase_machine import ConfinedOperationPhase
from confined_dual_continuum.workspace.model import ConfinedWorkspace


@dataclass(frozen=True)
class OperationPlan:
    position_tolerance_m: float
    base_gain: float
    maximum_base_speed_mps: float
    local_operation_duration_s: float
    bending_tolerance_inv_m: float
    straight_executor_tip_offset_m: np.ndarray

    def __post_init__(self) -> None:
        values = (
            self.position_tolerance_m,
            self.base_gain,
            self.maximum_base_speed_mps,
            self.local_operation_duration_s,
            self.bending_tolerance_inv_m,
        )
        if any(value <= 0.0 for value in values):
            raise ValueError("任务规划参数必须全部为正。")
        offset = np.asarray(self.straight_executor_tip_offset_m, dtype=float)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError("直线构型末端偏置必须是有限三维向量。")
        object.__setattr__(self, "straight_executor_tip_offset_m", offset.copy())


@dataclass(frozen=True)
class OperationSetpoint:
    signals: OperationSignals
    executor_target_world: np.ndarray
    task_point_world: np.ndarray
    surface_normal_world: np.ndarray
    surface_tangent_world: np.ndarray
    base_twist_world: np.ndarray


class OperationDirector:
    """任务层只产生意图，不求解机械臂或肌腱命令。"""

    def __init__(self, workspace: ConfinedWorkspace, plan: OperationPlan) -> None:
        self.workspace = workspace
        self.plan = plan
        self._initial_base_position: np.ndarray | None = None

    def setpoint(
        self,
        phase: ConfinedOperationPhase,
        phase_step: int,
        state: RobotState,
        dt: float,
    ) -> OperationSetpoint:
        if self._initial_base_position is None:
            self._initial_base_position = state.base_pose.position.copy()
        entry = self.workspace.entry.center_world
        surface = self.workspace.contact_surface
        task = (
            surface.point_world
            if surface is not None
            else self.workspace.task_region.center_world
        )
        tip_offset = self.plan.straight_executor_tip_offset_m
        staging_base = entry - tip_offset
        insertion_base = task - tip_offset
        target = task if phase in {
            ConfinedOperationPhase.COOPERATIVE_INSERTION,
            ConfinedOperationPhase.BASE_LOCK,
            ConfinedOperationPhase.LOCAL_OPERATION,
        } else entry
        if phase == ConfinedOperationPhase.RETREAT:
            # 退回时机械臂先保持相对基座的直线构型，随基座整体退出。
            target = state.base_pose.position + tip_offset
        insertion_phases = {
            ConfinedOperationPhase.COOPERATIVE_INSERTION,
            ConfinedOperationPhase.BASE_LOCK,
            ConfinedOperationPhase.LOCAL_OPERATION,
        }
        if phase == ConfinedOperationPhase.RETREAT:
            base_target = self._initial_base_position
        elif phase in insertion_phases:
            base_target = insertion_base
        else:
            base_target = staging_base
        base_twist = np.zeros(6)
        if phase in {
            ConfinedOperationPhase.BASE_APPROACH,
            ConfinedOperationPhase.COOPERATIVE_INSERTION,
            ConfinedOperationPhase.RETREAT,
        }:
            base_twist[:3] = self._limited_base_velocity(
                base_target - state.base_pose.position
            )
        tolerance = self.plan.position_tolerance_m
        signals = OperationSignals(
            initialized=phase == ConfinedOperationPhase.INITIALIZE,
            base_at_entry=np.linalg.norm(state.base_pose.position - staging_base) <= tolerance,
            insertion_complete=np.linalg.norm(state.executor.tip_pose.position - task) <= tolerance,
            base_locked=phase == ConfinedOperationPhase.BASE_LOCK and phase_step >= 1,
            local_operation_complete=(
                phase == ConfinedOperationPhase.LOCAL_OPERATION
                and phase_step * dt >= self.plan.local_operation_duration_s
            ),
            retreat_complete=(
                phase == ConfinedOperationPhase.RETREAT
                and np.linalg.norm(state.executor.bending)
                <= self.plan.bending_tolerance_inv_m
                and np.linalg.norm(
                    state.base_pose.position - self._initial_base_position
                )
                <= tolerance
            ),
        )
        normal = (
            surface.normal_world
            if surface is not None
            else self.workspace.entry.inward_direction_world
        )
        tangent = _orthogonal_tangent(normal)
        return OperationSetpoint(signals, target, task, normal, tangent, base_twist)

    def _limited_base_velocity(self, error: np.ndarray) -> np.ndarray:
        velocity = self.plan.base_gain * error
        speed = float(np.linalg.norm(velocity))
        if speed > self.plan.maximum_base_speed_mps:
            velocity *= self.plan.maximum_base_speed_mps / speed
        return velocity


def _orthogonal_tangent(normal: np.ndarray) -> np.ndarray:
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(reference, normal))) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    tangent = np.cross(normal, reference)
    return tangent / np.linalg.norm(tangent)
