"""把任务阶段转换为几何目标和阶段完成信号。"""

from dataclasses import dataclass

import numpy as np

from confined_dual_continuum.domain.state import RobotState
from confined_dual_continuum.task.confined_operation import OperationSignals
from confined_dual_continuum.task.phase_machine import ConfinedOperationPhase
from confined_dual_continuum.workspace.model import ConfinedWorkspace, TaskPath


@dataclass(frozen=True)
class OperationPlan:
    position_tolerance_m: float
    base_gain: float
    maximum_base_speed_mps: float
    waypoint_duration_s: float
    wiping_force_n: float
    observer_offset_m: float
    straight_executor_tip_offset_m: np.ndarray

    def __post_init__(self) -> None:
        values = (
            self.position_tolerance_m,
            self.base_gain,
            self.maximum_base_speed_mps,
            self.waypoint_duration_s,
            self.wiping_force_n,
            self.observer_offset_m,
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
    observer_target_world: np.ndarray
    surface_normal_world: np.ndarray
    base_twist_world: np.ndarray
    executor_force_target_n: float
    avoidance_arm: str


class OperationDirector:
    """任务层只产生意图，不求解机械臂或肌腱命令。"""

    def __init__(self, workspace: ConfinedWorkspace, plan: OperationPlan) -> None:
        self.workspace = workspace
        self.plan = plan

    def setpoint(
        self,
        phase: ConfinedOperationPhase,
        phase_step: int,
        state: RobotState,
        dt: float,
    ) -> OperationSetpoint:
        tip_offset = self.plan.straight_executor_tip_offset_m
        base_target = self._base_target(phase)
        base_twist = np.zeros(6)
        if phase in {
            ConfinedOperationPhase.BASE_APPROACH,
            ConfinedOperationPhase.CURVED_SURFACE_TRANSIT,
            ConfinedOperationPhase.WIPING_TRANSIT,
            ConfinedOperationPhase.OBSTACLE_TRANSIT,
        }:
            base_twist[:3] = self._limited_base_velocity(
                base_target - state.base_pose.position
            )
        path = self._active_path(phase)
        path_complete = False
        if path is None:
            executor_target = state.base_pose.position + tip_offset
            observer_target = state.base_pose.position + np.array([0.0, -0.01, 0.14])
            normal = np.array([0.0, 0.0, 1.0])
        else:
            waypoint = min(
                int(phase_step * dt / self.plan.waypoint_duration_s),
                path.points_world.shape[0] - 1,
            )
            path_complete = (
                phase_step * dt
                >= path.points_world.shape[0] * self.plan.waypoint_duration_s
            )
            normal = path.normal_world
            if phase in {
                ConfinedOperationPhase.CONFINED_VISUAL_INSPECTION,
                ConfinedOperationPhase.CURVED_SURFACE_INSPECTION,
            }:
                observer_target = path.points_world[waypoint]
                executor_target = state.base_pose.position + np.array([0.025, 0.01, 0.12])
            else:
                executor_target = path.points_world[waypoint]
                observer_target = (
                    executor_target
                    + self.plan.observer_offset_m * normal
                    + np.array([0.0, -0.02, 0.0])
                )
        transit_phases = {
            ConfinedOperationPhase.BASE_APPROACH,
            ConfinedOperationPhase.CURVED_SURFACE_TRANSIT,
            ConfinedOperationPhase.WIPING_TRANSIT,
            ConfinedOperationPhase.OBSTACLE_TRANSIT,
        }
        if phase == ConfinedOperationPhase.INITIALIZE:
            phase_complete = True
        elif phase == ConfinedOperationPhase.BASE_LOCK:
            phase_complete = phase_step >= 1
        elif phase in transit_phases:
            phase_complete = (
                np.linalg.norm(state.base_pose.position - base_target)
                <= self.plan.position_tolerance_m
            )
        else:
            phase_complete = path is not None and path_complete
        signals = OperationSignals(
            phase_complete=phase_complete,
            completion_reason=f"{phase.value}_complete" if phase_complete else "",
        )
        force_target = (
            self.plan.wiping_force_n
            if phase == ConfinedOperationPhase.FORCE_POSITION_WIPING
            else 0.0
        )
        avoidance_arm = (
            "executor"
            if phase in {
                ConfinedOperationPhase.CONFINED_VISUAL_INSPECTION,
                ConfinedOperationPhase.CURVED_SURFACE_INSPECTION,
            }
            else "observer"
        )
        return OperationSetpoint(
            signals,
            executor_target,
            observer_target,
            normal,
            base_twist,
            force_target,
            avoidance_arm,
        )

    def _base_target(self, phase: ConfinedOperationPhase) -> np.ndarray:
        if phase == ConfinedOperationPhase.CURVED_SURFACE_TRANSIT:
            return self.workspace.curved_base_world
        if phase == ConfinedOperationPhase.WIPING_TRANSIT:
            return self.workspace.wiping_base_world
        if phase == ConfinedOperationPhase.OBSTACLE_TRANSIT:
            return self.workspace.obstacle_base_world
        return self.workspace.entry_base_world

    def _active_path(self, phase: ConfinedOperationPhase) -> TaskPath | None:
        return {
            ConfinedOperationPhase.CONFINED_VISUAL_INSPECTION: self.workspace.confined_inspection,
            ConfinedOperationPhase.CURVED_SURFACE_INSPECTION: self.workspace.curved_inspection,
            ConfinedOperationPhase.FORCE_POSITION_WIPING: self.workspace.wiping,
            ConfinedOperationPhase.ASYMMETRIC_OBSTACLE_OPERATION: self.workspace.obstacle_operation,
        }.get(phase)

    def _limited_base_velocity(self, error: np.ndarray) -> np.ndarray:
        velocity = self.plan.base_gain * error
        speed = float(np.linalg.norm(velocity))
        if speed > self.plan.maximum_base_speed_mps:
            velocity *= self.plan.maximum_base_speed_mps / speed
        return velocity
