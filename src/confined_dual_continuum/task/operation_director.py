"""任务目标与闭环完成判据；时间只用于稳定驻留和失败超时。"""
from dataclasses import dataclass
import numpy as np
from confined_dual_continuum.domain.state import RobotState
from confined_dual_continuum.domain.geometry import quaternion_rotation
from confined_dual_continuum.task.confined_operation import OperationSignals
from confined_dual_continuum.task.phase_machine import ConfinedOperationPhase as Phase
from confined_dual_continuum.workspace.model import ConfinedWorkspace


@dataclass(frozen=True)
class OperationPlan:
    position_tolerance_m: float
    base_gain: float
    maximum_base_speed_mps: float
    waypoint_duration_s: float
    wiping_force_n: float
    observer_offset_m: float
    straight_executor_tip_offset_m: np.ndarray
    waypoint_timeout_s: float = 20.0
    force_tolerance_n: float = 0.2
    transit_height_m: float = -0.10
    view_tolerance_rad: float = 0.35

    def __post_init__(self):
        positive = (self.position_tolerance_m, self.base_gain, self.maximum_base_speed_mps,
                    self.waypoint_duration_s, self.wiping_force_n, self.observer_offset_m,
                    self.waypoint_timeout_s, self.force_tolerance_n, self.view_tolerance_rad)
        if not all(np.isfinite(v) and v > 0 for v in positive):
            raise ValueError("任务参数必须为有限正数。")
        if self.waypoint_timeout_s <= self.waypoint_duration_s:
            raise ValueError("超时必须大于到达后的稳定驻留时间。")
        offset = np.asarray(self.straight_executor_tip_offset_m, dtype=float)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)) or not np.isfinite(self.transit_height_m):
            raise ValueError("任务偏置和转场高度必须有限。")
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
    observer_view_direction_world: np.ndarray
    waypoint_index: int


class OperationDirector:
    def __init__(self, workspace: ConfinedWorkspace, plan: OperationPlan):
        self.workspace, self.plan = workspace, plan
        self.reset()

    def reset(self):
        self._phase = None
        self._waypoint = 0
        self._dwell = 0.0
        self._elapsed = 0.0
        self._transit_leg = 0
        self._transit_origin = np.zeros(3)

    def setpoint(self, phase: Phase, phase_step: int, state: RobotState, dt: float):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("任务周期必须为有限正数。")
        if phase != self._phase:
            self._phase, self._waypoint = phase, 0
            self._dwell, self._elapsed = 0.0, 0.0
            self._transit_leg = 0
            self._transit_origin = state.base_pose.position.copy()
        self._elapsed += dt
        position = state.base_pose.position
        rotation = quaternion_rotation(state.base_pose.quaternion_wxyz)
        executor_target = position + rotation @ np.array([0.025, 0.01, 0.11])
        observer_target = position + rotation @ np.array([0.0, -0.025, 0.13])
        normal = np.array([0., 0., -1.])
        view = np.array([0., 0., 1.])
        twist = np.zeros(6)
        force_target = 0.0
        avoidance = "observer"
        complete = False
        violation = None
        paths = {
            Phase.CONFINED_VISUAL_INSPECTION: self.workspace.confined_inspection,
            Phase.CURVED_SURFACE_INSPECTION: self.workspace.curved_inspection,
            Phase.FORCE_POSITION_WIPING: self.workspace.wiping,
            Phase.ASYMMETRIC_OBSTACLE_OPERATION: self.workspace.obstacle_operation,
        }
        stations = {
            Phase.BASE_APPROACH: self.workspace.entry_base_world,
            Phase.CURVED_SURFACE_TRANSIT: self.workspace.curved_base_world,
            Phase.WIPING_TRANSIT: self.workspace.wiping_base_world,
            Phase.OBSTACLE_TRANSIT: self.workspace.obstacle_base_world,
        }
        if phase in stations:
            destination = stations[phase]
            # 先沿原工位竖直脱离，再在空旷高度横移，最后进入下一工位。
            route = (np.array([*self._transit_origin[:2], self.plan.transit_height_m]),
                     np.array([*destination[:2], self.plan.transit_height_m]), destination)
            target = route[self._transit_leg]
            if np.linalg.norm(position-target) <= self.plan.position_tolerance_m:
                if self._transit_leg < 2:
                    self._transit_leg += 1
                    self._elapsed = 0.0
                    target = route[self._transit_leg]
                else:
                    complete = True
            velocity = self.plan.base_gain * (target-position)
            speed = np.linalg.norm(velocity)
            twist[:3] = velocity * min(1.0, self.plan.maximum_base_speed_mps / max(speed, 1e-12))
            # 正式模式禁用转场弯曲；持续激活消融仍会追踪预设收拢目标。
            executor_target = position + rotation @ self.plan.straight_executor_tip_offset_m
            observer_target = position + rotation @ np.array([0., -.01, .14])
        elif phase in paths:
            path = paths[phase]
            if path is None:
                violation = "missing_task_path"
            else:
                normal = path.normal_world
                point = path.points_world[self._waypoint]
                if phase in (Phase.CONFINED_VISUAL_INSPECTION, Phase.CURVED_SURFACE_INSPECTION):
                    observer_target = point
                    avoidance = "executor"
                    view = -normal
                    error = np.linalg.norm(state.observer.tip_pose.position-point)
                    axis = quaternion_rotation(state.observer.tip_pose.quaternion_wxyz)[:, 2]
                    arrived = error <= self.plan.position_tolerance_m and float(axis @ view) >= np.cos(self.plan.view_tolerance_rad)
                else:
                    executor_target = point
                    # 观察方向与目标位置独立表达，确保真正朝向操作区。
                    observer_target = point + self.plan.observer_offset_m * normal + np.array([0., -0.025, 0.])
                    view = point - state.observer.tip_pose.position
                    error = state.executor.tip_pose.position-point
                    if phase == Phase.FORCE_POSITION_WIPING:
                        force_target = self.plan.wiping_force_n
                        error = error-normal*np.dot(error, normal)
                        force = float(state.executor.contact_force_world @ normal)
                        arrived = np.linalg.norm(error) <= self.plan.position_tolerance_m and abs(force-force_target) <= self.plan.force_tolerance_n
                    else:
                        arrived = np.linalg.norm(error) <= self.plan.position_tolerance_m
                self._dwell = self._dwell + dt if arrived else 0.0
                if self._dwell >= self.plan.waypoint_duration_s:
                    if self._waypoint == len(path.points_world)-1:
                        complete = True
                    else:
                        self._waypoint += 1
                        self._dwell, self._elapsed = 0.0, 0.0
        elif phase == Phase.INITIALIZE:
            complete = True
        elif phase == Phase.BASE_LOCK:
            complete = phase_step >= 1
        if self._elapsed > self.plan.waypoint_timeout_s and phase not in (Phase.COMPLETE, Phase.SAFETY_STOP):
            violation = "waypoint_or_transit_timeout"
        view_norm = np.linalg.norm(view)
        if view_norm <= 1e-12:
            violation = "undefined_observer_view"
            view = np.array([0., 0., 1.])
        else:
            view = view/view_norm
        return OperationSetpoint(OperationSignals(complete, f"{phase.value}_complete" if complete else "", violation),
                                 executor_target, observer_target, normal, twist, force_target,
                                 avoidance, view, self._waypoint)
