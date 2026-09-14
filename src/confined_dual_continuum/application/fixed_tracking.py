"""固定基座单向协同；从臂结果不参与主臂求解或运行权限。"""
from dataclasses import dataclass, replace
import numpy as np
from confined_dual_continuum.actuation.compatible_tendon_control import CompatibleTendonController
from confined_dual_continuum.coordination.camera_follower import CameraFollower, camera_differential
from confined_dual_continuum.coordination.observation_candidates import select_observation_candidate
from confined_dual_continuum.domain.geometry import quaternion_rotation
from confined_dual_continuum.domain.state import ArmCommand, RobotCommand
from confined_dual_continuum.kinematics.pcc import position_jacobian
from confined_dual_continuum.kinematics.constrained_velocity import constrained_velocity


@dataclass(frozen=True)
class TrackingConfig:
    gain: float = 2.
    maximum_curvature_per_m: float = 13.5
    maximum_tendon_speed_mps: float = .001
    clearance_m: float = .003
    critical_clearance_m: float = .0005
    joint_margin_rad: float = .00872664626
    observation_distance_m: float = .04
    offset_switch_s: float = 2.
    observer_offsets_base_m: tuple = ((0., -.025, -.025), (.025, -.02, -.025), (-.025, -.02, -.025))

    def __post_init__(self):
        numbers = (self.gain, self.maximum_curvature_per_m, self.maximum_tendon_speed_mps,
                   self.clearance_m, self.critical_clearance_m, self.joint_margin_rad,
                   self.observation_distance_m, self.offset_switch_s)
        offsets = np.asarray(self.observer_offsets_base_m, float)
        if (not all(np.isfinite(x) and x > 0 for x in numbers)
                or self.critical_clearance_m >= self.clearance_m
                or offsets.ndim != 2 or offsets.shape[1] != 3 or not len(offsets)
                or not np.all(np.isfinite(offsets))):
            raise ValueError('跟随控制参数或观察偏置无效。')


@dataclass(frozen=True)
class TrackingOutput:
    command: RobotCommand
    diagnostics: dict


class FixedTrackingController:
    def __init__(self, robot, camera, config=TrackingConfig(), observation_candidates=()):
        self.robot, self.camera, self.config = robot, camera, config
        self.observation_candidates = tuple(observation_candidates)
        # 本任务显式采用曲率圆盘和绳速限制，不改变共享机器人配置。
        limits = replace(robot.limits, enabled=False, rate_enabled=True,
                         maximum_curvature_per_m=config.maximum_curvature_per_m,
                         maximum_rate_mps=np.full_like(robot.limits.maximum_rate_mps,
                                                      config.maximum_tendon_speed_mps))
        self.actuation = {a: CompatibleTendonController(robot.bending_model(a), limits)
                          for a in ('executor', 'observer')}
        self.follower = CameraFollower(camera, gain=config.gain,
                                       distance_m=config.observation_distance_m)

    def reset(self, state):
        self.targets = {}
        for name, actuator in self.actuation.items():
            actual = getattr(state, name).tendon_displacement_m
            actuator.reset(actual)
            self.targets[name] = actuator.model.project(actual)
        self.executor_fault = None
        self.offset_index, self.last_offset_time = 0, 0.
        self.candidate = None
        self.attempted_samples = set()
        self.observer_status = 'initializing'
        self.base_rotation = quaternion_rotation(state.base_pose.quaternion_wxyz)

    def _jacobian(self, state, geometry, arm, arc=None):
        root = getattr(geometry, arm+'_root_world')
        return root[:3, :3]@position_jacobian(getattr(state, arm).bending,
            self.robot.flexure_lengths_m, self.robot.distal_offsets_m, arc_m=arc)

    def _constraints(self, arm, state, geometry, dt, executor_rate=None):
        model = self.actuation[arm].model
        q = model.estimate(self.targets[arm])
        coupling = model.coupling_matrix
        rows = [*coupling, *(-coupling)]
        lower = [-self.config.maximum_tendon_speed_mps]*(2*len(coupling))
        # 内接正 32 边形使下一步曲率严格处于圆盘内，避免执行后改方向。
        bound = self.config.maximum_curvature_per_m*np.cos(np.pi/32)
        for section in range(len(q)//2):
            for angle in np.arange(32)*2*np.pi/32:
                normal = np.array([np.cos(angle), np.sin(angle)])
                row = np.zeros(len(q))
                row[2*section:2*section+2] = -dt*normal
                rows.append(row)
                lower.append(normal@q[2*section:2*section+2]-bound)
        for gap in geometry.clearances:
            if gap.arm != arm:
                continue
            # 主臂安全仅依赖自身及静态环境，明确不读取从臂。
            if arm == 'executor' and gap.other_arm == 'observer':
                continue
            jacobian = self._jacobian(state, geometry, arm, gap.arc_m)
            drift = 0.
            if gap.other_arm == arm:
                jacobian = jacobian-self._jacobian(state, geometry, arm, gap.other_arc_m)
            elif gap.other_arm == 'executor':
                drift = float(gap.normal_world@self._jacobian(
                    state, geometry, 'executor', gap.other_arc_m)@executor_rate)
            rows.append(gap.normal_world@jacobian)
            lower.append(-self.config.gain*(gap.distance_m-self.config.clearance_m)+drift)
        return np.asarray(rows), np.asarray(lower)

    def _apply(self, arm, requested, state, dt, matrix, lower, force_hold=False):
        actuator = self.actuation[arm]
        if force_hold:
            return ArmCommand(np.zeros(actuator.model.bending_size),
                              np.zeros(actuator.model.tendon_count), self.targets[arm].copy()), False
        result = actuator.step(requested, getattr(state, arm).tendon_displacement_m,
                               dt, allow_recovery=False)
        accepted = result.feasible and bool(np.all(matrix@result.bending_rate >= lower-1e-7))
        if not accepted:
            actuator.cancel_step(result, dt)
            return self._apply(arm, requested, state, dt, matrix, lower, True)
        self.targets[arm] = result.target_m.copy()
        return ArmCommand(result.bending_rate, result.tendon_rate_mps, result.target_m), True

    def step(self, state, geometry, target_world, velocity_world, dt, time_s):
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite(time_s):
            raise ValueError('控制时间必须有限且周期为正。')
        target, velocity = np.asarray(target_world, float), np.asarray(velocity_world, float)
        if target.shape != (3,) or velocity.shape != (3,) or not np.all(np.isfinite([target, velocity])):
            raise ValueError('主臂目标和前馈速度必须为有限三维向量。')

        # 主臂先完成求解和积分，从臂任何正常失败状态不得回写此结果。
        ex_matrix, ex_lower = self._constraints('executor', state, geometry, dt)
        ex_j = self._jacobian(state, geometry, 'executor')
        if geometry.joint_margin_rad['executor'] <= self.config.joint_margin_rad:
            self.executor_fault = 'executor_joint_guard'
        if any(g.arm == 'executor' and g.other_arm != 'observer'
               and g.distance_m <= self.config.critical_clearance_m for g in geometry.clearances):
            self.executor_fault = 'executor_clearance_guard'
        ex_rate, ex_ok = constrained_velocity(ex_j,
            velocity+self.config.gain*(target-state.executor.tip_pose.position), ex_matrix, ex_lower)
        if not ex_ok:
            self.executor_fault = self.executor_fault or 'executor_constraints_infeasible'
        executor, accepted = self._apply('executor', ex_rate, state, dt, ex_matrix, ex_lower,
                                         self.executor_fault is not None)
        if not accepted:
            self.executor_fault = self.executor_fault or 'executor_final_command_rejected'

        ob_matrix, ob_lower = self._constraints('observer', state, geometry, dt, executor.bending_rate)
        tip = np.eye(4)
        tip[:3, :3] = quaternion_rotation(state.observer.tip_pose.quaternion_wxyz)
        tip[:3, 3] = state.observer.tip_pose.position
        point, point_j, camera_j = camera_differential(state.observer.bending,
            self.robot.flexure_lengths_m, self.robot.distal_offsets_m,
            tip, geometry.camera_world, state.executor.tip_pose.position)
        visible = self.camera.visible(point, geometry.occluded)
        select_candidate = time_s == 0.
        if visible:
            self.last_offset_time = time_s
            self.attempted_samples.clear()
        elif time_s-self.last_offset_time >= self.config.offset_switch_s:
            self.offset_index = (self.offset_index+1) % len(self.config.observer_offsets_base_m)
            self.last_offset_time = time_s
            select_candidate = True
        offset = self.base_rotation@np.asarray(self.config.observer_offsets_base_m[self.offset_index])
        tip_camera = np.linalg.solve(tip, geometry.camera_world)
        if select_candidate:
            if self.candidate is not None:
                self.attempted_samples.add(self.candidate.sample)
            choices = tuple(c for c in self.observation_candidates if c.sample not in self.attempted_samples)
            self.candidate = select_observation_candidate(choices,
                geometry.observer_root_world, tip_camera, state.executor.tip_pose.position,
                state.observer.bending, self.camera, self.config.observation_distance_m)
        posture_rate = None
        if self.candidate is not None:
            sampled_camera = geometry.observer_root_world@self.candidate.tip_root@tip_camera
            offset = sampled_camera[:3, 3]-state.executor.tip_pose.position
            posture_rate = self.config.gain*(self.candidate.bending-state.observer.bending)
        result = self.follower.solve(point, point_j, camera_j, geometry.camera_world,
            state.executor.tip_pose.position, ex_j@executor.bending_rate, offset,
            ob_matrix, ob_lower, geometry.occluded, posture_rate)
        status = result.status
        if geometry.joint_margin_rad['observer'] <= self.config.joint_margin_rad:
            status = 'observer_joint_guard'
        elif any(g.arm == 'observer' and g.distance_m <= self.config.critical_clearance_m
                 for g in geometry.clearances):
            status = 'observer_clearance_guard'
        hold = status in ('observer_joint_guard', 'observer_clearance_guard', 'observer_safety_infeasible')
        observer, ob_ok = self._apply('observer', result.rate, state, dt, ob_matrix, ob_lower, hold)
        if not ob_ok and not hold:
            status = 'observer_final_command_rejected'
        elif ob_ok and np.any(result.view_matrix@observer.bending_rate < result.view_lower-1e-7):
            status = 'view_constraint_lost_after_actuation'
        self.observer_status = status
        return self._output(state, geometry, target, time_s, point, visible, executor, observer, status)

    def observe(self, state, geometry, target_world, time_s):
        """末时刻只记录实际状态，不求解或积分一个不会发送的额外周期。"""
        point = geometry.camera_world[:3, :3].T@(
            state.executor.tip_pose.position-geometry.camera_world[:3, 3])
        commands = {name: ArmCommand(np.zeros(a.model.bending_size),
            np.zeros(a.model.tendon_count), self.targets[name].copy()) for name, a in self.actuation.items()}
        return self._output(state, geometry, np.asarray(target_world), time_s, point,
            self.camera.visible(point, geometry.occluded), commands['executor'], commands['observer'],
            self.observer_status)

    def _output(self, state, geometry, target, time_s, point, visible, executor, observer, status):
        diagnostics = {
            'time_s': float(time_s), 'executor_status': self.executor_fault or 'tracking',
            'observer_status': status, 'visible': visible, 'occluded': bool(geometry.occluded),
            'target_camera_m': point.tolist(), 'view_margin_m': float(np.min(self.camera.margins(point))),
            'executor_target_world_m': target.tolist(),
            'executor_actual_world_m': state.executor.tip_pose.position.tolist(),
            'observer_actual_world_m': state.observer.tip_pose.position.tolist(),
            'executor_error_m': float(np.linalg.norm(target-state.executor.tip_pose.position)),
            'offset_index': self.offset_index,
            'observation_sample': self.candidate.sample if self.candidate is not None else None,
            'minimum_clearance_m': min((g.distance_m for g in geometry.clearances), default=None),
            'minimum_interarm_clearance_m': min((g.distance_m for g in geometry.clearances
                if g.arm == 'observer' and g.other_arm == 'executor'), default=None),
            'joint_margin_rad': geometry.joint_margin_rad,
            'nearest_clearance_source': {
                arm: min((g for g in geometry.clearances if g.arm == arm),
                         key=lambda g: g.distance_m).source
                if any(g.arm == arm for g in geometry.clearances) else None
                for arm in ('executor', 'observer')},
        }
        for name, command in (('executor', executor), ('observer', observer)):
            model = self.actuation[name].model
            diagnostics[name+'_tendon_target_m'] = command.tendon_target_m.tolist()
            diagnostics[name+'_tendon_rate_mps'] = command.tendon_rate_mps.tolist()
            diagnostics[name+'_compatibility_residual'] = float(np.linalg.norm(model.residual(command.tendon_rate_mps)))
        return TrackingOutput(RobotCommand(np.zeros(6), executor, observer), diagnostics)
