"""主臂专用控制；不依赖 MuJoCo、文件格式、窗口或从臂状态。"""
from dataclasses import dataclass
import numpy as np
from confined_dual_continuum.actuation.compatible_tendon_control import CompatibleTendonController
from confined_dual_continuum.domain.state import ArmCommand
from confined_dual_continuum.kinematics.fast_pcc import tip_position_jacobian
from confined_dual_continuum.kinematics.solver import solve_velocity


@dataclass(frozen=True)
class ExecutorResult:
    command: ArmCommand
    status: str
    step: object | None
    diagnostics: dict | None = None


class ExecutorTrackingController:
    def __init__(self, robot, root_rotation_world, *, gain=2., damping=1e-4,
                 joint_margin_rad=np.deg2rad(.5), joint_guard_enabled=True):
        self.robot, self.rotation = robot, np.asarray(root_rotation_world, float).copy()
        if (self.rotation.shape != (3, 3) or not np.all(np.isfinite(self.rotation))
                or not all(np.isfinite(x) and x > 0 for x in (gain, damping, joint_margin_rad))):
            raise ValueError('控制增益、阻尼、关节保护或臂根旋转无效。')
        self.gain, self.damping, self.joint_margin = gain, damping, joint_margin_rad
        self.joint_guard_enabled = bool(joint_guard_enabled)
        # 直接传递同一公共限制对象，不复制或覆盖肌腱速度、曲率与开关。
        self.actuation = CompatibleTendonController(robot.bending_model('executor'), robot.limits)
        self.status = 'not_initialized'

    def reset(self, feedback):
        self.actuation.reset(feedback.tendon_displacement_m)
        self.target = self.actuation.model.project(feedback.tendon_displacement_m)
        self.status = 'tracking'

    def step(self, feedback, reference_world_m, reference_velocity_mps, dt):
        target, velocity = np.asarray(reference_world_m, float), np.asarray(reference_velocity_mps, float)
        actual = np.asarray(feedback.tip_world_m, float)
        if (any(x.shape != (3,) for x in (target, velocity, actual))
                or not all(np.all(np.isfinite(x)) for x in (target, velocity, actual))
                or not np.isfinite(dt) or dt <= 0
                or not np.isfinite(feedback.minimum_joint_margin_rad)):
            raise ValueError('控制输入必须是有限三维向量和正周期。')
        if self.status == 'not_initialized':
            raise RuntimeError('主臂控制器必须先 reset。')
        if self.joint_guard_enabled and feedback.minimum_joint_margin_rad <= self.joint_margin:
            self.status = 'joint_guard'
        if self.status != 'tracking':
            return self._hold()
        bending = self.actuation.model.estimate(feedback.tendon_displacement_m)
        _, jacobian = tip_position_jacobian(bending, self.robot.flexure_lengths_m, self.robot.distal_offsets_m)
        world_jacobian = self.rotation@jacobian
        rate = solve_velocity(world_jacobian, velocity+self.gain*(target-actual), damping=self.damping)
        raw_tendon_rate = self.actuation.model.to_tendon(rate)
        result = self.actuation.step(rate, feedback.tendon_displacement_m, dt)
        current_curvature = bending.reshape(-1, 2)
        requested_curvature = (bending + dt*rate).reshape(-1, 2)
        applied_curvature = self.actuation.model.estimate(result.target_m).reshape(-1, 2)
        curvature_limit = self.actuation.limits.maximum_curvature_per_m
        diagnostics = {
            'raw_bending_rate': rate.copy(),
            'raw_tendon_rate_mps': raw_tendon_rate.copy(),
            'curvature_norm': np.linalg.norm(current_curvature, axis=1),
            'requested_curvature_norm': np.linalg.norm(requested_curvature, axis=1),
            'applied_curvature_norm': np.linalg.norm(applied_curvature, axis=1),
            'curvature_margin': (curvature_limit - np.linalg.norm(applied_curvature, axis=1)
                if curvature_limit is not None else np.full(3, np.nan)),
            'jacobian_singular_values': np.linalg.svd(world_jacobian, compute_uv=False),
        }
        sigma = diagnostics['jacobian_singular_values']
        diagnostics['jacobian_sigma_min'] = float(np.min(sigma))
        diagnostics['jacobian_condition_number'] = float(
            np.inf if diagnostics['jacobian_sigma_min'] <= 1.0e-15
            else np.max(sigma)/diagnostics['jacobian_sigma_min'])
        if not result.feasible:
            self.actuation.cancel_step(result, dt)
            self.status = 'tendon_constraints_infeasible'
            return self._hold(diagnostics)
        self.target = result.target_m.copy()
        return ExecutorResult(ArmCommand(result.bending_rate, result.tendon_rate_mps, result.target_m),
                              'tracking', result, diagnostics)

    def _hold(self, diagnostics=None):
        model = self.actuation.model
        return ExecutorResult(ArmCommand(np.zeros(model.bending_size), np.zeros(model.tendon_count),
                                        self.target.copy()), self.status, None, diagnostics)
