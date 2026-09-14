"""在弯曲空间内统一处理肌腱速率、位移和目标超前约束。"""

from dataclasses import dataclass

import numpy as np

from confined_dual_continuum.actuation.bending_space import BendingSpaceModel


@dataclass(frozen=True)
class TendonLimits:
    minimum_m: np.ndarray
    maximum_m: np.ndarray
    maximum_rate_mps: np.ndarray
    maximum_lead_m: np.ndarray
    enabled: bool = True
    rate_enabled: bool = False
    maximum_curvature_per_m: float | None = None

    def __post_init__(self) -> None:
        if self.maximum_curvature_per_m is not None and (
            not np.isfinite(self.maximum_curvature_per_m) or self.maximum_curvature_per_m <= 0
        ):
            raise ValueError("每段曲率模长上限必须为正有限数。")
        arrays = [np.asarray(item, dtype=float) for item in (
            self.minimum_m, self.maximum_m, self.maximum_rate_mps, self.maximum_lead_m
        )]
        if any(item.ndim != 1 or not np.all(np.isfinite(item)) for item in arrays):
            raise ValueError("肌腱约束必须是一维有限向量。")
        if len({item.shape for item in arrays}) != 1:
            raise ValueError("肌腱约束维度必须一致。")
        if np.any(arrays[0] >= arrays[1]) or np.any(arrays[2] <= 0.0) or np.any(arrays[3] <= 0.0):
            raise ValueError("肌腱上下界、速率和 lead 约束无效。")
        for name, value in zip(
            ("minimum_m", "maximum_m", "maximum_rate_mps", "maximum_lead_m"), arrays
        ):
            object.__setattr__(self, name, value.copy())


@dataclass(frozen=True)
class CompatibleTendonStep:
    bending_rate: np.ndarray
    tendon_rate_mps: np.ndarray
    target_m: np.ndarray
    compatibility_residual_norm: float
    common_scale: float
    saturated: bool
    feasible: bool
    actual_residual_norm: float
    maximum_lead_m: float
    constraint_status: str = "accepted"
    limiting_tendon_index: int = -1
    lead_excess_m: float = 0.0
    displacement_excess_m: float = 0.0


class CompatibleTendonController:
    """正常请求统一缩放；lead 越界仅通过弯曲空间内的合法增量恢复。"""

    def __init__(self, model: BendingSpaceModel, limits: TendonLimits) -> None:
        if model.tendon_count != limits.maximum_rate_mps.size:
            raise ValueError("耦合矩阵与肌腱约束数量不一致。")
        self.model = model
        self.limits = limits
        self._target = np.zeros(model.tendon_count)
        self._initialized = False

    def reset(self, actual_m: np.ndarray) -> None:
        # 实际反馈先投影回弯曲空间，避免把不可实现分量带入下一周期。
        self._target = self.model.project(actual_m)
        self._initialized = True

    def cancel_step(self, result: CompatibleTendonStep, dt: float) -> None:
        """另一臂或安全监督拒绝本周期命令时，同时撤销尚未发送的积分。"""
        self._target = result.target_m - dt * result.tendon_rate_mps

    def step(
        self,
        requested_bending_rate: np.ndarray,
        actual_m: np.ndarray,
        dt: float,
        *,
        allow_recovery: bool = True,
    ) -> CompatibleTendonStep:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt 必须为正。")
        if not self._initialized:
            self.reset(actual_m)
        raw_rate = self.model.to_tendon(requested_bending_rate)
        actual = self.model._vector(actual_m, self.model.tendon_count, "actual_m")
        # 圆盘投影在弯曲空间完成，正式全约束路径也使用同一边界。
        curvature_limited = False
        if self.limits.maximum_curvature_per_m is not None:
            current = self.model.estimate(self._target)
            endpoint = (current + dt * np.asarray(requested_bending_rate)).reshape(-1, 2)
            norms = np.linalg.norm(endpoint, axis=1)
            curvature_limited = bool(np.any(norms > self.limits.maximum_curvature_per_m + 1e-12))
            endpoint *= np.minimum(1., self.limits.maximum_curvature_per_m / np.maximum(norms, 1e-15))[:, None]
            requested_bending_rate = (endpoint.ravel() - current) / dt
            raw_rate = self.model.to_tendon(requested_bending_rate)
        if not self.limits.enabled:
            # 曲率圆盘与速率独立启用，不恢复旧行程或 lead 约束。
            rate = np.asarray(requested_bending_rate, dtype=float).copy()
            constrained = self.model.to_tendon(rate)
            scale = 1.
            moving = np.abs(constrained) > 1e-15
            if self.limits.rate_enabled and np.any(moving):
                scale = min(1., float(np.min(self.limits.maximum_rate_mps[moving] / np.abs(constrained[moving]))))
            rate *= scale
            constrained *= scale
            saturated = curvature_limited or not np.allclose(constrained, raw_rate, rtol=1e-10, atol=1e-15)
            raw_rate = constrained
            self._target = self._target + dt * raw_rate
            return CompatibleTendonStep(
                bending_rate=rate,
                tendon_rate_mps=raw_rate,
                target_m=self._target.copy(),
                compatibility_residual_norm=float(np.linalg.norm(self.model.residual(raw_rate))),
                common_scale=scale, saturated=saturated, feasible=True,
                actual_residual_norm=float(np.linalg.norm(self.model.residual(actual))),
                maximum_lead_m=float(np.max(np.abs(self._target - actual))),
                constraint_status=("curvature_rate_limited" if saturated else "accepted")
                    if self.limits.rate_enabled or self.limits.maximum_curvature_per_m is not None else "limits_disabled",
            )
        lower_target = np.maximum(self.limits.minimum_m, actual - self.limits.maximum_lead_m)
        upper_target = np.minimum(self.limits.maximum_m, actual + self.limits.maximum_lead_m)
        lead_excess = np.maximum(np.abs(self._target - actual) - self.limits.maximum_lead_m, 0.0)
        displacement_excess = np.maximum(np.maximum(self.limits.minimum_m-self._target,
                                                    self._target-self.limits.maximum_m), 0.0)
        # 超前区间可能因外力/反馈漂移而失去可行性；明确报告，禁止假装限幅成功。
        feasible = bool(np.all(lower_target <= upper_target) and
                        np.all(self._target >= lower_target - 1e-12) and
                        np.all(self._target <= upper_target + 1e-12))
        scale = self._largest_common_scale(raw_rate, lower_target, upper_target, dt) if feasible else 0.0
        status = "accepted" if scale >= 1.0-1e-12 else "saturated"
        limiting_index = -1
        if not feasible:
            if np.any(lower_target > upper_target):
                status = "empty_tendon_interval"
                limiting_index = int(np.argmax(lower_target-upper_target))
            elif np.max(displacement_excess) > 1e-12:
                status = "previous_target_outside_displacement"
                limiting_index = int(np.argmax(displacement_excess))
            else:
                status = "previous_target_outside_lead"
                limiting_index = int(np.argmax(lead_excess))
        tendon_rate = scale * raw_rate
        bending_rate = scale * np.asarray(requested_bending_rate, dtype=float)
        if status == "previous_target_outside_lead" and allow_recovery:
            delta, recovery_status = self._recover_increment(lower_target, upper_target, dt)
            if delta is not None:
                bending_rate = delta / dt
                tendon_rate = self.model.to_tendon(bending_rate)
                feasible = True
                status = "recovered_lead"
                # 原请求被恢复命令替换；scale=0 不代表本周期没有恢复运动。
            else:
                status = recovery_status
        if self.limits.maximum_curvature_per_m is not None:
            endpoint = self.model.estimate(self._target + dt * tendon_rate).reshape(-1, 2)
            if np.any(np.linalg.norm(endpoint, axis=1) > self.limits.maximum_curvature_per_m + 1e-10):
                # 恢复 lead 不能越过曲率边界；无合法命令则显式停止。
                bending_rate = np.zeros_like(bending_rate)
                tendon_rate = np.zeros_like(tendon_rate)
                feasible, scale, status = False, 0., "curvature_infeasible"
        self._target = self._target + dt * tendon_rate
        residual = float(np.linalg.norm(self.model.residual(tendon_rate)))
        return CompatibleTendonStep(
            bending_rate=bending_rate,
            tendon_rate_mps=tendon_rate,
            target_m=self._target.copy(),
            compatibility_residual_norm=residual,
            common_scale=scale,
            saturated=curvature_limited or scale < 1.0 - 1.0e-12,
            feasible=feasible,
            actual_residual_norm=float(np.linalg.norm(self.model.residual(actual))),
            maximum_lead_m=float(np.max(np.abs(self._target - actual))),
            constraint_status=status,
            limiting_tendon_index=limiting_index,
            lead_excess_m=float(np.max(lead_excess)),
            displacement_excess_m=float(np.max(displacement_excess)),
        )

    def _recover_increment(self, lower, upper, dt):
        """用循环半空间投影寻找可行弯曲增量，不扩大任何物理限制。

        约束为 lower <= previous_target + Cb @ delta <= upper，
        并与单周期速度区间相交。有限迭代找不到解只报告搜索失败，
        不把未收敛误称为已经证明整个相容集合无解。
        """
        lower = np.maximum(lower, self._target-dt*self.limits.maximum_rate_mps)
        upper = np.minimum(upper, self._target+dt*self.limits.maximum_rate_mps)
        if np.any(lower > upper):
            return None, "recovery_rate_interval_empty"
        coupling = self.model.coupling_matrix
        rows = np.vstack((coupling, -coupling))
        bounds = np.concatenate((upper-self._target, self._target-lower))
        norms = np.sum(rows*rows, axis=1)
        delta = np.zeros(self.model.bending_size)
        for _ in range(512):
            for row, bound, norm in zip(rows, bounds, norms):
                violation = float(row @ delta-bound)
                if violation > 0:
                    if norm == 0:
                        return None, "recovery_fixed_tendon_conflict"
                    delta -= violation/norm*row
            target = self._target+coupling@delta
            if np.all(np.isfinite(target)) and np.all(target >= lower-1e-12) and np.all(target <= upper+1e-12):
                return delta, "recovered_lead"
        return None, "recovery_search_exhausted"

    def _largest_common_scale(
        self,
        rate: np.ndarray,
        lower_target: np.ndarray,
        upper_target: np.ndarray,
        dt: float,
    ) -> float:
        scale = 1.0
        nonzero = np.abs(rate) > 1.0e-15
        if np.any(nonzero):
            scale = min(scale, float(np.min(self.limits.maximum_rate_mps[nonzero] / np.abs(rate[nonzero]))))
        positive = rate > 1.0e-15
        negative = rate < -1.0e-15
        if np.any(positive):
            scale = min(scale, float(np.min((upper_target[positive] - self._target[positive]) / (dt * rate[positive]))))
        if np.any(negative):
            scale = min(scale, float(np.min((lower_target[negative] - self._target[negative]) / (dt * rate[negative]))))
        return float(np.clip(scale, 0.0, 1.0))
