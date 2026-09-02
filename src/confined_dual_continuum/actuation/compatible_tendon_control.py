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

    def __post_init__(self) -> None:
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


class CompatibleTendonController:
    """用单一弯曲速度比例保持所有输出位于 Cb 的列空间。"""

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

    def step(
        self,
        requested_bending_rate: np.ndarray,
        actual_m: np.ndarray,
        dt: float,
    ) -> CompatibleTendonStep:
        if dt <= 0.0:
            raise ValueError("dt 必须为正。")
        if not self._initialized:
            self.reset(actual_m)
        raw_rate = self.model.to_tendon(requested_bending_rate)
        actual = np.asarray(actual_m, dtype=float)
        lower_target = np.maximum(self.limits.minimum_m, actual - self.limits.maximum_lead_m)
        upper_target = np.minimum(self.limits.maximum_m, actual + self.limits.maximum_lead_m)
        scale = self._largest_common_scale(raw_rate, lower_target, upper_target, dt)
        tendon_rate = scale * raw_rate
        bending_rate = scale * np.asarray(requested_bending_rate, dtype=float)
        self._target = self._target + dt * tendon_rate
        residual = float(np.linalg.norm(self.model.residual(tendon_rate)))
        return CompatibleTendonStep(
            bending_rate=bending_rate,
            tendon_rate_mps=tendon_rate,
            target_m=self._target.copy(),
            compatibility_residual_norm=residual,
            common_scale=scale,
            saturated=scale < 1.0 - 1.0e-12,
        )

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
