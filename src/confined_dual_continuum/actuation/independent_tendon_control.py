"""仅用于显式消融的逐肌腱限幅基线，禁止作为正式控制默认值。"""
import numpy as np
from confined_dual_continuum.actuation.compatible_tendon_control import CompatibleTendonController, CompatibleTendonStep


class IndependentTendonController(CompatibleTendonController):
    def step(self, requested_bending_rate, actual_m, dt, *, allow_recovery=True):
        if not self.limits.enabled:
            return super().step(requested_bending_rate, actual_m, dt, allow_recovery=allow_recovery)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt 必须为有限正数。")
        actual = self.model._vector(actual_m, self.model.tendon_count, "actual_m")
        if not self._initialized:
            self.reset(actual)
        raw = self.model.to_tendon(requested_bending_rate)
        lower = np.maximum(self.limits.minimum_m, actual-self.limits.maximum_lead_m)
        upper = np.minimum(self.limits.maximum_m, actual+self.limits.maximum_lead_m)
        feasible = bool(np.all(lower <= upper) and np.all(self._target >= lower-1e-12) and np.all(self._target <= upper+1e-12))
        rate = np.clip(raw, -self.limits.maximum_rate_mps, self.limits.maximum_rate_mps)
        rate = np.clip(rate, (lower-self._target)/dt, (upper-self._target)/dt) if feasible else np.zeros_like(raw)
        self._target += dt*rate
        ratios = np.divide(np.abs(rate), np.abs(raw), out=np.ones_like(raw), where=np.abs(raw)>1e-15)
        return CompatibleTendonStep(self.model.estimate(rate), rate, self._target.copy(),
            float(np.linalg.norm(self.model.residual(rate))), float(np.min(ratios)),
            not np.allclose(rate, raw, rtol=0, atol=1e-12), feasible,
            float(np.linalg.norm(self.model.residual(actual))), float(np.max(np.abs(self._target-actual))),
            constraint_status="independent_clipping" if feasible else "baseline_infeasible")
