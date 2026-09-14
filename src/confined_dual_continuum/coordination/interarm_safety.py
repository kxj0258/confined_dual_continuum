"""完整物理中心线的胶囊间隙监督，固定根部仍保留临界保护。"""
from dataclasses import dataclass
import numpy as np
from confined_dual_continuum.domain.segments import closest_segments


@dataclass(frozen=True)
class InterArmSafetyConfig:
    # XML 半径 6.25 mm；20 mm 安装间距对应 7.5 mm 表面间隙。
    arm_radius_m: float = 0.00625
    influence_distance_m: float = 0.006
    release_distance_m: float = 0.007
    critical_distance_m: float = 0.0025
    avoidance_gain: float = 2.0

    def __post_init__(self):
        values = (self.arm_radius_m, self.critical_distance_m, self.influence_distance_m,
                  self.release_distance_m, self.avoidance_gain)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("双臂安全参数必须为有限正数。")
        if not self.critical_distance_m < self.influence_distance_m < self.release_distance_m:
            raise ValueError("安全间隙必须满足 critical < influence < release。")


@dataclass(frozen=True)
class InterArmSafetyDecision:
    distance_m: float
    avoidance_velocity_world: np.ndarray
    avoidance_active: bool
    safety_stop: bool
    executor_arc_m: float = 0.0
    observer_arc_m: float = 0.0


class InterArmSafety:
    def __init__(self, config: InterArmSafetyConfig = InterArmSafetyConfig()) -> None:
        self.config = config
        self._active = False

    def evaluate(self, executor_points, observer_points):
        executor = np.asarray(executor_points, dtype=float)
        observer = np.asarray(observer_points, dtype=float)
        for points in (executor, observer):
            if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.all(np.isfinite(points)):
                raise ValueError("安全中心线必须包含至少两个有限三维点。")
        ex_lengths = np.linalg.norm(np.diff(executor, axis=0), axis=1)
        ob_lengths = np.linalg.norm(np.diff(observer, axis=0), axis=1)
        ex_arc = np.concatenate(([0.0], np.cumsum(ex_lengths)))
        ob_arc = np.concatenate(([0.0], np.cumsum(ob_lengths)))
        best = None
        for i in range(len(executor)-1):
            for j in range(len(observer)-1):
                a,b,s,t = closest_segments(executor[i], executor[i+1], observer[j], observer[j+1])
                distance = float(np.linalg.norm(b-a))
                if best is None or distance < best[0]:
                    best = (distance, b-a, ex_arc[i]+s*ex_lengths[i], ob_arc[j]+t*ob_lengths[j])
        distance, separation, ex_location, ob_location = best
        gap = distance-2*self.config.arm_radius_m
        threshold = self.config.release_distance_m if self._active else self.config.influence_distance_m
        self._active = gap < threshold
        direction = separation/distance if distance > 1e-12 else np.array([0.,-1.,0.])
        speed = self.config.avoidance_gain*max(self.config.release_distance_m-gap, 0.0) if self._active else 0.0
        return InterArmSafetyDecision(gap, speed*direction, self._active,
                                      gap <= self.config.critical_distance_m, ex_location, ob_location)
