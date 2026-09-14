"""相容路径的梯形速度规划与稳定事件；没有误差合格阈值。"""
from dataclasses import dataclass
import numpy as np


@dataclass
class DriveEvents:
    stable_hold_s: float
    dwell_s: float = 0.
    sent_s: float | None = None
    stable_s: float | None = None
    quiet_elapsed_s: float = 0.

    def update(self, time_s, dt, sent, quiet, safe=True):
        if sent and self.sent_s is None:
            self.sent_s = float(time_s)
        self.quiet_elapsed_s = self.quiet_elapsed_s + dt if sent and quiet and safe else 0.
        if self.stable_s is None and self.quiet_elapsed_s + 1e-12 >= self.stable_hold_s:
            self.stable_s = float(time_s)

    @property
    def ready(self):
        # 稳定后保持期间再次运动，必须重新积累整个连续静止窗口。
        return self.quiet_elapsed_s + 1e-12 >= self.stable_hold_s + self.dwell_s

    def as_dict(self):
        return {"sent_s": self.sent_s, "stable_s": self.stable_s, "quiet_elapsed_s": self.quiet_elapsed_s}


@dataclass(frozen=True)
class TrapezoidProfile:
    """以位移最大的肌腱为标尺，单位 m、m/s、m/s²。

    所有肌腱共享路径进度；加减速绝对值固定，短行程自动成为三角形，
    不通过扩大幅值或加速度强行达到请求速度。
    """
    distance_m: float
    speed_mps: float
    acceleration_mps2: float

    def __post_init__(self):
        if not np.all(np.isfinite([self.distance_m, self.speed_mps, self.acceleration_mps2])):
            raise ValueError("轨迹参数必须有限")
        if self.distance_m < 0 or self.speed_mps <= 0 or self.acceleration_mps2 <= 0:
            raise ValueError("行程必须非负，速度和加速度必须为正")

    @property
    def peak_speed_mps(self):
        return min(self.speed_mps, float(np.sqrt(self.distance_m * self.acceleration_mps2)))

    @property
    def acceleration_time_s(self):
        return self.peak_speed_mps / self.acceleration_mps2

    @property
    def cruise_time_s(self):
        if self.speed_mps**2 >= self.acceleration_mps2 * self.distance_m:
            return 0.
        v = self.peak_speed_mps
        return max(0., self.distance_m / v - self.acceleration_time_s) if v else 0.

    @property
    def duration_s(self):
        return 2 * self.acceleration_time_s + self.cruise_time_s

    def sample(self, time_s):
        """返回解析路径位置、速度、加速度及阶段；切换处取右侧值。"""
        t = max(0., time_s)
        a, v = self.acceleration_mps2, self.peak_speed_mps
        ta, tc = self.acceleration_time_s, self.cruise_time_s
        if t >= self.duration_s:
            return self.distance_m, 0., 0., "hold"
        if t < ta:
            return .5*a*t*t, a*t, a, "accelerate"
        if t < ta + tc:
            return .5*a*ta*ta + v*(t-ta), v, 0., "cruise"
        remaining = self.duration_s - t
        return self.distance_m - .5*a*remaining*remaining, a*remaining, -a, "decelerate"


def tracking_statistics(records):
    """按阶段统计每根肌腱误差，不产生合格标签，不用保持阶段稀释运动误差。"""
    result = {}
    for phase in ("all", "accelerate", "cruise", "decelerate", "hold", "step"):
        rows = records if phase == "all" else [r for r in records if r["phase"] == phase]
        if not rows:
            continue
        values = {"cycles": len(rows)}
        for name, unit in (("velocity", "mps"), ("acceleration", "mps2")):
            error = np.asarray([r[f"actual_{name}_{unit}"] for r in rows]) - np.asarray([r[f"sent_{name}_{unit}"] for r in rows])
            values[name] = {"rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
                            "p95_abs": np.percentile(np.abs(error), 95, axis=0).tolist(),
                            "max_abs": np.max(np.abs(error), axis=0).tolist()}
        result[phase] = values
    return result
