"""小幅相容参考轨迹：固定基座，单坐标加载、保持、卸载与回零观察。"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class ResponsePlan:
    control_period_s: float
    amplitude_per_m: float
    axes: tuple[int, ...]
    settle_s: float
    ramp_s: float
    hold_s: float
    return_s: float
    rest_s: float

    def __post_init__(self):
        values = (self.control_period_s, self.amplitude_per_m, self.settle_s,
                  self.ramp_s, self.hold_s, self.return_s, self.rest_s)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("响应参数必须为有限正数。")
        if not self.axes or len(set(self.axes)) != len(self.axes) or any(type(i) is not int or i < 0 or i >= 6 for i in self.axes):
            raise ValueError("axes 必须是无重复的六维弯曲坐标索引。")
        for duration in values[2:]:
            if round(duration/self.control_period_s) < 1 or not math.isclose(round(duration/self.control_period_s)*self.control_period_s, duration, rel_tol=0, abs_tol=1e-10):
                raise ValueError("各阶段时长必须是控制周期的正整数倍。")

    @property
    def duration_s(self):
        return self.settle_s + 2*len(self.axes)*(self.ramp_s+self.hold_s+self.return_s+self.rest_s)

    def samples(self):
        """返回下一控制时刻的参考值；保持时间不冒充响应已经稳定。"""
        for _ in range(round(self.settle_s/self.control_period_s)):
            yield "settle", -1, 0, np.zeros(6)
        for axis in self.axes:
            for sign in (1, -1):
                for stage, duration, start, end in (
                    ("load", self.ramp_s, 0., 1.), ("hold", self.hold_s, 1., 1.),
                    ("unload", self.return_s, 1., 0.), ("rest", self.rest_s, 0., 0.)):
                    count = round(duration/self.control_period_s)
                    for i in range(count):
                        target = np.zeros(6)
                        target[axis] = sign*self.amplitude_per_m*(start+(end-start)*(i+1)/count)
                        yield stage, axis, sign, target
