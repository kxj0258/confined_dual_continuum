"""固定时钟的基座系航点轨迹；不读取从臂状态。"""
import numpy as np


class TimedTrajectory:
    def __init__(self, times_s, points_base_m):
        self.times = np.asarray(times_s, dtype=float).copy()
        self.points = np.asarray(points_base_m, dtype=float).copy()
        if (self.times.ndim != 1 or len(self.times) < 2
                or self.points.shape != (len(self.times), 3)
                or not np.all(np.isfinite(self.times)) or not np.all(np.isfinite(self.points))
                or self.times[0] != 0. or np.any(np.diff(self.times) <= 0)):
            raise ValueError('航点要求从零开始严格递增的时间和有限 N×3 基座坐标。')

    @property
    def duration_s(self):
        return float(self.times[-1])

    def sample(self, time_s):
        """各航点零速度/零加速度连接；时间只由主任务时钟提供。"""
        if not np.isfinite(time_s):
            raise ValueError('轨迹时间必须有限。')
        t = float(np.clip(time_s, 0., self.duration_s))
        i = min(int(np.searchsorted(self.times, t, side='right')-1), len(self.times)-2)
        duration = self.times[i+1]-self.times[i]
        u = (t-self.times[i])/duration
        blend = 10*u**3-15*u**4+6*u**5
        derivative = (30*u**2-60*u**3+30*u**4)/duration
        delta = self.points[i+1]-self.points[i]
        return self.points[i]+blend*delta, derivative*delta
