"""可扩展解析轨迹与独立入场段；仅依赖数值参数。"""
import numpy as np
from confined_dual_continuum.task.timed_trajectory import TimedTrajectory


class CatalogTrajectory:
    def __init__(self, definition, time_scale=1.):
        self.definition = definition
        self.kind = definition['kind']
        self.center = np.asarray(definition['center_m'], float)
        self.u, self.v = np.asarray(definition['u'], float), np.asarray(definition['v'], float)
        self.a, self.b = map(float, definition['half_sizes_m'])
        self.entry_s = float(definition['entry_s'])*time_scale
        self.motion_s = float(definition['motion_s'])*time_scale
        if (self.kind not in ('circle', 'figure8', 'rectangle')
                or any(x.shape != (3,) or not np.all(np.isfinite(x)) for x in (self.center, self.u, self.v))
                or not np.allclose([self.u@self.u, self.v@self.v, self.u@self.v], [1, 1, 0], atol=1e-8)
                or not all(np.isfinite(x) and x > 0 for x in (self.a, self.b, self.entry_s, self.motion_s))):
            raise ValueError('轨迹类型、平面、尺寸或时间无效。')
        if self.kind == 'circle' and not np.isclose(self.a, self.b):
            raise ValueError('圆的两个半径必须相等。')
        entry = np.asarray(definition['entry_points_m'], float)
        if entry.ndim != 2 or entry.shape[1] != 3 or len(entry) < 2:
            raise ValueError('入场段至少需要两个点。')
        if not np.allclose(entry[-1], self.point(0.), atol=1e-10):
            raise ValueError('入场终点必须等于曲线起点。')
        self.entry = TimedTrajectory(np.linspace(0, self.entry_s, len(entry)), entry)

    @property
    def duration_s(self):
        return self.entry_s+self.motion_s

    def point(self, phase):
        s = np.asarray(phase, float)
        if self.kind == 'circle':
            x, y = self.a*np.cos(2*np.pi*s), self.a*np.sin(2*np.pi*s)
        elif self.kind == 'figure8':
            x, y = self.a*np.sin(2*np.pi*s), self.b*np.sin(4*np.pi*s)
        else:
            corners = np.array([[-self.a, -self.b], [self.a, -self.b],
                                [self.a, self.b], [-self.a, self.b], [-self.a, -self.b]])
            segment = np.minimum((s*4).astype(int), 3)
            fraction = s*4-segment
            xy = corners[segment]+fraction[..., None]*(corners[segment+1]-corners[segment])
            x, y = xy[..., 0], xy[..., 1]
        return self.center+x[..., None]*self.u+y[..., None]*self.v

    def sample(self, time_s):
        if not np.isfinite(time_s):
            raise ValueError('轨迹时间必须有限。')
        if time_s <= self.entry_s:
            return self.entry.sample(time_s)
        s = float(np.clip((time_s-self.entry_s)/self.motion_s, 0, 1))
        if self.kind == 'rectangle':
            segment = min(int(s*4), 3)
            u = s*4-segment
            blend = 10*u**3-15*u**4+6*u**5
            rate = (30*u*u-60*u**3+30*u**4)*4/self.motion_s
            p0, p1 = self.point(segment/4), self.point((segment+1)/4)
            return p0+blend*(p1-p0), rate*(p1-p0)
        phase = 10*s**3-15*s**4+6*s**5
        rate = (30*s*s-60*s**3+30*s**4)/self.motion_s
        theta = 2*np.pi*phase
        if self.kind == 'circle':
            derivative = 2*np.pi*(-self.a*np.sin(theta)*self.u+self.a*np.cos(theta)*self.v)
        else:
            derivative = 2*np.pi*(self.a*np.cos(theta)*self.u+2*self.b*np.cos(2*theta)*self.v)
        return self.point(phase), derivative*rate

    def coverage_points(self, maximum_step_m):
        # 导数范数的上界决定取样数；相邻点间曲线弧长不超过指定步长。
        length_bound = (4*(2*self.a+2*self.b) if self.kind == 'rectangle' else
                        2*np.pi*(self.a+(2*self.b if self.kind == 'figure8' else 0)))
        count = max(8, int(np.ceil(length_bound/maximum_step_m)))
        count = int(np.ceil(count/4))*4
        body = self.point(np.linspace(0, 1, count+1))
        entry = []
        for a, b in zip(self.entry.points[:-1], self.entry.points[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(b-a)/maximum_step_m)))
            entry.extend(a+(b-a)*np.linspace(0, 1, n+1)[:, None])
        return np.asarray(entry), body
