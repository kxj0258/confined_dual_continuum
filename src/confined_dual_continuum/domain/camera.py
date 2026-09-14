"""控制用相机系：+X 向右、+Y 向下、+Z 向前；不依赖渲染后端。"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CameraModel:
    fovy_deg: float
    width: int
    height: int
    near_m: float
    far_m: float
    edge_fraction: float
    target_radius_m: float

    def __post_init__(self):
        if (not all(np.isfinite(v) for v in (self.fovy_deg, self.width, self.height,
                self.near_m, self.far_m, self.edge_fraction, self.target_radius_m))
                or not 0 < self.fovy_deg < 180 or min(self.width, self.height) <= 0
                or int(self.width) != self.width or int(self.height) != self.height
                or not 0 < self.near_m < self.far_m or not 0 <= self.edge_fraction < 1
                or self.target_radius_m < 0):
            raise ValueError('相机视场、深度、边缘余量或目标半径无效。')

    @property
    def slopes(self):
        vertical = np.tan(np.deg2rad(self.fovy_deg)/2)*(1-self.edge_fraction)
        return np.array([vertical*self.width/self.height, vertical])

    def planes(self):
        """A p >= b：用球半径保守收缩四个侧面和两个深度平面。"""
        x, y = self.slopes
        a = np.array([[1., 0., x], [-1., 0., x], [0., 1., y], [0., -1., y],
                      [0., 0., 1.], [0., 0., -1.]])
        b = np.array([0., 0., 0., 0., self.near_m, -self.far_m])
        return a, b+self.target_radius_m*np.linalg.norm(a, axis=1)

    def margins(self, point_camera):
        p = np.asarray(point_camera, dtype=float)
        if p.shape != (3,) or not np.all(np.isfinite(p)):
            raise ValueError('目标必须是有限相机系三维点。')
        a, b = self.planes()
        return (a@p-b)/np.linalg.norm(a, axis=1)

    def visible(self, point_camera, occluded):
        return bool(not occluded and np.min(self.margins(point_camera)) >= 0.)
