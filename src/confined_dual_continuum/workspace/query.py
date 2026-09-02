"""环境查询端口；控制核心只依赖此协议。"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Clearance:
    distance_m: float
    point_world: np.ndarray
    normal_world: np.ndarray
    source: str


class WorkspaceQuery(Protocol):
    def nearest_clearance(self, points_world: np.ndarray) -> Clearance: ...

