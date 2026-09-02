"""控制核心使用的最小几何类型。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def vector(values: np.ndarray, size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} 必须是长度为 {size} 的有限向量。")
    return result.copy()


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    quaternion_wxyz: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )

    def __post_init__(self) -> None:
        position = vector(self.position, 3, "position")
        quaternion = vector(self.quaternion_wxyz, 4, "quaternion_wxyz")
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1.0e-12:
            raise ValueError("四元数长度不能为零。")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "quaternion_wxyz", quaternion / norm)

