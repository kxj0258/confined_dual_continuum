"""低维弯曲空间到物理肌腱相容子空间的映射。"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BendingSpaceModel:
    coupling_matrix: np.ndarray
    absolute_tolerance: float = 1.0e-9
    relative_tolerance: float = 1.0e-6

    def __post_init__(self) -> None:
        coupling = np.asarray(self.coupling_matrix, dtype=float)
        if coupling.ndim != 2 or not np.all(np.isfinite(coupling)):
            raise ValueError("coupling_matrix 必须是有限二维矩阵。")
        if np.linalg.matrix_rank(coupling) != coupling.shape[1]:
            raise ValueError("弯曲—肌腱耦合矩阵必须满列秩。")
        object.__setattr__(self, "coupling_matrix", coupling.copy())
        object.__setattr__(self, "_pseudoinverse", np.linalg.pinv(coupling))

    @property
    def bending_size(self) -> int:
        return self.coupling_matrix.shape[1]

    @property
    def tendon_count(self) -> int:
        return self.coupling_matrix.shape[0]

    def to_tendon(self, bending: np.ndarray) -> np.ndarray:
        values = self._vector(bending, self.bending_size, "bending")
        return self.coupling_matrix @ values

    def estimate(self, tendon: np.ndarray) -> np.ndarray:
        values = self._vector(tendon, self.tendon_count, "tendon")
        return self._pseudoinverse @ values

    def project(self, tendon: np.ndarray) -> np.ndarray:
        return self.to_tendon(self.estimate(tendon))

    def residual(self, tendon: np.ndarray) -> np.ndarray:
        values = self._vector(tendon, self.tendon_count, "tendon")
        return values - self.project(values)

    def is_compatible(self, tendon: np.ndarray) -> bool:
        values = self._vector(tendon, self.tendon_count, "tendon")
        tolerance = self.absolute_tolerance + self.relative_tolerance * np.linalg.norm(values)
        return float(np.linalg.norm(self.residual(values))) <= tolerance

    @staticmethod
    def _vector(values: np.ndarray, size: int, name: str) -> np.ndarray:
        result = np.asarray(values, dtype=float)
        if result.shape != (size,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} 必须是长度为 {size} 的有限向量。")
        return result.copy()

