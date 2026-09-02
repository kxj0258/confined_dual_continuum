"""根据 Executor 当前作业点生成 Observer 动态辅助目标。"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObserverTargetConfig:
    normal_offset_m: float = 0.08
    tangential_offset_m: float = 0.02
    max_target_speed_mps: float = 0.02


@dataclass(frozen=True)
class ObserverTarget:
    position_world: np.ndarray
    raw_position_world: np.ndarray
    limited: bool


class ObserverTargetGenerator:
    def __init__(self, config: ObserverTargetConfig = ObserverTargetConfig()) -> None:
        self.config = config
        self._previous: np.ndarray | None = None

    def generate(
        self,
        task_point_world: np.ndarray,
        surface_normal_world: np.ndarray,
        tangent_world: np.ndarray,
        dt: float,
    ) -> ObserverTarget:
        normal = self._unit(surface_normal_world, "surface_normal_world")
        tangent = self._unit(tangent_world, "tangent_world")
        raw = (
            np.asarray(task_point_world, dtype=float)
            + self.config.normal_offset_m * normal
            + self.config.tangential_offset_m * tangent
        )
        if self._previous is None:
            target = raw.copy()
        else:
            delta = raw - self._previous
            distance = float(np.linalg.norm(delta))
            limit = self.config.max_target_speed_mps * dt
            target = raw if distance <= limit else self._previous + limit * delta / distance
        limited = not np.allclose(target, raw)
        self._previous = target.copy()
        return ObserverTarget(target, raw, limited)

    @staticmethod
    def _unit(values: np.ndarray, name: str) -> np.ndarray:
        vector = np.asarray(values, dtype=float)
        norm = float(np.linalg.norm(vector))
        if vector.shape != (3,) or not np.isfinite(norm) or norm <= 1.0e-12:
            raise ValueError(f"{name} 必须是非零三维向量。")
        return vector / norm
