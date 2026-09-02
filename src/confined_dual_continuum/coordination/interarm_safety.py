"""双臂安全监督独立于任务目标生成。"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InterArmSafetyConfig:
    influence_distance_m: float = 0.03
    critical_distance_m: float = 0.015
    avoidance_gain: float = 2.0


@dataclass(frozen=True)
class InterArmSafetyDecision:
    distance_m: float
    avoidance_velocity_world: np.ndarray
    avoidance_active: bool
    safety_stop: bool


class InterArmSafety:
    def __init__(self, config: InterArmSafetyConfig = InterArmSafetyConfig()) -> None:
        self.config = config

    def evaluate(self, executor_points: np.ndarray, observer_points: np.ndarray) -> InterArmSafetyDecision:
        delta = np.asarray(observer_points)[:, None, :] - np.asarray(executor_points)[None, :, :]
        distances = np.linalg.norm(delta, axis=2)
        observer_index, executor_index = np.unravel_index(np.argmin(distances), distances.shape)
        distance = float(distances[observer_index, executor_index])
        separation = delta[observer_index, executor_index]
        direction = (
            separation / distance
            if distance > 1.0e-12
            else np.array([0.0, -1.0, 0.0])
        )
        avoidance_active = distance < self.config.influence_distance_m
        speed = self.config.avoidance_gain * max(
            self.config.influence_distance_m - distance,
            0.0,
        )
        return InterArmSafetyDecision(
            distance_m=distance,
            avoidance_velocity_world=speed * direction,
            avoidance_active=avoidance_active,
            safety_stop=distance <= self.config.critical_distance_m,
        )
