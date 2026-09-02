"""不依赖具体机器人后端的速度级最小二乘求解。"""

import numpy as np


def solve_velocity(
    jacobian: np.ndarray,
    target_velocity: np.ndarray,
    *,
    damping: float = 1.0e-4,
) -> np.ndarray:
    matrix = np.asarray(jacobian, dtype=float)
    target = np.asarray(target_velocity, dtype=float)
    if matrix.ndim != 2 or target.shape != (matrix.shape[0],):
        raise ValueError("Jacobian 与目标速度维度不匹配。")
    if damping < 0.0:
        raise ValueError("damping 不能为负。")
    lhs = matrix @ matrix.T + damping * damping * np.eye(matrix.shape[0])
    return matrix.T @ np.linalg.solve(lhs, target)

