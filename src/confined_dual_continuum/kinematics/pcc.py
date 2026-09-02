"""三段常曲率连续体运动学。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCCResult:
    tip_transform: np.ndarray
    centerline: np.ndarray


def constant_curvature_transform(bending: np.ndarray, length_m: float) -> np.ndarray:
    """计算单段 SE(3) 变换；bending 为 [kx, ky]，单位 1/m。"""

    kx, ky = _vector(bending, 2, "bending")
    if length_m < 0.0:
        raise ValueError("length_m 不能为负。")
    omega = np.array([-ky, kx, 0.0])
    curvature = float(np.linalg.norm(omega))
    transform = np.eye(4)
    if curvature < 1.0e-10:
        transform[2, 3] = length_m
        return transform

    omega_hat = _skew(omega)
    theta = curvature * length_m
    rotation = (
        np.eye(3)
        + np.sin(theta) / curvature * omega_hat
        + (1.0 - np.cos(theta)) / curvature**2 * (omega_hat @ omega_hat)
    )
    translation_operator = (
        length_m * np.eye(3)
        + (1.0 - np.cos(theta)) / curvature**2 * omega_hat
        + (theta - np.sin(theta)) / curvature**3 * (omega_hat @ omega_hat)
    )
    transform[:3, :3] = rotation
    transform[:3, 3] = translation_operator @ np.array([0.0, 0.0, 1.0])
    return transform


def forward_kinematics(
    bending: np.ndarray,
    flexure_lengths_m: tuple[float, ...],
    distal_offsets_m: tuple[float, ...],
    *,
    samples_per_segment: int = 8,
) -> PCCResult:
    """计算机械臂末端位姿及去重后的完整中心线。"""

    if samples_per_segment < 2:
        raise ValueError("samples_per_segment 至少为 2。")
    if len(distal_offsets_m) != len(flexure_lengths_m):
        raise ValueError("柔性段长度与段末偏置数量必须一致。")
    if any(offset < 0.0 for offset in distal_offsets_m):
        raise ValueError("段末偏置不能为负。")
    values = _vector(bending, 2 * len(flexure_lengths_m), "bending")
    current = np.eye(4)
    points: list[np.ndarray] = []
    for index, (length, distal_offset) in enumerate(
        zip(flexure_lengths_m, distal_offsets_m, strict=True)
    ):
        segment_bending = values[2 * index : 2 * index + 2]
        for sample_index, partial_length in enumerate(
            np.linspace(0.0, float(length), samples_per_segment)
        ):
            if points and sample_index == 0:
                continue
            local = constant_curvature_transform(segment_bending, float(partial_length))
            points.append((current @ local)[:3, 3].copy())
        current = current @ constant_curvature_transform(segment_bending, float(length))
        if distal_offset > 0.0:
            # 刚性连接区保持上一段末端朝向，不参与曲率积分。
            distal_tip = current[:3, 3] + current[:3, 2] * float(distal_offset)
            points.append(distal_tip.copy())
            current = current.copy()
            current[:3, 3] = distal_tip
    return PCCResult(current, np.asarray(points))


def position_jacobian(
    bending: np.ndarray,
    flexure_lengths_m: tuple[float, ...],
    distal_offsets_m: tuple[float, ...],
    *,
    step: float = 1.0e-5,
) -> np.ndarray:
    """用中心差分计算末端位置对六维弯曲坐标的雅可比。"""

    values = _vector(bending, 2 * len(flexure_lengths_m), "bending")
    if step <= 0.0:
        raise ValueError("step 必须为正。")
    jacobian = np.zeros((3, values.size))
    for index in range(values.size):
        offset = np.zeros_like(values)
        offset[index] = step
        plus = forward_kinematics(
            values + offset, flexure_lengths_m, distal_offsets_m
        ).tip_transform[:3, 3]
        minus = forward_kinematics(
            values - offset, flexure_lengths_m, distal_offsets_m
        ).tip_transform[:3, 3]
        jacobian[:, index] = (plus - minus) / (2.0 * step)
    return jacobian


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """将局部中心线变换到世界坐标系。"""

    matrix = np.asarray(transform, dtype=float)
    values = np.asarray(points, dtype=float)
    if matrix.shape != (4, 4) or values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("transform 或 points 形状无效。")
    homogeneous = np.column_stack((values, np.ones(values.shape[0])))
    return (matrix @ homogeneous.T).T[:, :3]


def _vector(values: np.ndarray, size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} 必须是长度为 {size} 的有限向量。")
    return result


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
