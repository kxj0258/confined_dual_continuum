"""根据三段肌腱路由构造 bending-space 耦合矩阵 Cb。"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TendonRoute:
    angle_deg: float
    radial_offset_m: float
    path_segment_indices: tuple[int, ...]


def build_bending_coupling(
    segment_lengths_m: tuple[float, ...],
    routes: tuple[TendonRoute, ...],
) -> np.ndarray:
    coupling = np.zeros((len(routes), 2 * len(segment_lengths_m)))
    for row, route in enumerate(routes):
        angle = np.deg2rad(route.angle_deg)
        for segment in route.path_segment_indices:
            if segment < 0 or segment >= len(segment_lengths_m):
                raise ValueError("肌腱路由包含无效段索引。")
            length = segment_lengths_m[segment]
            coupling[row, 2 * segment : 2 * segment + 2] = (
                -length
                * route.radial_offset_m
                * np.array([np.cos(angle), np.sin(angle)])
            )
    return coupling

