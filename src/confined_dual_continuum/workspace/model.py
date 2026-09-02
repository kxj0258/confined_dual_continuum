"""与具体场景几何实现无关的受限空间描述。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from confined_dual_continuum.domain.geometry import vector


@dataclass(frozen=True)
class WorkspaceEntry:
    center_world: np.ndarray
    inward_direction_world: np.ndarray
    radius_m: float

    def __post_init__(self) -> None:
        center = vector(self.center_world, 3, "entry.center_world")
        direction = vector(self.inward_direction_world, 3, "entry.inward_direction")
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-12 or self.radius_m <= 0.0:
            raise ValueError("入口方向必须非零且半径必须为正。")
        object.__setattr__(self, "center_world", center)
        object.__setattr__(self, "inward_direction_world", direction / norm)


@dataclass(frozen=True)
class Region:
    name: str
    center_world: np.ndarray
    radius_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center_world", vector(self.center_world, 3, self.name))
        if not self.name or self.radius_m <= 0.0:
            raise ValueError("区域必须具有名称和正半径。")


@dataclass(frozen=True)
class ContactSurface:
    point_world: np.ndarray
    normal_world: np.ndarray

    def __post_init__(self) -> None:
        point = vector(self.point_world, 3, "surface.point_world")
        normal = vector(self.normal_world, 3, "surface.normal_world")
        norm = float(np.linalg.norm(normal))
        if norm <= 1.0e-12:
            raise ValueError("接触面法向不能为零。")
        object.__setattr__(self, "point_world", point)
        object.__setattr__(self, "normal_world", normal / norm)


@dataclass(frozen=True)
class ConfinedWorkspace:
    entry: WorkspaceEntry
    task_region: Region
    forbidden_regions: tuple[Region, ...] = ()
    contact_surface: ContactSurface | None = None
