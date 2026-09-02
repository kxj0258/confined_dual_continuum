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
class TaskPath:
    name: str
    points_world: np.ndarray
    normal_world: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points_world, dtype=float)
        normal = vector(self.normal_world, 3, f"{self.name}.normal_world")
        if not self.name or points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
            raise ValueError("任务路径必须具有名称和至少两个三维点。")
        if not np.all(np.isfinite(points)) or np.linalg.norm(normal) <= 1.0e-12:
            raise ValueError("任务路径必须包含有限坐标和非零法向。")
        object.__setattr__(self, "points_world", points.copy())
        object.__setattr__(self, "normal_world", normal / np.linalg.norm(normal))


@dataclass(frozen=True)
class ConfinedWorkspace:
    entry: WorkspaceEntry
    task_region: Region
    forbidden_regions: tuple[Region, ...] = ()
    contact_surface: ContactSurface | None = None
    confined_inspection: TaskPath | None = None
    curved_inspection: TaskPath | None = None
    wiping: TaskPath | None = None
    obstacle_operation: TaskPath | None = None
    entry_base_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    curved_base_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    wiping_base_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    obstacle_base_world: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        for name in (
            "entry_base_world",
            "curved_base_world",
            "wiping_base_world",
            "obstacle_base_world",
        ):
            object.__setattr__(self, name, vector(getattr(self, name), 3, name))
