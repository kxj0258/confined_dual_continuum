"""通用受限空间 YAML 配置边界。"""

from pathlib import Path

import numpy as np
import yaml

from confined_dual_continuum.workspace.model import (
    ConfinedWorkspace,
    ContactSurface,
    Region,
    TaskPath,
    WorkspaceEntry,
)


def load_workspace(path: str | Path) -> ConfinedWorkspace:
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    entry = values["entry"]
    task = values["task_region"]
    surface = values["contact_surface"]
    stations = values["base_stations"]
    return ConfinedWorkspace(
        entry=WorkspaceEntry(
            center_world=np.asarray(entry["center_world_m"], dtype=float),
            inward_direction_world=np.asarray(entry["inward_direction_world"], dtype=float),
            radius_m=float(entry["radius_m"]),
        ),
        task_region=Region(
            name=str(task["name"]),
            center_world=np.asarray(task["center_world_m"], dtype=float),
            radius_m=float(task["radius_m"]),
        ),
        forbidden_regions=tuple(
            Region(
                name=str(region["name"]),
                center_world=np.asarray(region["center_world_m"], dtype=float),
                radius_m=float(region["radius_m"]),
            )
            for region in values.get("forbidden_regions", ())
        ),
        contact_surface=ContactSurface(
            point_world=np.asarray(surface["point_world_m"], dtype=float),
            normal_world=np.asarray(surface["normal_world"], dtype=float),
        ),
        confined_inspection=_task_path(values["confined_inspection"]),
        curved_inspection=_task_path(values["curved_inspection"]),
        wiping=_task_path(values["wiping"]),
        obstacle_operation=_task_path(values["obstacle_operation"]),
        entry_base_world=np.asarray(stations["entry"], dtype=float),
        curved_base_world=np.asarray(stations["curved_surface"], dtype=float),
        wiping_base_world=np.asarray(stations["wiping"], dtype=float),
        obstacle_base_world=np.asarray(stations["obstacle"], dtype=float),
    )


def _task_path(values: dict[str, object]) -> TaskPath:
    return TaskPath(
        name=str(values["name"]),
        points_world=np.asarray(values["points_world_m"], dtype=float),
        normal_world=np.asarray(values["normal_world"], dtype=float),
    )
