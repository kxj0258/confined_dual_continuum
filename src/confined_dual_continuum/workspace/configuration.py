"""通用受限空间 YAML 配置边界。"""

from pathlib import Path

import numpy as np
import yaml

from confined_dual_continuum.workspace.model import (
    ConfinedWorkspace,
    ContactSurface,
    Region,
    WorkspaceEntry,
)


def load_workspace(path: str | Path) -> ConfinedWorkspace:
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    entry = values["entry"]
    task = values["task_region"]
    surface = values["contact_surface"]
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
    )
