"""把发动机实施例配置转换为通用受限空间模型。"""

from pathlib import Path

import numpy as np
import yaml

from confined_dual_continuum.workspace.model import (
    ConfinedWorkspace,
    Region,
    WorkspaceEntry,
)


def load_engine_workspace(path: str | Path) -> ConfinedWorkspace:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    engine = raw["engine"]
    pose = engine["pose"]
    scale = float(engine["scale"])
    translation = np.asarray(pose["position_m"], dtype=float) + np.asarray(
        pose.get("frame_offset_m", [0.0, 0.0, 0.0]), dtype=float
    )
    rotation = _quaternion_rotation(np.asarray(pose["quat_wxyz"], dtype=float))
    entry_raw = raw["regions"]["entry_port"]
    entry_center = _point_to_world(
        entry_raw["center_m"], entry_raw.get("frame", "world"), scale, translation, rotation
    )
    entry_normal = _direction_to_world(
        entry_raw["normal"], entry_raw.get("frame", "world"), rotation
    )
    path = next(item for item in raw["exploration_paths"] if item.get("enabled", True))
    task_center = _point_to_world(
        path["points_m"][-1], path.get("frame", "world"), scale, translation, rotation
    )
    return ConfinedWorkspace(
        entry=WorkspaceEntry(
            center_world=entry_center,
            inward_direction_world=entry_normal,
            radius_m=float(entry_raw["radius_m"]) * (scale if entry_raw.get("frame") == "engine" else 1.0),
        ),
        task_region=Region(
            name="engine_task_region",
            center_world=task_center,
            radius_m=float(path.get("radius_m", 0.008)) * (scale if path.get("frame") == "engine" else 1.0),
        ),
    )


def _point_to_world(values, frame, scale, translation, rotation) -> np.ndarray:
    point = np.asarray(values, dtype=float)
    return translation + rotation @ (scale * point) if frame == "engine" else point


def _direction_to_world(values, frame, rotation) -> np.ndarray:
    direction = np.asarray(values, dtype=float)
    return rotation @ direction if frame == "engine" else direction


def _quaternion_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = quaternion_wxyz / np.linalg.norm(quaternion_wxyz)
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )

