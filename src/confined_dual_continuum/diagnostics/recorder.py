"""正式实验运行记录：CSV 时序与 JSON 汇总。"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from confined_dual_continuum.diagnostics.records import ControlDiagnostics
from confined_dual_continuum.domain.state import RobotState


@dataclass(frozen=True)
class RunSummary:
    terminal_phase: str
    completed: bool
    steps: int
    duration_s: float
    minimum_inter_arm_distance_m: float
    maximum_compatibility_residual: float
    maximum_executor_contact_force_n: float


class DiagnosticsRecorder:
    def __init__(self) -> None:
        self._rows: list[dict[str, object]] = []

    def append(
        self,
        time_s: float,
        diagnostics: ControlDiagnostics,
        state: RobotState,
    ) -> None:
        row = {"time_s": time_s, **diagnostics.as_dict()}
        for prefix, position in (
            ("base", state.base_pose.position),
            ("executor_tip", state.executor.tip_pose.position),
            ("observer_tip", state.observer.tip_pose.position),
        ):
            row.update({
                f"{prefix}_{axis}_m": float(value)
                for axis, value in zip("xyz", position)
            })
        self._rows.append(row)

    def summary(self, terminal_phase: str, dt: float) -> RunSummary:
        distances = [float(row["inter_arm_distance_m"]) for row in self._rows]
        residuals = [
            max(
                float(row["executor_compatibility_residual"]),
                float(row["observer_compatibility_residual"]),
            )
            for row in self._rows
        ]
        return RunSummary(
            terminal_phase=terminal_phase,
            completed=terminal_phase == "complete",
            steps=len(self._rows),
            duration_s=len(self._rows) * dt,
            minimum_inter_arm_distance_m=min(distances, default=float("inf")),
            maximum_compatibility_residual=max(residuals, default=0.0),
            maximum_executor_contact_force_n=max(
                (float(row["executor_contact_force_n"]) for row in self._rows),
                default=0.0,
            ),
        )

    def save(self, output_directory: str | Path, summary: RunSummary) -> None:
        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self._rows:
            with (directory / "timeseries.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self._rows[0]))
                writer.writeheader()
                writer.writerows(self._rows)
        (directory / "summary.json").write_text(
            json.dumps(asdict(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
