"""运行记录：同一控制前时刻的状态、意图和输出；失败仍保存证据。"""
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import numpy as np
from confined_dual_continuum.diagnostics.records import ControlDiagnostics
from confined_dual_continuum.domain.state import RobotState, RobotCommand


@dataclass(frozen=True)
class RunSummary:
    terminal_phase: str
    completed: bool
    steps: int
    duration_s: float
    termination_reason: str
    minimum_inter_arm_distance_m: float | None
    minimum_environment_clearance_m: float | None
    maximum_compatibility_residual: float | None
    maximum_actual_residual: float | None
    maximum_executor_contact_force_n: float | None
    executor_tracking_rmse_m: float | None
    observer_tracking_rmse_m: float | None
    maximum_target_lead_m: float | None
    saturation_count: int
    maximum_pcc_tip_error_m: float | None


class DiagnosticsRecorder:
    def __init__(self):
        self._rows = []

    def append(self, time_s: float, diagnostics: ControlDiagnostics, state: RobotState,
               command: RobotCommand, executor_target, observer_target):
        row = {"time_s": time_s, **diagnostics.as_dict()}
        for prefix, position in (("base", state.base_pose.position),
                                 ("executor_tip", state.executor.tip_pose.position),
                                 ("observer_tip", state.observer.tip_pose.position),
                                 ("executor_target", executor_target), ("observer_target", observer_target)):
            row.update({f"{prefix}_{axis}_m": float(v) for axis, v in zip("xyz", position)})
        for arm in ("executor", "observer"):
            feedback, output = getattr(state, arm), getattr(command, arm)
            for name, values in (("bending", feedback.bending), ("actual_tendon_m", feedback.tendon_displacement_m),
                                 ("target_tendon_m", output.tendon_target_m), ("tendon_rate_mps", output.tendon_rate_mps)):
                row.update({f"{arm}_{name}_{i}": float(v) for i,v in enumerate(values)})
        self._rows.append(row)

    def summary(self, terminal_phase: str, duration_s: float, reason: str, steps: int):
        def maximum(*keys):
            values = [float(row[key]) for row in self._rows for key in keys]
            return max(values) if values else None
        def minimum(key):
            return min((float(row[key]) for row in self._rows), default=None)
        def rmse(key, enabled):
            values = [float(row[key]) for row in self._rows if row[enabled] and row["target_phase"] == row["phase"]]
            return float(np.sqrt(np.mean(np.square(values)))) if values else None
        return RunSummary(
            terminal_phase, terminal_phase == "complete" and reason == "complete", steps, duration_s, reason,
            minimum("inter_arm_distance_m"), minimum("environment_clearance_m"),
            maximum("executor_compatibility_residual", "observer_compatibility_residual"),
            maximum("executor_actual_residual", "observer_actual_residual"),
            maximum("executor_contact_force_n"),
            rmse("executor_tracking_error_m", "executor_enabled"), rmse("observer_tracking_error_m", "observer_enabled"),
            maximum("executor_lead_m", "observer_lead_m"),
            sum(row["executor_scale"] < 1-1e-12 or row["observer_scale"] < 1-1e-12 for row in self._rows),
            maximum("executor_pcc_tip_error_m", "observer_pcc_tip_error_m"))

    def save(self, output_directory, summary):
        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self._rows:
            with (directory / "timeseries.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self._rows[0]))
                writer.writeheader()
                writer.writerows(self._rows)
        (directory / "summary.json").write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
