"""正式受限空间任务会话，贯通任务、协同、执行与后端。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from confined_dual_continuum.application.controller import (
    ConfinedDualContinuumController,
    ControlInput,
)
from confined_dual_continuum.application.ports import RobotBackend
from confined_dual_continuum.diagnostics.recorder import DiagnosticsRecorder, RunSummary
from confined_dual_continuum.domain.state import RobotState
from confined_dual_continuum.task.operation_director import OperationDirector


@dataclass(frozen=True)
class SessionConfig:
    control_period_s: float
    maximum_duration_s: float

    def __post_init__(self) -> None:
        if self.control_period_s <= 0.0 or self.maximum_duration_s <= 0.0:
            raise ValueError("控制周期和最大运行时间必须为正。")


class OperationSession:
    def __init__(
        self,
        backend: RobotBackend,
        controller: ConfinedDualContinuumController,
        director: OperationDirector,
        config: SessionConfig,
    ) -> None:
        self.backend = backend
        self.controller = controller
        self.director = director
        self.config = config

    def run(
        self,
        output_directory: str,
        on_step: Callable[[RobotState], bool | None] | None = None,
    ) -> RunSummary:
        state = self.backend.reset()
        recorder = DiagnosticsRecorder()
        maximum_steps = int(self.config.maximum_duration_s / self.config.control_period_s)
        for step in range(maximum_steps):
            setpoint = self.director.setpoint(
                self.controller.phase,
                self.controller.phase_step,
                state,
                self.config.control_period_s,
            )
            output = self.controller.step(ControlInput(
                state=state,
                signals=setpoint.signals,
                executor_target_world=setpoint.executor_target_world,
                observer_target_world=setpoint.observer_target_world,
                surface_normal_world=setpoint.surface_normal_world,
                executor_force_target_n=setpoint.executor_force_target_n,
                avoidance_arm=setpoint.avoidance_arm,
                requested_base_twist_world=setpoint.base_twist_world,
                dt=self.config.control_period_s,
            ))
            self.backend.apply_command(output.command)
            self.backend.step(self.config.control_period_s)
            state = self.backend.read_state()
            recorder.append((step + 1) * self.config.control_period_s, output.diagnostics, state)
            if on_step is not None and on_step(state) is False:
                break
            if self.controller.terminal:
                break
        summary = recorder.summary(self.controller.phase.value, self.config.control_period_s)
        recorder.save(output_directory, summary)
        return summary
