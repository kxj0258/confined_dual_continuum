"""正式任务会话；停止、超时、用户中止和异常采用不同结束原因。"""
from collections.abc import Callable
from dataclasses import dataclass
import math
from confined_dual_continuum.application.controller import ConfinedDualContinuumController, ControlInput
from confined_dual_continuum.application.ports import RobotBackend
from confined_dual_continuum.diagnostics.recorder import DiagnosticsRecorder, RunSummary
from confined_dual_continuum.domain.state import RobotState
from confined_dual_continuum.task.operation_director import OperationDirector


@dataclass(frozen=True)
class SessionConfig:
    control_period_s: float
    maximum_duration_s: float

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in (self.control_period_s, self.maximum_duration_s)):
            raise ValueError("控制周期和最大运行时间必须为有限正数。")
        if self.maximum_duration_s < self.control_period_s:
            raise ValueError("运行时间必须至少包含一个控制周期。")


class OperationSession:
    def __init__(self, backend: RobotBackend, controller: ConfinedDualContinuumController,
                 director: OperationDirector, config: SessionConfig):
        self.backend, self.controller, self.director, self.config = backend, controller, director, config

    def run(self, output_directory: str, on_step: Callable[[RobotState], bool | None] | None = None) -> RunSummary:
        recorder = DiagnosticsRecorder()
        steps = 0
        reason = "maximum_duration"
        dt = self.config.control_period_s
        try:
            state = self.backend.reset()
            self.controller.reset(state)
            self.director.reset()
            for _ in range(int(self.config.maximum_duration_s / dt)):
                setpoint = self.director.setpoint(self.controller.phase, self.controller.phase_step, state, dt)
                output = self.controller.step(ControlInput(
                    state=state, signals=setpoint.signals,
                    executor_target_world=setpoint.executor_target_world,
                    observer_target_world=setpoint.observer_target_world,
                    surface_normal_world=setpoint.surface_normal_world,
                    executor_force_target_n=setpoint.executor_force_target_n,
                    avoidance_arm=setpoint.avoidance_arm,
                    requested_base_twist_world=setpoint.base_twist_world, dt=dt,
                    observer_view_direction_world=setpoint.observer_view_direction_world,
                    waypoint_index=setpoint.waypoint_index))
                recorder.append(steps*dt, output.diagnostics, state, output.command,
                                setpoint.executor_target_world, setpoint.observer_target_world)
                if self.controller.terminal:
                    reason = "complete" if self.controller.phase.value == "complete" else output.diagnostics.transition_reason
                    break
                self.backend.apply_command(output.command)
                self.backend.step(dt)
                steps += 1
                state = self.backend.read_state()
                if on_step is not None and on_step(state) is False:
                    reason = "user_cancelled"
                    break
        except BaseException as exc:
            reason = f"exception:{type(exc).__name__}:{exc}"
            raise
        finally:
            try:
                self.backend.stop()
            finally:
                summary = recorder.summary(self.controller.phase.value, steps*dt, reason, steps)
                recorder.save(output_directory, summary)
        return summary
