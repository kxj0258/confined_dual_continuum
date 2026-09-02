"""把任务进度转换为统一阶段决策。"""

from dataclasses import dataclass

from confined_dual_continuum.task.phase_machine import (
    ConfinedOperationPhase,
    ConfinedOperationStateMachine,
    PhaseActivation,
)


@dataclass(frozen=True)
class OperationSignals:
    phase_complete: bool = False
    completion_reason: str = ""
    safety_violation: str | None = None


@dataclass(frozen=True)
class PhaseDecision:
    phase: ConfinedOperationPhase
    activation: PhaseActivation
    phase_step: int
    transition_reason: str


class ConfinedOperationSupervisor:
    """只判断任务阶段，不直接调用任何控制器。"""

    def __init__(self) -> None:
        self.machine = ConfinedOperationStateMachine()

    def update(self, signals: OperationSignals) -> PhaseDecision:
        if signals.safety_violation and not self.machine.terminal:
            self.machine.safety_stop(signals.safety_violation)
        elif signals.phase_complete:
            reason = signals.completion_reason or f"{self.machine.phase.value}_complete"
            self.machine.advance(reason)
        else:
            self.machine.tick()
        return PhaseDecision(
            phase=self.machine.phase,
            activation=self.machine.activation,
            phase_step=self.machine.phase_step,
            transition_reason=self.machine.transition_reason,
        )
