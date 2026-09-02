"""把任务进度转换为统一阶段决策。"""

from dataclasses import dataclass

from confined_dual_continuum.task.phase_machine import (
    ConfinedOperationPhase,
    ConfinedOperationStateMachine,
    PhaseActivation,
)


@dataclass(frozen=True)
class OperationSignals:
    initialized: bool = False
    base_at_entry: bool = False
    insertion_complete: bool = False
    base_locked: bool = False
    local_operation_complete: bool = False
    retreat_complete: bool = False
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
        elif self._phase_complete(signals):
            self.machine.advance(self._completion_reason())
        else:
            self.machine.tick()
        return PhaseDecision(
            phase=self.machine.phase,
            activation=self.machine.activation,
            phase_step=self.machine.phase_step,
            transition_reason=self.machine.transition_reason,
        )

    def _phase_complete(self, signals: OperationSignals) -> bool:
        return {
            ConfinedOperationPhase.INITIALIZE: signals.initialized,
            ConfinedOperationPhase.BASE_APPROACH: signals.base_at_entry,
            ConfinedOperationPhase.COOPERATIVE_INSERTION: signals.insertion_complete,
            ConfinedOperationPhase.BASE_LOCK: signals.base_locked,
            ConfinedOperationPhase.LOCAL_OPERATION: signals.local_operation_complete,
            ConfinedOperationPhase.RETREAT: signals.retreat_complete,
        }.get(self.machine.phase, False)

    def _completion_reason(self) -> str:
        return f"{self.machine.phase.value}_complete"
