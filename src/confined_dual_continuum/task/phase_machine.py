"""专利任务层的统一阶段与自由度激活规则。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ConfinedOperationPhase(str, Enum):
    INITIALIZE = "initialize"
    BASE_APPROACH = "base_approach"
    COOPERATIVE_INSERTION = "cooperative_insertion"
    BASE_LOCK = "base_lock"
    LOCAL_OPERATION = "local_operation"
    RETREAT = "retreat"
    COMPLETE = "complete"
    SAFETY_STOP = "safety_stop"


@dataclass(frozen=True)
class PhaseActivation:
    base_enabled: bool
    executor_enabled: bool
    observer_enabled: bool


ACTIVATION = {
    ConfinedOperationPhase.INITIALIZE: PhaseActivation(False, False, False),
    ConfinedOperationPhase.BASE_APPROACH: PhaseActivation(True, False, False),
    ConfinedOperationPhase.COOPERATIVE_INSERTION: PhaseActivation(True, True, True),
    ConfinedOperationPhase.BASE_LOCK: PhaseActivation(False, False, False),
    ConfinedOperationPhase.LOCAL_OPERATION: PhaseActivation(False, True, True),
    ConfinedOperationPhase.RETREAT: PhaseActivation(True, True, True),
    ConfinedOperationPhase.COMPLETE: PhaseActivation(False, False, False),
    ConfinedOperationPhase.SAFETY_STOP: PhaseActivation(False, False, False),
}

NEXT_PHASE = {
    ConfinedOperationPhase.INITIALIZE: ConfinedOperationPhase.BASE_APPROACH,
    ConfinedOperationPhase.BASE_APPROACH: ConfinedOperationPhase.COOPERATIVE_INSERTION,
    ConfinedOperationPhase.COOPERATIVE_INSERTION: ConfinedOperationPhase.BASE_LOCK,
    ConfinedOperationPhase.BASE_LOCK: ConfinedOperationPhase.LOCAL_OPERATION,
    ConfinedOperationPhase.LOCAL_OPERATION: ConfinedOperationPhase.RETREAT,
    ConfinedOperationPhase.RETREAT: ConfinedOperationPhase.COMPLETE,
}


class ConfinedOperationStateMachine:
    def __init__(self) -> None:
        self.phase = ConfinedOperationPhase.INITIALIZE
        self.phase_step = 0
        self.transition_reason = "initialized"

    @property
    def activation(self) -> PhaseActivation:
        return ACTIVATION[self.phase]

    @property
    def terminal(self) -> bool:
        return self.phase in {
            ConfinedOperationPhase.COMPLETE,
            ConfinedOperationPhase.SAFETY_STOP,
        }

    def tick(self) -> None:
        self.phase_step += 1

    def advance(self, reason: str) -> None:
        if self.phase not in NEXT_PHASE:
            raise ValueError(f"阶段 {self.phase.value} 没有后继阶段。")
        self._enter(NEXT_PHASE[self.phase], reason)

    def safety_stop(self, reason: str) -> None:
        if self.terminal:
            raise ValueError("终态不能再次进入安全停止。")
        self._enter(ConfinedOperationPhase.SAFETY_STOP, reason)

    def _enter(self, phase: ConfinedOperationPhase, reason: str) -> None:
        reason = str(reason).strip()
        if not reason:
            raise ValueError("阶段转换必须记录原因。")
        self.phase = phase
        self.phase_step = 0
        self.transition_reason = reason

