"""专利任务层的统一阶段与自由度激活规则。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ConfinedOperationPhase(str, Enum):
    INITIALIZE = "initialize"
    BASE_APPROACH = "base_approach"
    BASE_LOCK = "base_lock"
    CONFINED_VISUAL_INSPECTION = "confined_visual_inspection"
    CURVED_SURFACE_TRANSIT = "curved_surface_transit"
    CURVED_SURFACE_INSPECTION = "curved_surface_inspection"
    WIPING_TRANSIT = "wiping_transit"
    FORCE_POSITION_WIPING = "force_position_wiping"
    OBSTACLE_TRANSIT = "obstacle_transit"
    ASYMMETRIC_OBSTACLE_OPERATION = "asymmetric_obstacle_operation"
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
    ConfinedOperationPhase.BASE_LOCK: PhaseActivation(False, False, False),
    ConfinedOperationPhase.CONFINED_VISUAL_INSPECTION: PhaseActivation(False, True, True),
    ConfinedOperationPhase.CURVED_SURFACE_TRANSIT: PhaseActivation(True, False, False),
    ConfinedOperationPhase.CURVED_SURFACE_INSPECTION: PhaseActivation(False, True, True),
    ConfinedOperationPhase.WIPING_TRANSIT: PhaseActivation(True, False, False),
    ConfinedOperationPhase.FORCE_POSITION_WIPING: PhaseActivation(False, True, True),
    ConfinedOperationPhase.OBSTACLE_TRANSIT: PhaseActivation(True, False, False),
    ConfinedOperationPhase.ASYMMETRIC_OBSTACLE_OPERATION: PhaseActivation(False, True, True),
    ConfinedOperationPhase.COMPLETE: PhaseActivation(False, False, False),
    ConfinedOperationPhase.SAFETY_STOP: PhaseActivation(False, False, False),
}

NEXT_PHASE = {
    ConfinedOperationPhase.INITIALIZE: ConfinedOperationPhase.BASE_APPROACH,
    ConfinedOperationPhase.BASE_APPROACH: ConfinedOperationPhase.BASE_LOCK,
    ConfinedOperationPhase.BASE_LOCK: ConfinedOperationPhase.CONFINED_VISUAL_INSPECTION,
    ConfinedOperationPhase.CONFINED_VISUAL_INSPECTION: ConfinedOperationPhase.CURVED_SURFACE_TRANSIT,
    ConfinedOperationPhase.CURVED_SURFACE_TRANSIT: ConfinedOperationPhase.CURVED_SURFACE_INSPECTION,
    ConfinedOperationPhase.CURVED_SURFACE_INSPECTION: ConfinedOperationPhase.WIPING_TRANSIT,
    ConfinedOperationPhase.WIPING_TRANSIT: ConfinedOperationPhase.FORCE_POSITION_WIPING,
    ConfinedOperationPhase.FORCE_POSITION_WIPING: ConfinedOperationPhase.OBSTACLE_TRANSIT,
    ConfinedOperationPhase.OBSTACLE_TRANSIT: ConfinedOperationPhase.ASYMMETRIC_OBSTACLE_OPERATION,
    ConfinedOperationPhase.ASYMMETRIC_OBSTACLE_OPERATION: ConfinedOperationPhase.COMPLETE,
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
