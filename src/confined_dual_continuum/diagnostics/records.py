"""专利主链的强类型诊断记录。"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ControlDiagnostics:
    phase: str
    phase_step: int
    transition_reason: str
    base_enabled: bool
    executor_enabled: bool
    observer_enabled: bool
    inter_arm_distance_m: float
    executor_compatibility_residual: float
    observer_compatibility_residual: float
    inter_arm_avoidance_active: bool
    safety_stop: bool
    executor_contact_force_n: float
    avoidance_arm: str
    target_phase: str
    waypoint_index: int
    executor_tracking_error_m: float
    observer_tracking_error_m: float
    observer_view_error_rad: float
    executor_force_error_n: float
    environment_clearance_m: float
    executor_actual_residual: float
    observer_actual_residual: float
    executor_lead_m: float
    observer_lead_m: float
    executor_scale: float
    observer_scale: float
    constraints_feasible: bool
    executor_pcc_tip_error_m: float
    observer_pcc_tip_error_m: float
    executor_constraint_status: str
    observer_constraint_status: str
    executor_limiting_tendon_index: int
    observer_limiting_tendon_index: int
    executor_lead_excess_m: float
    observer_lead_excess_m: float

    def as_dict(self) -> dict[str, object]:
        # 字典化仅发生在记录边界，控制层内部不依赖字符串键通信。
        return asdict(self)
