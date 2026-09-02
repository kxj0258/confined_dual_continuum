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

    def as_dict(self) -> dict[str, object]:
        # 字典化仅发生在记录边界，控制层内部不依赖字符串键通信。
        return asdict(self)

