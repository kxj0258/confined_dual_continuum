"""应用层对运行后端的单向端口。"""

from typing import Protocol

from confined_dual_continuum.domain.state import RobotCommand, RobotState


class RobotBackend(Protocol):
    def reset(self) -> RobotState: ...

    def read_state(self) -> RobotState: ...

    def apply_command(self, command: RobotCommand) -> None: ...

    def step(self, dt: float) -> None: ...
