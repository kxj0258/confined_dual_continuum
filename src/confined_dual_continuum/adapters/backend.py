"""仿真和物理机器人共同实现的后端端口。"""

from typing import Protocol

from confined_dual_continuum.domain.state import RobotCommand, RobotState


class RobotBackend(Protocol):
    def read_state(self) -> RobotState: ...

    def apply_command(self, command: RobotCommand) -> None: ...

    def step(self, dt: float) -> None: ...

