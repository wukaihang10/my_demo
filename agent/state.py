from dataclasses import asdict, dataclass, field
from typing import Any, Generic, TypeVar
from enum import Enum

from agent.plan import AgentPlan

TaskStateT = TypeVar("TaskStateT")


class RunStatus(str, Enum):
    INITIAL = "initial"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentState(Generic[TaskStateT]):
    """Task-independent state owned by the agent runtime."""

    task_state: TaskStateT
    status: RunStatus = RunStatus.INITIAL
    plan: AgentPlan | None = None
    errors: list[str] = field(default_factory=list)

    def add_error(self, error: str) -> None:
        self.errors.append(error)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
