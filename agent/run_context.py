from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from agent.outcome import AgentRunOutcome
from agent.state import AgentState
from agent.stagnation import StagnationTracker
from agent.tool_history import ToolHistory
from agent.trace import AgentTrace
from llm.messages import ChatMessage

TaskStateT = TypeVar("TaskStateT")


@dataclass
class RunContext(Generic[TaskStateT]):
    """Mutable data belonging to one Agent.run() execution."""

    user_input: str
    task_input: dict[str, Any]

    state: AgentState[TaskStateT]
    trace: AgentTrace
    messages: list[ChatMessage]
    tool_history: ToolHistory

    stagnation_tracker: StagnationTracker | None = None
    outcome: AgentRunOutcome | None = None
