from dataclasses import dataclass, field
from typing import Any

from agent.outcome import AgentRunOutcome


@dataclass
class ToolBatchResult:
    """Result of executing one executor tool-call batch."""

    latest_evidence: list[dict[str, Any]] = field(default_factory=list)

    terminal_outcome: AgentRunOutcome | None = None

    @property
    def should_stop(self) -> bool:
        return self.terminal_outcome is not None


@dataclass
class PostToolBatchResult:
    terminal_outcome: AgentRunOutcome | None = None
    should_continue: bool = False

    @property
    def should_stop(self) -> bool:
        return self.terminal_outcome is not None
