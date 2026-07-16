from dataclasses import asdict, dataclass
from typing import Any

from agent.plan import AgentPlan


@dataclass(frozen=True)
class FinalAnswerDecision:
    """Decision about whether an LLM response may end the run."""

    allowed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FinalAnswerPolicy:
    """Controls when a plain-text executor response may be treated as the final answer."""

    allow_direct_answer_before_tool_use: bool = True

    def evaluate(
        self,
        *,
        plan: AgentPlan | None,
        tool_calls_used: int,
        response_content: str,
    ) -> FinalAnswerDecision:
        if tool_calls_used < 0:
            raise ValueError("tool_calls_used must not be negative.")

        if not isinstance(response_content, str):
            raise TypeError("response_content must be a string.")

        if not response_content.strip():
            return FinalAnswerDecision(
                allowed=False,
                reason="The proposed final answer is empty.",
            )

        if plan is None:
            return FinalAnswerDecision(allowed=True, reason="No task plan is active.")

        if plan.status == "completed":
            return FinalAnswerDecision(
                allowed=True,
                reason="The task plan has been completed.",
            )

        if plan.status == "failed":
            return FinalAnswerDecision(
                allowed=False,
                reason="A failed task plan cannot be completed with a normal final answer.",
            )

        if tool_calls_used == 0 and self.allow_direct_answer_before_tool_use:
            return FinalAnswerDecision(
                allowed=True,
                reason="A direct answer is allowed before tool execution has started.",
            )

        return FinalAnswerDecision(
            allowed=False,
            reason="The task plan is still in progress after tool execution has started.",
        )
