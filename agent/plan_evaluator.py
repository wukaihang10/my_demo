import json

from collections.abc import Callable
from typing import Any

from agent.plan import AgentPlan
from agent.plan_update import (
    PlanUpdate,
    PlanUpdateValidationError,
)
from agent.state import RepositoryState
from llm.client import LLMClientError, chat

PlanEvaluationChatFunction = Callable[
    [list[dict[str, Any]], list[dict[str, Any]]],
    Any,
]


class PlanEvaluationError(RuntimeError):
    """Base exception raised while evaluating plan progress."""


class PlanEvaluationResponseError(PlanEvaluationError):
    """Raised when the evaluator returns an invalid response."""


PLAN_EVALUATOR_SYSTEM_PROMPT = """
You are the plan-progress evaluation component of an AI agent.

Your task is to inspect the current plan and the latest execution
evidence, then propose exactly one controlled plan update.

Return exactly one JSON object.

Allowed response structures:

1. Keep working on the current step:

{
  "action": "keep_current_step",
  "reason": "Why the current step is not complete yet"
}

2. Complete the current step:

{
  "action": "complete_current_step",
  "result": "The evidence or outcome that completed the step"
}

3. Skip the current step:

{
  "action": "skip_current_step",
  "reason": "Why the step is unnecessary or impossible"
}

4. Fail the current step:

{
  "action": "fail_current_step",
  "error": "The unrecoverable reason the step failed"
}

5. Append missing steps:

{
  "action": "append_steps",
  "steps": [
    {
      "description": "A new executable step",
      "completion_criteria": "How to determine it is complete"
    }
  ]
}

Rules:

1. Return JSON only. Do not use Markdown code fences.
2. Do not call tools.
3. Do not directly modify IDs, statuses, indexes or budgets.
4. Base the decision only on the supplied evidence.
5. Choose keep_current_step when the current step is still
   necessary but the completion criteria have not been satisfied.
6. Choose complete_current_step only when the evidence clearly
   satisfies the current step's completion criteria.
7. Choose skip_current_step only when the step is unnecessary,
   redundant or no longer applicable.
8. Choose fail_current_step only for an unrecoverable failure.
9. Choose append_steps only when the existing plan is missing
   necessary work.
10. Do not append duplicate or overlapping steps.
11. Propose exactly one action.
""".strip()


class LLMPlanProgressEvaluator:
    def __init__(
        self,
        *,
        chat_function: PlanEvaluationChatFunction | None = None,
    ) -> None:
        self._chat = chat_function or chat

    def evaluate_progress(
        self,
        *,
        plan: AgentPlan,
        state: RepositoryState,
        latest_evidence: list[dict[str, Any]],
        updates_remaining: int,
        max_total_steps: int,
        max_added_steps_per_update: int,
    ) -> PlanUpdate:
        if plan.status != "in_progress":
            raise PlanEvaluationError(
                "Plan progress can only be evaluated while " "the plan is in progress."
            )

        if plan.current_step is None:
            raise PlanEvaluationError("The running plan has no current step.")

        if updates_remaining < 0:
            raise ValueError("updates_remaining must not be negative.")

        if max_total_steps <= 0:
            raise ValueError("max_total_steps must be greater than zero.")

        if max_added_steps_per_update <= 0:
            raise ValueError("max_added_steps_per_update must be greater " "than zero.")

        state_payload = state.to_dict()

        # The plan is supplied separately below. Removing it avoids
        # sending the same data to the evaluator twice.
        state_payload.pop("plan", None)

        evaluation_payload = {
            "plan": plan.to_dict(),
            "agent_state": state_payload,
            "latest_evidence": latest_evidence,
            "plan_update_limits": {
                "updates_remaining": updates_remaining,
                "max_total_steps": max_total_steps,
                "max_added_steps_per_update": (max_added_steps_per_update),
            },
        }

        messages = [
            {
                "role": "system",
                "content": PLAN_EVALUATOR_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    "Evaluate the plan progress using the "
                    "following execution data:\n\n"
                    + json.dumps(
                        evaluation_payload,
                        ensure_ascii=False,
                        default=str,
                    )
                ),
            },
        ]

        try:
            response = self._chat(messages, [])
        except LLMClientError as error:
            raise PlanEvaluationError(
                "Plan evaluation request failed: " f"{error}"
            ) from error

        content = getattr(response, "content", None)

        if not isinstance(content, str) or not content.strip():
            raise PlanEvaluationResponseError("Plan evaluator returned empty content.")

        payload = self._parse_json_object(content)

        try:
            return PlanUpdate.from_dict(payload)
        except PlanUpdateValidationError as error:
            raise PlanEvaluationResponseError(
                "Plan evaluator returned an invalid update: " f"{error}"
            ) from error

    @staticmethod
    def _parse_json_object(
        content: str,
    ) -> dict[str, Any]:
        text = content.strip()

        # Defensive fallback for models that still return fenced JSON.
        if text.startswith("```"):
            lines = text.splitlines()

            if lines:
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            text = "\n".join(lines).strip()

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise PlanEvaluationResponseError(
                "Plan evaluator returned invalid JSON: " f"{error}"
            ) from error

        if not isinstance(payload, dict):
            raise PlanEvaluationResponseError(
                "Plan evaluator response must be a JSON object."
            )

        return payload
