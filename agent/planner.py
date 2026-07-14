import json
from collections.abc import Callable
from typing import Any

from agent.plan import (
  AgentPlan,
  PlanStepSpec,
  PlanValidationError,
)
from llm.client import LLMClientError, chat


ChatFunction = Callable[
  [list[dict[str, Any]], list[dict[str, Any]]],
  Any,
]


class PlannerError(RuntimeError):
  """Base exception raised while creating an Agent plan."""


class PlannerResponseError(PlannerError):
  """Raised when the planner returns an invalid response."""


PLANNER_SYSTEM_PROMPT = """
You are the planning component of a general-purpose AI agent.

Your task is to divide the user's goal into a small, executable sequence of
high-level steps.

Return exactly one JSON object using this structure:

{
  "steps": [
    {
      "description": "A concise executable step",
      "completion_criteria": "How the agent can determine that the step is done"
    }
  ]
}

Rules:

1. Return JSON only. Do not use Markdown code fences.
2. Do not answer the user's task.
3. Do not call tools.
4. Do not include step IDs.
5. Do not include status, result, error or progress fields.
6. Each step must describe an outcome, not a specific tool call.
7. Use as few steps as reasonably necessary.
8. The steps must be ordered.
9. Do not include duplicate or overlapping steps.
""".strip()


class LLMPlanner:
  def __init__(
    self,
    *,
    max_plan_steps: int = 6,
    chat_function: ChatFunction | None = None,
  ) -> None:
    if max_plan_steps <= 0:
      raise ValueError("max_plan_steps must be greater than zero.")

    self.max_plan_steps = max_plan_steps
    self._chat = chat_function or chat

  def create_plan(self, goal: str) -> AgentPlan:
    goal = goal.strip()

    if not goal:
      raise PlannerError("Cannot create a plan for an empty goal.")

    messages = [
      {
        "role": "system",
        "content": PLANNER_SYSTEM_PROMPT,
      },
      {
        "role": "user",
        "content": (
          f"Create a plan for the following goal.\n\n"
          f"Goal:\n{goal}\n\n"
          f"The plan must contain no more than "
          f"{self.max_plan_steps} steps."
        ),
      },
    ]

    try:
      response = self._chat(messages, [])
    except LLMClientError as error:
      raise PlannerError(f"Plan generation request failed: {error}") from error

    content = getattr(response, "content", None)

    if not isinstance(content, str) or not content.strip():
      raise PlannerResponseError("Planner returned no textual content.")

    payload = self._parse_json_object(content)
    step_specs = self._parse_step_specs(payload)

    try:
      return AgentPlan.from_step_specs(
        goal=goal,
        step_specs=step_specs,
        max_steps=self.max_plan_steps,
      )
    except PlanValidationError as error:
      raise PlannerResponseError(f"Planner produced an invalid plan: {error}") from error

  def _parse_json_object(
    self,
    content: str,
  ) -> dict[str, Any]:
    text = content.strip()

    # A small defensive fallback for models that still return a fenced
    # JSON block despite the prompt.
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
      raise PlannerResponseError(f"Planner returned invalid JSON: {error}") from error

    if not isinstance(payload, dict):
      raise PlannerResponseError("Planner response must be a JSON object.")

    unknown_fields = set(payload) - {"steps"}

    if unknown_fields:
      raise PlannerResponseError("Planner response contains unsupported fields: "
        + 
      ", ".join(sorted(unknown_fields)))

    return payload

  def _parse_step_specs(
    self,
    payload: dict[str, Any],
  ) -> list[PlanStepSpec]:
    raw_steps = payload.get("steps")

    if not isinstance(raw_steps, list):
      raise PlannerResponseError("Planner response field 'steps' must be a list.")

    if not raw_steps:
      raise PlannerResponseError("Planner must return at least one step.")

    if len(raw_steps) > self.max_plan_steps:
      raise PlannerResponseError(f"Planner returned {len(raw_steps)} steps, but the limit is {self.max_plan_steps}.")

    step_specs: list[PlanStepSpec] = []

    for index, raw_step in enumerate(raw_steps, start=1):
      if not isinstance(raw_step, dict):
        raise PlannerResponseError(f"Planner step {index} must be a JSON object.")

      allowed_fields = {
        "description",
        "completion_criteria",
      }

      unknown_fields = set(raw_step) - allowed_fields

      if unknown_fields:
        raise PlannerResponseError(f"Planner step {index} contains unsupported fields: {', '.join(sorted(unknown_fields))}")

      description = raw_step.get("description")
      completion_criteria = raw_step.get("completion_criteria")

      if not isinstance(description, str):
        raise PlannerResponseError(f"Planner step {index} field 'description' must be a string.")

      if (
        completion_criteria is not None
        and not isinstance(completion_criteria, str)
      ):
        raise PlannerResponseError(f"Planner step {index} field 'completion_criteria' must be a string or null.")

      try:
        step_specs.append(
          PlanStepSpec(
            description=description,
            completion_criteria=completion_criteria,
          )
        )
      except PlanValidationError as error:
        raise PlannerResponseError(f"Planner step {index} is invalid: {error}") from error

    return step_specs