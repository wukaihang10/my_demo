from types import SimpleNamespace

import pytest

from agent.planner import (
    LLMPlanner,
    PlannerResponseError,
)


def make_response(content: str):
    return SimpleNamespace(
        content=content,
    )


def test_create_plan_from_valid_response() -> None:
    def fake_chat(messages, tools):
        assert tools == []

        return make_response("""
      {
        "steps": [
          {
            "description": "Inspect the available information.",
            "completion_criteria": "Relevant evidence has been collected."
          },
          {
            "description": "Produce the final answer.",
            "completion_criteria": "The answer addresses the user's goal."
          }
        ]
      }
      """)

    planner = LLMPlanner(
        max_plan_steps=4,
        chat_function=fake_chat,
    )

    plan = planner.create_plan("Explain the target project.")

    assert plan.goal == "Explain the target project."
    assert plan.status == "pending"
    assert len(plan.steps) == 2

    assert plan.steps[0].id == 1
    assert plan.steps[1].id == 2

    assert plan.steps[0].status == "pending"
    assert plan.steps[0].description == "Inspect the available information."


def test_planner_rejects_runtime_fields() -> None:
    def fake_chat(messages, tools):
        return make_response("""
      {
        "steps": [
          {
            "id": 100,
            "description": "Do something.",
            "status": "completed"
          }
        ]
      }
      """)

    planner = LLMPlanner(
        chat_function=fake_chat,
    )

    with pytest.raises(
        PlannerResponseError,
        match="unsupported fields",
    ):
        planner.create_plan("Complete a task.")


def test_planner_rejects_too_many_steps() -> None:
    def fake_chat(messages, tools):
        return make_response("""
      {
        "steps": [
          {"description": "Step one."},
          {"description": "Step two."},
          {"description": "Step three."}
        ]
      }
      """)

    planner = LLMPlanner(
        max_plan_steps=2,
        chat_function=fake_chat,
    )

    with pytest.raises(
        PlannerResponseError,
        match="limit is 2",
    ):
        planner.create_plan("Complete a task.")


def test_planner_rejects_invalid_json() -> None:
    def fake_chat(messages, tools):
        return make_response("This is not JSON.")

    planner = LLMPlanner(
        chat_function=fake_chat,
    )

    with pytest.raises(
        PlannerResponseError,
        match="invalid JSON",
    ):
        planner.create_plan("Complete a task.")
