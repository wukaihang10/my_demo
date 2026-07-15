from types import SimpleNamespace
from typing import Any

import pytest

from agent.plan import AgentPlan, PlanStepSpec
from agent.plan_evaluator import (
    LLMPlanProgressEvaluator,
    PlanEvaluationResponseError,
)
from agent.state import RepositoryState


def make_running_plan() -> AgentPlan:
    plan = AgentPlan.from_step_specs(
        goal="Analyze the repository.",
        step_specs=[
            PlanStepSpec(
                description=("Inspect the repository structure."),
                completion_criteria=(
                    "The important files and directories " "are known."
                ),
            ),
            PlanStepSpec(
                description=("Inspect the relevant source files."),
                completion_criteria=(
                    "The important implementation details " "are understood."
                ),
            ),
        ],
    )

    plan.start()
    return plan


def make_response(content: str):
    return SimpleNamespace(content=content)


def test_evaluator_returns_complete_update() -> None:
    received_messages: list[list[dict[str, Any]]] = []

    def fake_chat(messages, tools):
        received_messages.append(messages)

        assert tools == []

        return make_response("""
            {
              "action": "complete_current_step",
              "result": "The repository structure was listed."
            }
            """)

    plan = make_running_plan()
    state = RepositoryState(
        repo_url="https://github.com/example/demo",
        plan=plan,
    )

    evaluator = LLMPlanProgressEvaluator(
        chat_function=fake_chat,
    )

    update = evaluator.evaluate_progress(
        plan=plan,
        state=state,
        latest_evidence=[
            {
                "type": "tool_result",
                "tool_name": "list_files",
                "result": {
                    "success": True,
                    "files": [
                        "README.md",
                        "agent/agent.py",
                    ],
                },
            }
        ],
        updates_remaining=10,
        max_total_steps=10,
        max_added_steps_per_update=3,
    )

    assert update.action == "complete_current_step"
    assert update.result == ("The repository structure was listed.")

    user_message = received_messages[0][1]["content"]

    assert "latest_evidence" in user_message
    assert "list_files" in user_message
    assert "Inspect the repository structure" in user_message


def test_evaluator_returns_keep_update() -> None:
    def fake_chat(messages, tools):
        return make_response("""
            {
              "action": "keep_current_step",
              "reason": "Only part of the structure is known."
            }
            """)

    plan = make_running_plan()
    state = RepositoryState(plan=plan)

    evaluator = LLMPlanProgressEvaluator(
        chat_function=fake_chat,
    )

    update = evaluator.evaluate_progress(
        plan=plan,
        state=state,
        latest_evidence=[],
        updates_remaining=10,
        max_total_steps=10,
        max_added_steps_per_update=3,
    )

    assert update.action == "keep_current_step"
    assert update.reason == ("Only part of the structure is known.")


def test_evaluator_accepts_fenced_json() -> None:
    def fake_chat(messages, tools):
        return make_response("""```json
            {
              "action": "keep_current_step",
              "reason": "More evidence is required."
            }
            ```""")

    plan = make_running_plan()
    state = RepositoryState(plan=plan)

    evaluator = LLMPlanProgressEvaluator(
        chat_function=fake_chat,
    )

    update = evaluator.evaluate_progress(
        plan=plan,
        state=state,
        latest_evidence=[],
        updates_remaining=10,
        max_total_steps=10,
        max_added_steps_per_update=3,
    )

    assert update.action == "keep_current_step"


def test_evaluator_rejects_invalid_json() -> None:
    def fake_chat(messages, tools):
        return make_response("not valid json")

    plan = make_running_plan()
    state = RepositoryState(plan=plan)

    evaluator = LLMPlanProgressEvaluator(
        chat_function=fake_chat,
    )

    with pytest.raises(
        PlanEvaluationResponseError,
        match="invalid JSON",
    ):
        evaluator.evaluate_progress(
            plan=plan,
            state=state,
            latest_evidence=[],
            updates_remaining=10,
            max_total_steps=10,
            max_added_steps_per_update=3,
        )


def test_evaluator_rejects_invalid_update() -> None:
    def fake_chat(messages, tools):
        return make_response("""
            {
              "action": "complete_current_step",
              "status": "completed"
            }
            """)

    plan = make_running_plan()
    state = RepositoryState(plan=plan)

    evaluator = LLMPlanProgressEvaluator(
        chat_function=fake_chat,
    )

    with pytest.raises(
        PlanEvaluationResponseError,
        match="invalid update",
    ):
        evaluator.evaluate_progress(
            plan=plan,
            state=state,
            latest_evidence=[],
            updates_remaining=10,
            max_total_steps=10,
            max_added_steps_per_update=3,
        )
