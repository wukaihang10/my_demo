import pytest

from agent.plan import (
    AgentPlan,
    PlanStepSpec,
    PlanValidationError,
)


def make_plan(goal: str = "Complete the task.") -> AgentPlan:
    return AgentPlan.from_step_specs(
        goal=goal,
        step_specs=[
            PlanStepSpec(
                description="Collect the required information.",
                completion_criteria="Enough relevant information has been collected.",
            ),
            PlanStepSpec(
                description="Produce the final answer.",
                completion_criteria="The final answer addresses the user's goal.",
            ),
        ],
    )


def test_plan_finish_skips_unfinished_steps() -> None:
    plan = make_plan()

    plan.start()
    plan.finish("Task completed.")

    assert plan.status == "completed"
    assert plan.current_step is None
    assert plan.current_step_index == len(plan.steps)
    assert plan.result == "Task completed."
    assert all(step.status == "skipped" for step in plan.steps)


def test_plan_finish_preserves_completed_steps() -> None:
    plan = make_plan()

    plan.start()
    plan.complete_current_step("Information collected.")
    plan.finish("Task completed.")

    assert plan.steps[0].status == "completed"
    assert plan.steps[0].result == "Information collected."
    assert plan.steps[1].status == "skipped"
    assert plan.status == "completed"


def test_plan_fail_marks_current_step_failed() -> None:
    plan = make_plan()

    plan.start()
    plan.fail("Execution failed.")

    assert plan.status == "failed"
    assert plan.current_step is not None
    assert plan.current_step.status == "failed"
    assert plan.current_step.error == "Execution failed."


def test_plan_fail_is_idempotent() -> None:
    plan = make_plan()

    plan.start()
    plan.fail("First failure.")
    plan.fail("Second failure.")

    assert plan.status == "failed"
    assert plan.current_step is not None
    assert plan.current_step.error == "First failure."


def test_append_step_specs_generates_runtime_ids() -> None:
    plan = make_plan()

    plan.start()

    new_steps = plan.append_step_specs(
        [
            PlanStepSpec(
                description="Verify the result.",
                completion_criteria="The result has been checked.",
            )
        ],
        max_steps=3,
    )

    assert len(new_steps) == 1
    assert new_steps[0].id == 3
    assert new_steps[0].status == "pending"
    assert plan.steps[-1] is new_steps[0]


def test_append_step_specs_respects_max_steps() -> None:
    plan = make_plan()

    plan.start()

    with pytest.raises(PlanValidationError):
        plan.append_step_specs(
            [
                PlanStepSpec(
                    description="Verify the result.",
                )
            ],
            max_steps=2,
        )


def test_finish_records_result_for_completed_plan() -> None:
    plan = make_plan()

    plan.start()
    plan.complete_current_step("Information collected.")
    plan.complete_current_step("Answer produced.")

    assert plan.status == "completed"
    assert plan.result is None

    plan.finish("Final answer.")

    assert plan.status == "completed"
    assert plan.result == "Final answer."
