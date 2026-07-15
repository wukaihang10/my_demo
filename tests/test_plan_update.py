import pytest

from agent.plan import AgentPlan, PlanStepSpec
from agent.plan_update import (
    PlanController,
    PlanUpdate,
    PlanUpdateBudgetError,
    PlanUpdatePolicy,
    PlanUpdateValidationError,
)


def make_running_plan() -> AgentPlan:
    plan = AgentPlan.from_step_specs(
        goal="Complete the task.",
        step_specs=[
            PlanStepSpec(
                description="Collect relevant information.",
                completion_criteria="The necessary information has been collected.",
            ),
            PlanStepSpec(
                description="Produce the final answer.",
                completion_criteria="The answer addresses the user's request.",
            ),
        ],
        max_steps=6,
    )

    plan.start()
    return plan


def test_parse_complete_current_step_update() -> None:
    update = PlanUpdate.from_dict(
        {
            "action": "complete_current_step",
            "result": "Relevant information was collected.",
        }
    )

    assert update.action == "complete_current_step"
    assert update.result == "Relevant information was collected."
    assert update.new_steps == ()


def test_plan_update_rejects_runtime_fields() -> None:
    with pytest.raises(
        PlanUpdateValidationError,
        match="unsupported fields",
    ):
        PlanUpdate.from_dict(
            {
                "action": "complete_current_step",
                "status": "completed",
                "current_step_index": 10,
            }
        )


def test_controller_completes_current_step() -> None:
    plan = make_running_plan()
    controller = PlanController()

    update = PlanUpdate.from_dict(
        {
            "action": "complete_current_step",
            "result": "Information collected.",
        }
    )

    result = controller.apply_update(
        plan,
        update,
    )

    assert result["success"] is True
    assert result["action"] == "complete_current_step"

    assert plan.steps[0].status == "completed"
    assert plan.steps[0].result == "Information collected."

    assert plan.current_step_index == 1
    assert plan.steps[1].status == "in_progress"

    assert controller.updates_used == 1


def test_controller_appends_steps_with_program_generated_ids() -> None:
    plan = make_running_plan()

    controller = PlanController(
        PlanUpdatePolicy(
            max_updates=5,
            max_total_steps=5,
            max_added_steps_per_update=2,
        )
    )

    update = PlanUpdate.from_dict(
        {
            "action": "append_steps",
            "steps": [
                {
                    "description": "Verify the final result.",
                    "completion_criteria": (
                        "The result has been checked for correctness."
                    ),
                }
            ],
        }
    )

    result = controller.apply_update(
        plan,
        update,
    )

    assert len(plan.steps) == 3
    assert plan.steps[2].id == 3
    assert plan.steps[2].status == "pending"

    assert result["added_step_ids"] == [3]

    # The current running step is not changed by append_steps.
    assert plan.current_step_index == 0
    assert plan.current_step.id == 1


def test_controller_rejects_too_many_added_steps() -> None:
    plan = make_running_plan()

    controller = PlanController(
        PlanUpdatePolicy(
            max_updates=5,
            max_total_steps=6,
            max_added_steps_per_update=1,
        )
    )

    update = PlanUpdate.from_dict(
        {
            "action": "append_steps",
            "steps": [
                {"description": "Additional step one."},
                {"description": "Additional step two."},
            ],
        }
    )

    with pytest.raises(
        PlanUpdateBudgetError,
        match="at most 1 steps",
    ):
        controller.apply_update(
            plan,
            update,
        )

    assert len(plan.steps) == 2
    assert controller.updates_used == 0


def test_controller_rejects_total_step_budget_overflow() -> None:
    plan = make_running_plan()

    controller = PlanController(
        PlanUpdatePolicy(
            max_updates=5,
            max_total_steps=2,
            max_added_steps_per_update=2,
        )
    )

    update = PlanUpdate.from_dict(
        {
            "action": "append_steps",
            "steps": [{"description": "One additional step."}],
        }
    )

    with pytest.raises(
        PlanUpdateBudgetError,
        match="at most 2 steps",
    ):
        controller.apply_update(
            plan,
            update,
        )

    assert len(plan.steps) == 2


def test_controller_enforces_update_count_budget() -> None:
    plan = make_running_plan()

    controller = PlanController(
        PlanUpdatePolicy(
            max_updates=1,
            max_total_steps=5,
            max_added_steps_per_update=2,
        )
    )

    first_update = PlanUpdate.from_dict(
        {
            "action": "complete_current_step",
            "result": "First step completed.",
        }
    )

    controller.apply_update(
        plan,
        first_update,
    )

    second_update = PlanUpdate.from_dict(
        {
            "action": "skip_current_step",
            "reason": "No longer required.",
        }
    )

    with pytest.raises(
        PlanUpdateBudgetError,
        match="budget has been exhausted",
    ):
        controller.apply_update(
            plan,
            second_update,
        )

    assert controller.updates_used == 1
    assert plan.steps[1].status == "in_progress"


def test_controller_can_fail_current_step() -> None:
    plan = make_running_plan()
    controller = PlanController()

    update = PlanUpdate.from_dict(
        {
            "action": "fail_current_step",
            "error": "Required information is unavailable.",
        }
    )

    result = controller.apply_update(
        plan,
        update,
    )

    assert result["success"] is True
    assert plan.status == "failed"
    assert plan.current_step.status == "failed"
    assert plan.error == "Required information is unavailable."


def test_plan_update_rejects_empty_failure_error() -> None:
    with pytest.raises(
        PlanUpdateValidationError,
        match="must not be empty",
    ):
        PlanUpdate.from_dict(
            {
                "action": "fail_current_step",
                "error": "   ",
            }
        )


def test_plan_update_rejects_runtime_fields_in_new_step() -> None:
    with pytest.raises(
        PlanUpdateValidationError,
        match="unsupported fields",
    ):
        PlanUpdate.from_dict(
            {
                "action": "append_steps",
                "steps": [
                    {
                        "id": 100,
                        "description": "Verify the result.",
                        "status": "completed",
                    }
                ],
            }
        )


def test_parse_keep_current_step_update() -> None:
    update = PlanUpdate.from_dict(
        {
            "action": "keep_current_step",
            "reason": ("More repository files must be inspected."),
        }
    )

    assert update.action == "keep_current_step"
    assert update.reason == ("More repository files must be inspected.")


def test_controller_keeps_current_step_without_using_budget() -> None:
    plan = make_running_plan()
    controller = PlanController()

    update = PlanUpdate.from_dict(
        {
            "action": "keep_current_step",
            "reason": ("The available evidence is not sufficient."),
        }
    )

    result = controller.apply_update(
        plan,
        update,
    )

    assert result["success"] is True
    assert result["action"] == "keep_current_step"
    assert result["plan_changed"] is False

    assert plan.status == "in_progress"
    assert plan.current_step_index == 0
    assert plan.current_step is not None
    assert plan.current_step.status == "in_progress"

    assert controller.updates_used == 0
    assert controller.updates_remaining == controller.policy.max_updates
