from agent.plan import AgentPlan, PlanStepSpec
from agent.state import RepositoryState


def make_test_plan(
    goal: str = "Complete the task.",
) -> AgentPlan:
    return AgentPlan.from_step_specs(
        goal=goal,
        step_specs=[
            PlanStepSpec(
                description="Collect relevant information.",
                completion_criteria=("The necessary information has been collected."),
            ),
            PlanStepSpec(
                description="Produce the final answer.",
                completion_criteria=("The final answer addresses the user's goal."),
            ),
        ],
    )


def test_repository_state_has_no_plan_by_default() -> None:
    state = RepositoryState()

    assert state.plan is None
    assert state.to_dict()["plan"] is None


def test_repository_state_serializes_nested_plan() -> None:
    plan = make_test_plan("Analyze the repository architecture.")
    plan.start()

    state = RepositoryState(
        repo_url="https://github.com/example/demo",
        plan=plan,
    )

    data = state.to_dict()

    assert data["repo_url"] == ("https://github.com/example/demo")

    plan_data = data["plan"]

    assert plan_data is not None
    assert plan_data["goal"] == ("Analyze the repository architecture.")
    assert plan_data["status"] == "in_progress"
    assert plan_data["current_step_index"] == 0

    steps = plan_data["steps"]

    assert len(steps) == 2
    assert steps[0]["id"] == 1
    assert steps[0]["status"] == "in_progress"
    assert steps[1]["id"] == 2
    assert steps[1]["status"] == "pending"


def test_repository_state_serializes_plan_progress() -> None:
    plan = make_test_plan("Analyze the repository architecture.")
    plan.start()
    plan.complete_current_step("Relevant repository information was collected.")

    state = RepositoryState(plan=plan)

    data = state.to_dict()
    plan_data = data["plan"]

    assert plan_data is not None
    assert plan_data["current_step_index"] == 1
    assert plan_data["status"] == "in_progress"

    first_step = plan_data["steps"][0]
    second_step = plan_data["steps"][1]

    assert first_step["status"] == "completed"
    assert first_step["result"] == ("Relevant repository information was collected.")

    assert second_step["status"] == "in_progress"
