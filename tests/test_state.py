from agent.plan import AgentPlan, PlanStepSpec
from agent.state import AgentState, RepositoryState


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


def test_agent_state_has_no_plan_by_default() -> None:
    state = AgentState(
        task_state=RepositoryState(),
    )

    assert state.plan is None
    assert state.to_dict()["plan"] is None


def test_agent_state_serializes_nested_plan_and_task_state() -> None:
    plan = make_test_plan("Analyze the repository architecture.")
    plan.start()

    state = AgentState(
        status="running",
        plan=plan,
        task_state=RepositoryState(
            repo_url="https://github.com/example/demo",
        ),
    )

    data = state.to_dict()

    assert data["status"] == "running"
    assert data["errors"] == []

    task_data = data["task_state"]
    assert task_data["repo_url"] == ("https://github.com/example/demo")
    assert "plan" not in task_data
    assert "errors" not in task_data

    plan_data = data["plan"]
    assert plan_data is not None
    assert plan_data["goal"] == ("Analyze the repository architecture.")
    assert plan_data["status"] == "in_progress"
    assert plan_data["current_step_index"] == 0


def test_repository_state_only_contains_task_data() -> None:
    state = RepositoryState(
        repo_url="https://github.com/example/demo",
        phase="reading_code",
    )

    data = state.to_dict()

    assert data["repo_url"] == ("https://github.com/example/demo")
    assert data["phase"] == "reading_code"
    assert "status" not in data
    assert "plan" not in data
    assert "errors" not in data
