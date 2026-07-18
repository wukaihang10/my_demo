from agent.context import build_state_context
from agent.plan import AgentPlan, PlanStepSpec
from agent.state import AgentState, RepositoryState


def make_test_plan(goal: str) -> AgentPlan:
    return AgentPlan.from_step_specs(
        goal=goal,
        step_specs=[
            PlanStepSpec(
                description="Prepare the repository for inspection.",
                completion_criteria=("A valid local path to the target repository."),
            ),
            PlanStepSpec(
                description=("Collect a high-level overview of the repository."),
                completion_criteria=(
                    "The repository structure and important files are known."
                ),
            ),
        ],
    )


def test_context_includes_repository_state() -> None:
    state = AgentState(
        status="running",
        errors=[
            "A previous file was not found.",
        ],
        task_state=RepositoryState(
            repo_url="https://github.com/example/demo",
            repo_path="workspace/demo",
            phase="reading_code",
            important_files=[
                "README.md",
                "agent/agent.py",
            ],
            read_files=[
                "README.md",
            ],
            searched_keywords=[
                "execute_tool",
            ],
            findings=[
                "Agent.run controls the execution loop.",
            ],
        ),
    )

    context = build_state_context(state)

    assert "Current phase: reading_code" in context
    assert "Repository URL: https://github.com/example/demo" in context

    assert "Repository path: workspace/demo" in context
    assert "Important files discovered:" in context
    assert "- agent/agent.py" in context
    assert "Files already inspected:" in context
    assert "- README.md" in context
    assert "Keywords already searched:" in context
    assert "- execute_tool" in context
    assert "Findings gathered:" in context
    assert "- Agent.run controls the execution loop." in context
    assert "Previous errors:" in context

    assert "Run status: running" in context


def test_context_includes_active_plan() -> None:
    plan = make_test_plan("Analyze the repository architecture.")
    plan.start()

    state = AgentState(
        status="running",
        plan=plan,
        task_state=RepositoryState(
            repo_url="https://github.com/example/demo",
        ),
    )

    context = build_state_context(state)

    assert "Task plan:" in context
    assert "Goal: Analyze the repository architecture." in context
    assert "Plan status: in_progress" in context

    assert "Current step: 1. Prepare the repository for inspection." in context

    assert "Current step status: in_progress" in context

    assert (
        "Completion criteria: A valid local path to the target repository." in context
    )

    assert "-> [in_progress] 1. Prepare the repository for inspection." in context

    assert "[pending] 2. Collect a high-level overview of the repository." in context


def test_context_includes_plan_progress() -> None:
    plan = make_test_plan("Analyze the repository architecture.")

    plan.start()

    plan.complete_current_step("Repository is available at workspace/demo.")

    state = AgentState(
        task_state=RepositoryState(),
        plan=plan,
    )

    context = build_state_context(state)

    assert "[completed] 1. Prepare the repository for inspection." in context

    assert "Result: Repository is available at workspace/demo." in context

    assert (
        "-> [in_progress] 2. Collect a high-level overview of the repository."
        in context
    )

    assert (
        "Current step: 2. Collect a high-level overview of the repository." in context
    )


def test_context_requests_final_answer_when_plan_completed() -> None:
    plan = make_test_plan("Complete the task.")
    plan.start()

    while plan.status == "in_progress":
        plan.complete_current_step("Step completed.")

    state = AgentState(
        task_state=RepositoryState(),
        plan=plan,
    )

    context = build_state_context(state)

    assert "Plan status: completed" in context
    assert "Current step: none" in context
    assert "The task plan is complete. Produce the final " "answer" in context
