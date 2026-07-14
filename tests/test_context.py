from agent.context import build_state_context
from agent.plan import build_repository_analysis_plan
from agent.state import RepositoryState


def test_context_includes_repository_state() -> None:
  state = RepositoryState(
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
    errors=[
      "A previous file was not found.",
    ],
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


def test_context_includes_active_plan() -> None:
  plan = build_repository_analysis_plan(
    "Analyze the repository architecture."
  )
  plan.start()

  state = RepositoryState(
    repo_url="https://github.com/example/demo",
    plan=plan,
  )

  context = build_state_context(state)

  assert "Task plan:" in context
  assert "Goal: Analyze the repository architecture." in context
  assert "Plan status: in_progress" in context

  assert "Current step: 1. Prepare the repository for inspection." in context

  assert "Current step status: in_progress" in context

  assert "Expected evidence: A valid local path to the target repository." in context

  assert "-> [in_progress] 1. Prepare the repository for inspection." in context

  assert "[pending] 2. Collect a high-level overview of the repository." in context


def test_context_includes_plan_progress() -> None:
  plan = build_repository_analysis_plan("Analyze the repository architecture.")

  plan.start()

  plan.complete_current_step("Repository is available at workspace/demo.")

  state = RepositoryState(plan=plan)

  context = build_state_context(state)

  assert "[completed] 1. Prepare the repository for inspection." in context

  assert "Result: Repository is available at workspace/demo." in context

  assert "-> [in_progress] 2. Collect a high-level overview of the repository." in context

  assert "Current step: 2. Collect a high-level overview of the repository." in context


def test_context_works_without_plan() -> None:
  state = RepositoryState()

  context = build_state_context(state)

  assert "Task plan:" not in context
  assert "Repository analysis state:" in context
  assert "Current phase: initial" in context