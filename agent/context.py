from typing import Any

from agent.state import AgentState, RepositoryState


def _format_summary_value(value: Any) -> str:
    if isinstance(value, dict):
        return ",".join(f"{key}={item}" for key, item in value.items())

    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in value)

    return str(value)


def _build_plan_section(
    state: AgentState[Any],
) -> str | None:
    plan = state.plan

    if plan is None:
        return None

    lines = [f"Goal: {plan.goal}", f"Plan status: {plan.status}"]

    if plan.result:
        lines.append(f"Plan result: {plan.result}")

    if plan.error:
        lines.append(f"Plan error: {plan.error}")

    current_step = plan.current_step

    if current_step is None:
        lines.append("Current step: none")
    else:
        lines.extend(
            [
                (f"Current step: {current_step.id}. " f"{current_step.description}"),
                ("Current step status: " f"{current_step.status}"),
            ]
        )

        if current_step.completion_criteria:
            lines.append("Completion criteria: " f"{current_step.completion_criteria}")

    lines.append("Plan steps:")

    for index, step in enumerate(plan.steps):
        marker = "->" if index == plan.current_step_index else "  "

        lines.append(f"{marker} [{step.status}] " f"{step.id}. {step.description}")

        if step.result:
            lines.append(f"  Result: {step.result}")

        if step.error:
            lines.append(f"  Error: {step.error}")

    return "\n".join(lines)


def _build_agent_section(
    state: AgentState[Any],
) -> str:
    lines = [
        f"Run status: {state.status}",
    ]

    if state.errors:
        lines.append("Previous errors:")
        lines.extend(f"- {error}" for error in state.errors)

    return "\n".join(lines)


def _build_repository_section(
    state: RepositoryState,
) -> str:
    lines = [f"Current phase: {state.phase}"]

    if state.repo_url:
        lines.append(f"Repository URL: {state.repo_url}")

    if state.repo_path:
        lines.append(f"Repository path: {state.repo_path}")

    if state.repository_summary:
        lines.append("Repository summary:")

        for key, value in state.repository_summary.items():
            formatted_value = _format_summary_value(value)

            lines.append(f"- {key}: {formatted_value}")

    if state.listed_files:
        lines.append(f"Number of files already listed:" f"{len(state.listed_files)}")

    if state.important_files:
        lines.append("Important files discovered:")
        lines.extend(f"- {file_path}" for file_path in state.important_files)

    if state.read_files:
        lines.append("Files already inspected:")
        lines.extend(f"- {file_path}" for file_path in state.read_files)

    if state.searched_keywords:
        lines.append("Keywords already searched:")
        lines.extend(f"- {keyword}" for keyword in state.searched_keywords)

    if state.findings:
        lines.append("Findings gathered:")
        lines.extend(f"- {finding}" for finding in state.findings)

    return "\n".join(lines)


def build_state_context(
    state: AgentState[RepositoryState],
) -> str:
    sections: list[str] = []

    agent_section = _build_agent_section(state)
    sections.append(f"Agent execution state:\n{agent_section}")

    plan_section = _build_plan_section(state)

    if plan_section is not None:
        sections.append(f"Task plan:\n{plan_section}")

    repository_section = _build_repository_section(state.task_state)
    sections.append("Repository analysis state:\n" + repository_section)

    plan = state.plan

    if plan is not None and plan.status == "completed":
        next_action_instruction = (
            "The task plan is complete. Produce the final "
            "answer using the gathered evidence. Do not call "
            "additional tools unless they are strictly necessary."
        )
    else:
        next_action_instruction = (
            "Use this state to choose the next action. "
            "Do not repeat completed work unless it is "
            "necessary. Focus on the current plan step."
        )

    return (
        "Current agent state:\n\n"
        + "\n\n".join(sections)
        + "\n\n"
        + next_action_instruction
    )
