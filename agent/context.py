from typing import Any, TypeVar

from agent.state import AgentState
from agent.task import TaskProfile

TaskStateT = TypeVar("TaskStateT")


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


def _build_plan_section(
    state: AgentState[Any],
) -> str | None:
    plan = state.plan

    if plan is None:
        return None

    lines = [
        f"Goal: {plan.goal}",
        f"Plan status: {plan.status}",
    ]

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
            lines.append(f"   Result: {step.result}")

        if step.error:
            lines.append(f"   Error: {step.error}")

    return "\n".join(lines)


def build_state_context(
    state: AgentState[TaskStateT],
    task: TaskProfile[TaskStateT],
) -> str:
    sections: list[str] = []

    agent_section = _build_agent_section(state)
    sections.append("Agent execution state:\n" + agent_section)

    plan_section = _build_plan_section(state)

    if plan_section is not None:
        sections.append("Task plan:\n" + plan_section)

    task_section = task.build_context(state.task_state)

    if task_section.strip():
        sections.append(f"Task-specific state ({task.name}):\n" + task_section)

    plan = state.plan

    if plan is not None and plan.status == "completed":
        next_action_instruction = (
            "The task plan is complete. Produce the final "
            "answer using the gathered evidence. Do not call "
            "additional tools unless they are strictly necessary."
        )

    else:
        next_action_instruction = (
            "Use the current execution state, task state, "
            "observations, and plan to choose the next action. "
            "Do not repeat completed work unless necessary."
        )

    return (
        "Current agent state:\n\n"
        + "\n\n".join(sections)
        + "\n\n"
        + next_action_instruction
    )
