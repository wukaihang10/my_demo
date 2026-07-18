from typing import Any, TypeVar

from agent.state import AgentState
from agent.task import TaskProfile
from agent.config import PlanningMode

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
    planning_mode: PlanningMode,
) -> str | None:
    plan = state.plan

    if plan is None:
        return None

    if planning_mode is PlanningMode.STATIC:
        lines = [
            "Planning mode: static",
            f"Goal: {plan.goal}",
            (
                "This plan is an advisory roadmap. "
                "Step statuses are not updated during execution."
            ),
            "Roadmap:",
        ]

        for step in plan.steps:
            lines.append(f"- {step.id}: {step.description}")

            if step.completion_criteria:
                lines.append("  Completion criteria: " f"{step.completion_criteria}")

        return "\n".join(lines)

    lines = [
        "Plannig mode: dynamic",
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
    planning_mode: PlanningMode,
) -> str:
    sections: list[str] = []

    agent_section = _build_agent_section(state)
    sections.append("Agent execution state:\n" + agent_section)

    plan_section = _build_plan_section(state, planning_mode)

    if plan_section is not None:
        sections.append("Task plan:\n" + plan_section)

    task_section = task.build_context(state.task_state)

    if task_section.strip():
        sections.append(f"Task-specific state ({task.name}):\n" + task_section)

    plan = state.plan

    if planning_mode is PlanningMode.NONE:
        next_action_instruction = (
            "Choose the next action directly from the user request, "
            "task state, and observations. Use tools only when needed."
        )

    elif planning_mode is PlanningMode.STATIC:
        next_action_instruction = (
            "Use the static plan as an advisory roadmap. "
            "Its step statuses are not updated during execution. "
            "Choose the next useful action from the available evidence "
            "and avoid repeating completed work."
        )

    elif plan is not None and plan.status == "completed":
        next_action_instruction = (
            "The dynamic task plan is complete. Produce the final "
            "answer using the gathered evidence. Do not call additional "
            "tools unless they are strictly necessary."
        )

    else:
        next_action_instruction = (
            "Follow the current dynamic plan step. Use the execution "
            "state, task state, and observations to choose the next "
            "action. Avoid repeating completed work."
        )

    return (
        "Current agent state:\n\n"
        + "\n\n".join(sections)
        + "\n\n"
        + next_action_instruction
    )
