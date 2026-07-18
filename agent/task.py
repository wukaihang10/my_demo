from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from tools.base import Tool

TaskStateT = TypeVar("TaskStateT")


TaskStateFactory = Callable[
    [dict[str, Any]],
    TaskStateT,
]

TaskStateReducer = Callable[
    [
        TaskStateT,
        str,
        dict[str, Any],
        dict[str, Any],
    ],
    None,
]

TaskContextBuilder = Callable[
    [TaskStateT],
    str,
]


@dataclass(frozen=True)
class TaskProfile(Generic[TaskStateT]):
    """Defines the task-specific parts used by Agent."""

    name: str
    system_prompt: str
    tools: tuple[Tool, ...]

    create_state: TaskStateFactory[TaskStateT]
    reduce_tool_result: TaskStateReducer[TaskStateT]
    build_context: TaskContextBuilder[TaskStateT]
