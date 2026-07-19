from typing import Any

from tasks.repository_state import RepositoryState
from agent.task import TaskProfile
from tools.base import Tool
from tools.registry import TOOL_MAP

REPOSITORY_SYSTEM_PROMPT = """
You are a GitHub repository analysis agent.

Use the available tools to inspect repositories and answer questions
using evidence from actual repository files.

Rules:
1. Do not guess repository details.
2. Clone a repository when needed.
3. Use summarize_repository to obtain a high-level overview.
4. Inspect the repository structure before selecting files.
5. Read the README when it exists.
6. Read relevant source and configuration files before explaining code.
7. Use search_code when you do not know where something is implemented.
8. Avoid reading every file.
9. If a tool fails, inspect the error and try a reasonable alternative.
10. Only give the final answer after gathering enough evidence.
""".strip()

REPOSITORY_TOOL_NAMES = (
    "clone_repository",
    "summarize_repository",
    "list_files",
    "read_file",
    "search_code",
)


def create_repository_state(
    input_data: dict[str, Any],
) -> RepositoryState:
    repo_url = input_data.get("repo_url")

    if repo_url is not None:
        repo_url = str(repo_url)

    return RepositoryState(
        repo_url=repo_url,
    )


def reduce_repository_tool_result(
    state: RepositoryState,
    tool_name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Update repository task state from one tool result."""

    if result.get("success") is not True:
        return

    if tool_name == "clone_repository":
        repo_path = result.get("repo_path")

        if repo_path:
            state.repo_path = str(repo_path)

        state.phase = "understanding"

    elif tool_name == "summarize_repository":
        important_files = result.get(
            "important_files",
            [],
        )

        state.important_files = list(dict.fromkeys(important_files))

        summary_fields = (
            "repo_name",
            "total_files",
            "top_level_structure",
            "languages",
            "extensions",
            "readme_path",
        )

        state.repository_summary = {
            field: result.get(field) for field in summary_fields
        }

        state.phase = "reading_code"

    elif tool_name == "list_files":
        files = result.get("files", [])

        for file_path in files:
            state.add_list_file(str(file_path))

    elif tool_name == "read_file":
        file_path = result.get("file_path") or arguments.get("file_path")

        if file_path:
            state.add_read_file(str(file_path))

    elif tool_name == "search_code":
        keyword = result.get("keyword") or arguments.get("keyword")

        if keyword:
            state.add_search_keyword(str(keyword))


def _format_summary_value(value: Any) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{key}={item}" for key, item in value.items())

    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)

    return str(value)


def build_repository_context(
    state: RepositoryState,
) -> str:
    """Build task-specific context for repository analysis."""

    lines = [
        f"Current repository phase: {state.phase}",
    ]

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
        lines.append("Number of files already listed: " f"{len(state.listed_files)}")

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


def _build_repository_tools() -> tuple[Tool, ...]:
    missing_tools = [
        tool_name for tool_name in REPOSITORY_TOOL_NAMES if tool_name not in TOOL_MAP
    ]

    if missing_tools:
        missing_text = ", ".join(missing_tools)

        raise RuntimeError(
            "Repository task references unregistered tools: " f"{missing_text}"
        )

    return tuple(TOOL_MAP[tool_name] for tool_name in REPOSITORY_TOOL_NAMES)


REPOSITORY_TOOLS = _build_repository_tools()

REPOSITORY_TASK = TaskProfile[RepositoryState](
    name="repository_analysis",
    system_prompt=REPOSITORY_SYSTEM_PROMPT,
    tools=REPOSITORY_TOOLS,
    create_state=create_repository_state,
    reduce_tool_result=reduce_repository_tool_result,
    build_context=build_repository_context,
)
