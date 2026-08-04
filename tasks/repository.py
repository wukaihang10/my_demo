from typing import Any

from tasks.repository_state import RepositoryState
from agent.task import TaskProfile
from rag.repository_manager import RepositoryKnowledgeManager
from tools.base import Tool
from tools.registry import create_repository_tool_map

REPOSITORY_SYSTEM_PROMPT = """
You are a GitHub repository analysis agent.

Use the available tools to inspect repositories and answer questions using evidence from actual repository files.

Rules:

1. Do not guess repository details.
2. Clone a repository when needed.
3. Use summarize_repository to obtain a high-level overview.
4. Inspect the repository structure before selecting files.
5. Read the README when it exists.
6. After cloning a Python repository, use index_repository_knowledge before repository knowledge search.
7. Use search_repository_knowledge for natural-language questions, architecture, behavior, data flow, implementation logic, and queries that combine concepts with code identifiers. This tool uses vector and BM25 hybrid retrieval.
8. Use search_code when you need exhaustive exact matches for a precise identifier, literal string, error code, or configuration key.
9. Use read_file when retrieved snippets are incomplete or when  surrounding code is required to confirm behavior.
10. Do not treat semantic search results as complete files.
11. Avoid reading every file.
12. If a tool fails, inspect the error and try a reasonable alternative.
13. Only give the final answer after gathering enough evidence.

Requirements:

- Do not guess.
- Do not claim anything unless repository evidence supports it.
- Response in Chinese.
- Clearly distinguish confirmed facts from reasonable inferences.
- If a tool fails, inspect the error and try a reasonable alternative.
""".strip()

REPOSITORY_TOOL_NAMES = (
    "clone_repository",
    "summarize_repository",
    "list_files",
    "read_file",
    "search_code",
    "index_repository_knowledge",
    "search_repository_knowledge",
)


def create_repository_state(
    input_data: dict[str, Any],
) -> RepositoryState:
    raw_repo_url = input_data.get("repo_url")

    repo_url: str | None = None

    if raw_repo_url is not None:
        repo_url = str(raw_repo_url).strip()

        if not repo_url:
            repo_url = None

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

    elif tool_name == "index_repository_knowledge":
        repository_path = result.get("repository_path")

        if repository_path:
            state.repo_path = str(repository_path)

        state.rag_indexed = True
        state.phase = "reading_code"

    elif tool_name == "search_repository_knowledge":
        query = result.get("query") or arguments.get("query")

        if query:
            state.add_rag_search_query(str(query))

        state.phase = "reading_code"


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

    if state.rag_indexed:
        lines.append("Repository semantic index: ready")

    if state.rag_search_queries:
        lines.append("Semantic repository queries " "already performed:")
        lines.extend(f"- {query}" for query in state.rag_search_queries)

    if state.findings:
        lines.append("Findings gathered:")
        lines.extend(f"- {finding}" for finding in state.findings)

    return "\n".join(lines)


def _build_repository_tools(
    repository_manager: RepositoryKnowledgeManager,
) -> tuple[Tool, ...]:
    tool_map = create_repository_tool_map(repository_manager)

    missing_tools = [
        tool_name for tool_name in REPOSITORY_TOOL_NAMES if tool_name not in tool_map
    ]

    if missing_tools:
        missing_text = ", ".join(missing_tools)

        raise RuntimeError(
            "Repository task references unregistered tools: " f"{missing_text}"
        )

    return tuple(tool_map[tool_name] for tool_name in REPOSITORY_TOOL_NAMES)


def create_repository_task(
    repository_manager: RepositoryKnowledgeManager | None = None,
) -> TaskProfile[RepositoryState]:
    """
    Create an isolated repository task for one Agent.

    When no manager is supplied, a fresh manager is created. Its bound methods
    are retained by the task's Tool objects, so the owning Agent keeps the
    manager alive for as long as it keeps the task.
    """

    manager = repository_manager or RepositoryKnowledgeManager()

    return TaskProfile[RepositoryState](
        name="repository_analysis",
        system_prompt=REPOSITORY_SYSTEM_PROMPT,
        tools=_build_repository_tools(manager),
        create_state=create_repository_state,
        reduce_tool_result=reduce_repository_tool_result,
        build_context=build_repository_context,
    )
