from tasks.repository import (
    REPOSITORY_TASK,
    build_repository_context,
    create_repository_state,
    reduce_repository_tool_result,
)
from tasks.repository_state import RepositoryState


def test_repository_task_exposes_only_registered_repository_tools() -> None:
    assert [tool.name for tool in REPOSITORY_TASK.tools] == [
        "clone_repository",
        "summarize_repository",
        "list_files",
        "read_file",
        "search_code",
    ]


def test_create_repository_state_normalizes_url() -> None:
    assert create_repository_state({"repo_url": "  https://example.test/repo  "}) == (
        RepositoryState(repo_url="https://example.test/repo")
    )
    assert create_repository_state({"repo_url": "   "}).repo_url is None
    assert create_repository_state({}).repo_url is None


def test_reducer_tracks_successful_repository_work_without_duplicates() -> None:
    state = RepositoryState(repo_url="https://example.test/repo")

    reduce_repository_tool_result(
        state,
        "clone_repository",
        {},
        {"success": True, "repo_path": "workspace/repo"},
    )
    reduce_repository_tool_result(
        state,
        "list_files",
        {},
        {"success": True, "files": ["README.md", "README.md", "src/app.py"]},
    )
    reduce_repository_tool_result(
        state,
        "read_file",
        {"file_path": "README.md"},
        {"success": True},
    )
    reduce_repository_tool_result(
        state,
        "search_code",
        {"keyword": "Agent"},
        {"success": True},
    )
    reduce_repository_tool_result(
        state,
        "read_file",
        {"file_path": "ignored.py"},
        {"success": False, "error": "not found"},
    )

    assert state.repo_path == "workspace/repo"
    assert state.phase == "understanding"
    assert state.listed_files == ["README.md", "src/app.py"]
    assert state.read_files == ["README.md"]
    assert state.searched_keywords == ["Agent"]


def test_summary_reducer_keeps_context_fields_and_deduplicates_important_files() -> None:
    state = RepositoryState()
    result = {
        "success": True,
        "repo_name": "demo",
        "total_files": 12,
        "top_level_structure": ["agent", "tests"],
        "languages": {"Python": 10},
        "extensions": {".py": 10},
        "readme_path": "README.md",
        "important_files": ["README.md", "agent/agent.py", "README.md"],
        "unrelated": "not persisted",
    }

    reduce_repository_tool_result(state, "summarize_repository", {}, result)

    assert state.phase == "reading_code"
    assert state.important_files == ["README.md", "agent/agent.py"]
    assert state.repository_summary == {
        "repo_name": "demo",
        "total_files": 12,
        "top_level_structure": ["agent", "tests"],
        "languages": {"Python": 10},
        "extensions": {".py": 10},
        "readme_path": "README.md",
    }


def test_repository_context_formats_collected_state() -> None:
    state = RepositoryState(
        repo_url="https://example.test/repo",
        repo_path="workspace/repo",
        phase="reading_code",
        repository_summary={
            "languages": {"Python": 10, "Markdown": 1},
            "top_level_structure": ["agent", "tests"],
        },
        listed_files=["README.md", "agent/agent.py"],
        important_files=["agent/agent.py"],
        read_files=["README.md"],
        searched_keywords=["Agent"],
        findings=["Agent.run owns the loop."],
    )

    context = build_repository_context(state)

    assert "Current repository phase: reading_code" in context
    assert "- languages: Python=10, Markdown=1" in context
    assert "- top_level_structure: agent, tests" in context
    assert "Number of files already listed: 2" in context
    assert "- Agent.run owns the loop." in context
