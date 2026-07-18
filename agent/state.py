from dataclasses import asdict, dataclass, field
from typing import Any, Generic, TypeVar

from agent.plan import AgentPlan

TaskStateT = TypeVar("TaskStateT")


@dataclass
class RepositoryState:
    """Repository-analysis-specific state."""

    repo_url: str | None = None
    repo_path: str | None = None

    # Repository task progress, not the whole agent run status.
    phase: str = "initial"

    repository_summary: dict[str, Any] | None = None
    listed_files: list[str] = field(default_factory=list)
    read_files: list[str] = field(default_factory=list)
    searched_keywords: list[str] = field(default_factory=list)
    important_files: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)

    def add_list_file(self, file_path: str) -> None:
        if file_path not in self.listed_files:
            self.listed_files.append(file_path)

    def add_read_file(self, file_path: str) -> None:
        if file_path not in self.read_files:
            self.read_files.append(file_path)

    def add_search_keyword(self, keyword: str) -> None:
        if keyword not in self.searched_keywords:
            self.searched_keywords.append(keyword)

    def add_finding(self, finding: str) -> None:
        if finding not in self.findings:
            self.findings.append(finding)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentState(Generic[TaskStateT]):
    """Task-independent state owned by the agent runtime."""

    task_state: TaskStateT
    status: str = "initial"
    plan: AgentPlan | None = None
    errors: list[str] = field(default_factory=list)

    def add_error(self, error: str) -> None:
        self.errors.append(error)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
