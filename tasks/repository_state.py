from dataclasses import dataclass, field
from typing import Any


@dataclass
class RepositoryState:
    repo_url: str | None = None
    repo_path: str | None = None
    phase: str = "initial"

    repository_summary: dict[str, Any] | None = None
    listed_files: list[str] = field(default_factory=list)
    read_files: list[str] = field(default_factory=list)
    searched_keywords: list[str] = field(default_factory=list)
    important_files: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)

    rag_indexed: bool = False
    rag_search_queries: list[str] = field(default_factory=list)

    def add_list_file(
        self,
        file_path: str,
    ) -> None:
        if file_path not in self.listed_files:
            self.listed_files.append(file_path)

    def add_read_file(
        self,
        file_path: str,
    ) -> None:
        if file_path not in self.read_files:
            self.read_files.append(file_path)

    def add_search_keyword(
        self,
        keyword: str,
    ) -> None:
        if keyword not in self.searched_keywords:
            self.searched_keywords.append(keyword)

    def add_rag_search_query(
        self,
        query: str,
    ) -> None:
        if query not in self.rag_search_queries:
            self.rag_search_queries.append(query)
