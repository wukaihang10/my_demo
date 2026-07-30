from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.embedding import (
    SentenceTransformerEmbeddingClient,
)
from rag.interfaces import EmbeddingClient
from rag.python_repository_rag import (
    PythonRepositoryRAG,
)

ToolResult = dict[str, Any]


class RepositoryKnowledgeManager:
    """
    管理 Agent 当前使用的代码仓库 RAG 索引。

    一个 Manager 当前维护一个活动仓库。

    index_repository_knowledge():
        建立或切换活动仓库索引。

    search_repository_knowledge():
        查询当前活动仓库。
    """

    def __init__(
        self,
        model_name: str = ("BAAI/bge-small-zh-v1.5"),
        max_chunk_characters: int = 2400,
        overlap_lines: int = 8,
        max_context_characters: int = 10000,
        max_context_items: int = 6,
        show_progress_bar: bool = False,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_chunk_characters = max_chunk_characters
        self.overlap_lines = overlap_lines
        self.max_context_characters = max_context_characters
        self.max_context_items = max_context_items
        self.show_progress_bar = show_progress_bar
        self.device = device

        # 模型在第一次构建索引时才加载。
        self._embedding_client: EmbeddingClient | None = None

        self._repository_rag: PythonRepositoryRAG | None = None

        self._repository_path: Path | None = None

    @property
    def is_indexed(self) -> bool:
        return self._repository_rag is not None and self._repository_rag.is_indexed

    def index_repository_knowledge(
        self,
        repository_path: str,
    ) -> ToolResult:
        """
        为本地 Python 仓库建立语义索引。

        构建成功后，该仓库成为当前活动知识库。
        """

        if not isinstance(repository_path, str):
            raise TypeError("repository_path must be a string")

        stripped_path = repository_path.strip()

        if not stripped_path:
            return self._failure(
                error_type="invalid_repository_path",
                message=("repository_path cannot be empty"),
            )

        resolved_path = Path(stripped_path).resolve()

        if not resolved_path.exists():
            return self._failure(
                error_type="repository_not_found",
                message=("Repository does not exist: " f"{resolved_path}"),
            )

        if not resolved_path.is_dir():
            return self._failure(
                error_type="not_a_directory",
                message=("Repository path is not a directory: " f"{resolved_path}"),
            )

        embedding_client = self._get_embedding_client()

        # 先构造一套候选 RAG。
        # 只有索引完整建立成功后，才替换当前活动仓库。
        candidate_rag = PythonRepositoryRAG(
            repository_path=resolved_path,
            embedding_client=embedding_client,
            max_chunk_characters=(self.max_chunk_characters),
            overlap_lines=self.overlap_lines,
            max_context_characters=(self.max_context_characters),
            max_context_items=(self.max_context_items),
        )

        index_result = candidate_rag.rebuild()

        if index_result.is_empty:
            return self._failure(
                error_type="empty_repository_index",
                message=(
                    "No indexable Python code was found "
                    f"in repository: {resolved_path}"
                ),
            )

        # 到这里说明候选索引已经完整建立。
        self._repository_rag = candidate_rag
        self._repository_path = resolved_path

        return {
            "success": True,
            "repository_path": str(resolved_path),
            "document_count": (index_result.document_count),
            "chunk_count": (index_result.chunk_count),
            "vector_dimension": (index_result.vector_dimension),
            "message": ("Repository knowledge index " "was built successfully."),
        }

    def search_repository_knowledge(
        self,
        query: str,
        top_k: int = 8,
    ) -> ToolResult:
        """
        在当前活动代码仓库中进行语义检索。

        只返回代码证据，不调用额外 LLM。
        """

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        stripped_query = query.strip()

        if not stripped_query:
            return self._failure(
                error_type="invalid_query",
                message="query cannot be empty",
            )

        if not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")

        if not 1 <= top_k <= 12:
            return self._failure(
                error_type="invalid_top_k",
                message=("top_k must be between 1 and 12"),
            )

        repository_rag = self._repository_rag

        if repository_rag is None:
            return self._failure(
                error_type="repository_not_indexed",
                message=(
                    "No repository knowledge index "
                    "is currently available. Call "
                    "index_repository_knowledge first."
                ),
            )

        search_response = repository_rag.search(
            query=stripped_query,
            top_k=top_k,
        )

        evidence = [self._serialize_evidence(item) for item in search_response.sources]

        return {
            "success": True,
            "repository_path": str(self._repository_path),
            "query": stripped_query,
            "retrieved_count": len(search_response.search_results),
            "evidence_count": len(evidence),
            "evidence": evidence,
        }

    def _get_embedding_client(
        self,
    ) -> EmbeddingClient:
        if self._embedding_client is None:
            self._embedding_client = SentenceTransformerEmbeddingClient(
                model_name=self.model_name,
                show_progress_bar=(self.show_progress_bar),
                device=self.device,
            )

        return self._embedding_client

    @staticmethod
    def _serialize_evidence(
        item,
    ) -> ToolResult:
        chunk = item.chunk
        metadata = chunk.metadata

        return {
            "source_id": item.context_id,
            "source": chunk.source,
            "symbol": metadata.get("symbol"),
            "symbol_type": metadata.get("symbol_type"),
            "start_line": metadata.get("start_line"),
            "end_line": metadata.get("end_line"),
            "part_index": metadata.get("part_index"),
            "part_count": metadata.get("part_count"),
            "score": round(
                item.score,
                6,
            ),
            "content": chunk.content,
        }

    @staticmethod
    def _failure(
        error_type: str,
        message: str,
    ) -> ToolResult:
        return {
            "success": False,
            "error_type": error_type,
            "message": message,
        }
