from __future__ import annotations

from pathlib import Path

from rag.interfaces import EmbeddingClient

from rag.code_prompt_builder import (
    CodeRAGPromptBuilder,
)
from rag.context_builder import ContextBuilder
from rag.embedding import (
    SentenceTransformerEmbeddingClient,
)
from rag.generator import RAGGenerator
from rag.indexer import RAGIndexer
from rag.models import (
    IndexBuildResult,
    RAGAnswer,
    RAGSearchResponse,
)
from rag.python_chunker import (
    PythonASTChunker,
)
from rag.python_loader import (
    PythonDocumentLoader,
)
from rag.retriever import VectorRetriever
from rag.service import NaiveRAG
from rag.vector_store import (
    InMemoryVectorStore,
)


class PythonRepositoryRAG:
    """
    面向本地 Python 仓库的完整 RAG。

    rebuild():
        扫描并建立代码索引。

    answer():
        对已经建立的代码索引进行问答。
    """

    def __init__(
        self,
        repository_path: str | Path,
        model_name: str = ("BAAI/bge-small-zh-v1.5"),
        embedding_client: EmbeddingClient | None = None,
        max_chunk_characters: int = 2400,
        overlap_lines: int = 8,
        max_context_characters: int = 10000,
        max_context_items: int = 6,
        show_progress_bar: bool = False,
        device: str | None = None,
    ) -> None:
        self.repository_path = Path(repository_path).resolve()

        if not self.repository_path.exists():
            raise FileNotFoundError(
                "Repository does not exist: " f"{self.repository_path}"
            )

        if not self.repository_path.is_dir():
            raise NotADirectoryError(
                "Repository path is not a directory: " f"{self.repository_path}"
            )

        self.loader = PythonDocumentLoader()

        self.chunker = PythonASTChunker(
            max_chunk_characters=(max_chunk_characters),
            overlap_lines=overlap_lines,
        )

        if embedding_client is None:
            self.embedding_client = SentenceTransformerEmbeddingClient(
                model_name=model_name,
                show_progress_bar=(show_progress_bar),
                device=device,
            )

        else:
            embedding_client = embedding_client

        self.vector_store = InMemoryVectorStore(
            dimension=(self.embedding_client.dimension)
        )

        self.indexer = RAGIndexer(
            loader=self.loader,
            chunker=self.chunker,
            embedding_client=(self.embedding_client),
            vector_store=self.vector_store,
        )

        self.retriever = VectorRetriever(
            embedding_client=(self.embedding_client),
            vector_store=self.vector_store,
        )

        self.context_builder = ContextBuilder(
            max_context_characters=(max_context_characters),
            max_items=max_context_items,
            include_scores=False,
        )

        self.generator = RAGGenerator(prompt_builder=(CodeRAGPromptBuilder()))

        self.service = NaiveRAG(
            retriever=self.retriever,
            context_builder=(self.context_builder),
            generator=self.generator,
        )

    @property
    def is_indexed(self) -> bool:
        return not self.vector_store.is_empty

    def rebuild(self) -> IndexBuildResult:
        return self.indexer.rebuild_directory(self.repository_path)

    def answer(
        self,
        question: str,
        top_k: int = 12,
        minimum_score: float | None = None,
    ) -> RAGAnswer:
        if not self.is_indexed:
            raise RuntimeError(
                "Repository index has not been built. "
                "Call rebuild() before answer()."
            )

        return self.service.answer(
            question=question,
            top_k=top_k,
            minimum_score=minimum_score,
        )

    def search(
        self,
        query: str,
        top_k: int = 12,
        minimum_score: float | None = None,
    ) -> RAGSearchResponse:
        if not self.is_indexed:
            raise RuntimeError(
                "Repository index has not been built. "
                "Call rebuild() before search()."
            )

        return self.service.search(
            query=query,
            top_k=top_k,
            minimum_score=minimum_score,
        )
