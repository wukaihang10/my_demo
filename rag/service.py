from __future__ import annotations

from rag.context_builder import ContextBuilder
from rag.generator import RAGGenerator
from rag.models import RAGAnswer
from rag.retriever import VectorRetriever


class NaiveRAG:
    """
    最基础的完整 RAG 流程。

    query
        ↓
    VectorRetriever
        ↓
    SearchResult
        ↓
    ContextBuilder
        ↓
    BuiltContext
        ↓
    RAGGenerator
        ↓
    answer
    """

    def __init__(
        self,
        retriever: VectorRetriever,
        context_builder: ContextBuilder,
        generator: RAGGenerator,
    ) -> None:
        self.retriever = retriever
        self.context_builer = context_builder
        self.generator = generator

    def answer(
        self,
        question: str,
        top_k: int = 5,
        minimum_score: float | None = None,
    ) -> RAGAnswer:
        search_results = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            minimum_score=minimum_score,
        )

        context = self.context_builer.build(search_results)

        if context.is_empty:
            return RAGAnswer(
                question=question,
                answer="知识库中没有检索到可用于回答该问题的资料。",
                context=context,
                search_results=search_results,
            )

        generated_answer = self.generator.generate(
            question=question,
            context=context,
        )

        return RAGAnswer(
            question=question,
            answer=generated_answer,
            context=context,
            search_results=search_results,
        )
