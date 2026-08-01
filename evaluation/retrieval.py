from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.interfaces import Retriever
from rag.models import Chunk, SearchResult


@dataclass(frozen=True)
class RelevantTarget:
    """
    一个检索问题对应的正确代码目标。

    source 必须匹配。

    symbol / symbol_type 为可选条件。
    """

    source: str
    symbol: str | None = None
    symbol_type: str | None = None

    def matches(
        self,
        chunk: Chunk,
    ) -> bool:
        if chunk.source != self.source:
            return False

        metadata = chunk.metadata

        if self.symbol is not None and metadata.get("symbol") != self.symbol:
            return False

        if (
            self.symbol_type is not None
            and metadata.get("symbol_type") != self.symbol_type
        ):
            return False

        return True


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    id: str
    query: str
    relevant: tuple[RelevantTarget, ...]


@dataclass(frozen=True)
class RetrievalCaseResult:
    case_id: str
    query: str

    retrieved: tuple[SearchResult, ...]

    relevant_count: int
    retrieved_relevant_count: int

    first_relevant_rank: int | None

    @property
    def top1_correct(self) -> bool:
        return self.first_relevant_rank == 1

    def hit_at(
        self,
        k: int,
    ) -> bool:
        return self.first_relevant_rank is not None and self.first_relevant_rank <= k

    def recall_at(
        self,
        k: int,
        case: RetrievalEvaluationCase,
    ) -> float:
        if not case.relevant:
            return 0.0

        retrieved_targets: set[int] = set()

        for result in self.retrieved[:k]:
            for index, target in enumerate(case.relevant):
                if target.matches(result.chunk):
                    retrieved_targets.add(index)

        return len(retrieved_targets) / len(case.relevant)

    @property
    def reciprocal_rank(self) -> float:
        if self.first_relevant_rank is None:
            return 0.0

        return 1.0 / self.first_relevant_rank


@dataclass(frozen=True)
class RetrievalEvaluationSummary:
    retriever_name: str
    case_count: int

    top1_accuracy: float
    hit_rate_at_3: float
    hit_rate_at_5: float

    recall_at_3: float
    recall_at_5: float

    mrr: float

    case_results: tuple[RetrievalCaseResult, ...]


class RetrievalEvaluator:
    def __init__(
        self,
        top_k: int = 10,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        self.top_k = top_k

    def evaluate(
        self,
        *,
        retriever_name: str,
        retriever: Retriever,
        cases: list[RetrievalEvaluationCase],
    ) -> RetrievalEvaluationSummary:
        case_results: list[RetrievalCaseResult] = []

        recalls_at_3: list[float] = []
        recalls_at_5: list[float] = []

        for case in cases:
            results = retriever.retrieve(
                query=case.query,
                top_k=self.top_k,
            )

            case_result = self._evaluate_case(
                case=case,
                results=results,
            )

            case_results.append(case_result)

            recalls_at_3.append(
                case_result.recall_at(
                    3,
                    case,
                )
            )

            recalls_at_5.append(
                case_result.recall_at(
                    5,
                    case,
                )
            )

        case_count = len(case_results)

        if case_count == 0:
            raise ValueError("Evaluation cases cannot " "be empty")

        return RetrievalEvaluationSummary(
            retriever_name=retriever_name,
            case_count=case_count,
            top1_accuracy=self._average(result.top1_correct for result in case_results),
            hit_rate_at_3=self._average(result.hit_at(3) for result in case_results),
            hit_rate_at_5=self._average(result.hit_at(5) for result in case_results),
            recall_at_3=sum(recalls_at_3) / case_count,
            recall_at_5=sum(recalls_at_5) / case_count,
            mrr=sum(result.reciprocal_rank for result in case_results) / case_count,
            case_results=tuple(case_results),
        )

    @staticmethod
    def _evaluate_case(
        *,
        case: RetrievalEvaluationCase,
        results: list[SearchResult],
    ) -> RetrievalCaseResult:
        matched_targets: set[int] = set()

        first_relevant_rank: int | None = None

        for result in results:
            result_is_relevant = False

            for target_index, target in enumerate(case.relevant):
                if target.matches(result.chunk):
                    matched_targets.add(target_index)

                    result_is_relevant = True

            if result_is_relevant and first_relevant_rank is None:
                first_relevant_rank = result.rank

        return RetrievalCaseResult(
            case_id=case.id,
            query=case.query,
            retrieved=tuple(results),
            relevant_count=len(case.relevant),
            retrieved_relevant_count=len(matched_targets),
            first_relevant_rank=(first_relevant_rank),
        )

    @staticmethod
    def _average(
        values,
    ) -> float:
        normalized = [1.0 if value else 0.0 for value in values]

        if not normalized:
            return 0.0

        return sum(normalized) / len(normalized)


def load_retrieval_cases(
    path: str | Path,
) -> list[RetrievalEvaluationCase]:
    case_path = Path(path)

    with case_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Retrieval evaluation dataset " "must contain a JSON list")

    cases: list[RetrievalEvaluationCase] = []

    seen_ids: set[str] = set()

    for raw_case in payload:
        if not isinstance(
            raw_case,
            dict,
        ):
            raise ValueError("Each evaluation case must " "be an object")

        case_id = raw_case.get("id")
        query = raw_case.get("query")
        raw_relevant = raw_case.get("relevant")

        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Evaluation case id must " "be a non-empty string")

        if case_id in seen_ids:
            raise ValueError("Duplicate evaluation case " f"id: {case_id}")

        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Case {case_id} has an " "invalid query")

        if (
            not isinstance(
                raw_relevant,
                list,
            )
            or not raw_relevant
        ):
            raise ValueError(
                f"Case {case_id} must have " "at least one relevant target"
            )

        relevant_targets: list[RelevantTarget] = []

        for raw_target in raw_relevant:
            relevant_targets.append(
                _parse_relevant_target(
                    case_id,
                    raw_target,
                )
            )

        cases.append(
            RetrievalEvaluationCase(
                id=case_id,
                query=query.strip(),
                relevant=tuple(relevant_targets),
            )
        )

        seen_ids.add(case_id)

    return cases


def _parse_relevant_target(
    case_id: str,
    payload: Any,
) -> RelevantTarget:
    if not isinstance(payload, dict):
        raise ValueError(f"Case {case_id} contains " "an invalid relevant target")

    source = payload.get("source")

    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"Case {case_id} relevant " "target requires source")

    symbol = payload.get("symbol")
    symbol_type = payload.get("symbol_type")

    if symbol is not None and not isinstance(symbol, str):
        raise ValueError(f"Case {case_id} symbol " "must be a string or null")

    if symbol_type is not None and not isinstance(
        symbol_type,
        str,
    ):
        raise ValueError(f"Case {case_id} symbol_type " "must be a string or null")

    return RelevantTarget(
        source=source.strip(),
        symbol=(symbol.strip() if symbol else None),
        symbol_type=(symbol_type.strip() if symbol_type else None),
    )
