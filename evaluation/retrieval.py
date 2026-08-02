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
class RelevantGroup:
    """
    一个必须被召回的证据组。

    alternatives 中的 Target 是 OR 关系：
    只要命中其中一个，就认为该证据组被召回。
    """

    name: str
    alternatives: tuple[RelevantTarget, ...]

    def matches(
        self,
        chunk: Chunk,
    ) -> bool:
        return any(target.matches(chunk) for target in self.alternatives)


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    """
    一个检索评估问题。

    relevant_groups 之间是 AND 关系：
    一个问题可能需要多个独立证据组。
    """

    id: str
    query: str
    relevant_groups: tuple[RelevantGroup, ...]


@dataclass(frozen=True)
class RetrievalCaseResult:
    case_id: str
    query: str

    retrieved: tuple[SearchResult, ...]

    group_first_ranks: tuple[int | None, ...]

    @property
    def first_relevant_rank(
        self,
    ) -> int | None:
        ranks = [rank for rank in self.group_first_ranks if rank is not None]

        if not ranks:
            return None

        return min(ranks)

    @property
    def top1_correct(self) -> bool:
        """
        排名第一的结果是否属于任意正确证据组。
        """

        return self.first_relevant_rank == 1

    def hit_at(
        self,
        k: int,
    ) -> bool:
        """
        Top-k 中是否至少出现一个正确证据组。

        这是“有没有找到答案入口”。
        """

        return any(rank is not None and rank <= k for rank in self.group_first_ranks)

    def recall_at(
        self,
        k: int,
    ) -> float:
        """
        Top-k 找到了多少 required group。

        例如需要两个证据组，
        Top-5 只找到其中一个：

            Recall@5 = 0.5
        """

        group_count = len(self.group_first_ranks)

        if group_count == 0:
            return 0.0

        matched_count = sum(
            1 for rank in self.group_first_ranks if (rank is not None and rank <= k)
        )

        return matched_count / group_count

    def complete_at(
        self,
        k: int,
    ) -> bool:
        """
        回答问题所需的全部证据组
        是否都进入 Top-k。
        """

        if not self.group_first_ranks:
            return False

        return all(rank is not None and rank <= k for rank in self.group_first_ranks)

    @property
    def reciprocal_rank(self) -> float:
        """
        第一个相关结果的 Reciprocal Rank。
        """

        rank = self.first_relevant_rank

        if rank is None:
            return 0.0

        return 1.0 / rank


@dataclass(frozen=True)
class RetrievalEvaluationSummary:
    retriever_name: str
    case_count: int

    top1_accuracy: float

    hit_rate_at_3: float
    hit_rate_at_5: float
    hit_rate_at_8: float

    recall_at_3: float
    recall_at_5: float
    recall_at_8: float

    complete_rate_at_3: float
    complete_rate_at_5: float
    complete_rate_at_8: float

    candidate_recall_at_30: float
    candidate_complete_at_30: float

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
        if not cases:
            raise ValueError("Evaluation cases cannot " "be empty")

        case_results: list[RetrievalCaseResult] = []

        for case in cases:
            results = retriever.retrieve(
                query=case.query,
                top_k=self.top_k,
            )

            case_results.append(
                self._evaluate_case(
                    case=case,
                    results=results,
                )
            )

        case_count = len(case_results)

        return RetrievalEvaluationSummary(
            retriever_name=retriever_name,
            case_count=case_count,
            top1_accuracy=self._average(result.top1_correct for result in case_results),
            hit_rate_at_3=self._average(result.hit_at(3) for result in case_results),
            hit_rate_at_5=self._average(result.hit_at(5) for result in case_results),
            hit_rate_at_8=self._average(result.hit_at(8) for result in case_results),
            recall_at_3=sum(result.recall_at(3) for result in case_results)
            / case_count,
            recall_at_5=sum(result.recall_at(5) for result in case_results)
            / case_count,
            recall_at_8=sum(result.recall_at(8) for result in case_results)
            / case_count,
            complete_rate_at_3=self._average(
                result.complete_at(3) for result in case_results
            ),
            complete_rate_at_5=self._average(
                result.complete_at(5) for result in case_results
            ),
            complete_rate_at_8=self._average(
                result.complete_at(8) for result in case_results
            ),
            candidate_recall_at_30=sum(result.recall_at(30) for result in case_results)
            / case_count,
            candidate_complete_at_30=(
                self._average(result.complete_at(30) for result in case_results)
            ),
            mrr=sum(result.reciprocal_rank for result in case_results) / case_count,
            case_results=tuple(case_results),
        )

    @staticmethod
    def _evaluate_case(
        *,
        case: RetrievalEvaluationCase,
        results: list[SearchResult],
    ) -> RetrievalCaseResult:
        group_first_ranks: list[int | None] = [None for _ in case.relevant_groups]

        for result in results:
            for group_index, group in enumerate(case.relevant_groups):
                # 已经找到该证据组的最高排名，
                # 后面的结果不需要继续覆盖。
                if group_first_ranks[group_index] is not None:
                    continue

                if group.matches(result.chunk):
                    group_first_ranks[group_index] = result.rank

        return RetrievalCaseResult(
            case_id=case.id,
            query=case.query,
            retrieved=tuple(results),
            group_first_ranks=tuple(group_first_ranks),
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
        raw_groups = raw_case.get("relevant_groups")

        if (
            not isinstance(
                raw_groups,
                list,
            )
            or not raw_groups
        ):
            raise ValueError(f"Case {case_id} must have " "at least one relevant group")

        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Evaluation case id must " "be a non-empty string")

        if case_id in seen_ids:
            raise ValueError("Duplicate evaluation case " f"id: {case_id}")

        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Case {case_id} has an " "invalid query")

        relevant_groups: list[RelevantGroup] = []

        for raw_group in raw_groups:
            relevant_groups.append(
                _parse_relevant_group(
                    case_id,
                    raw_group,
                )
            )

        cases.append(
            RetrievalEvaluationCase(
                id=case_id,
                query=query.strip(),
                relevant_groups=tuple(relevant_groups),
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


def _parse_relevant_group(
    case_id: str,
    payload: Any,
) -> RelevantGroup:
    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(f"Case {case_id} contains " "an invalid relevant group")

    name = payload.get("name")

    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Case {case_id} relevant " "group requires a name")

    alternatives = payload.get("alternatives")

    if (
        not isinstance(
            alternatives,
            list,
        )
        or not alternatives
    ):
        raise ValueError(
            f"Case {case_id} group " f"{name} requires at least " "one alternative"
        )

    targets = tuple(
        _parse_relevant_target(
            case_id,
            raw_target,
        )
        for raw_target in alternatives
    )

    return RelevantGroup(
        name=name.strip(),
        alternatives=targets,
    )
