from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

import torch

from evaluation.query_expander import (
    FrozenQueryExpander,
)
from evaluation.retrieval import (
    load_retrieval_cases,
)
from rag.multi_query_retriever import (
    MultiQueryRetriever,
)
from rag.python_repository_rag import (
    PythonRepositoryRAG,
)
from rag.reranker import (
    CrossEncoderReranker,
    RerankingRetriever,
)

REPOSITORY_PATH = Path(".")

CASES_PATH = Path("evaluation/repository_retrieval_cases.json")

EXPANSIONS_PATH = Path("evaluation/repository_query_expansions.json")

INDEX_DIRECTORY = Path(".rag_index")

CANDIDATE_COUNT = 30
FINAL_TOP_K = 8

# None = 跑全部 36 道。
# 如果只是快速试运行，可以临时改成 12。
CASE_LIMIT: int | None = None


try:
    import psutil
except ImportError:
    psutil = None


def synchronize_device() -> None:
    """
    CUDA 运算可能是异步的。

    benchmark 前后同步，
    避免 perf_counter 只测到任务提交时间。
    """

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def process_memory_mb() -> float | None:
    """
    返回当前进程 RSS。

    psutil 不存在时不影响 benchmark，
    只是跳过 CPU 内存统计。
    """

    if psutil is None:
        return None

    process = psutil.Process(os.getpid())

    return process.memory_info().rss / 1024 / 1024


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    if not values:
        raise ValueError("values must not be empty")

    ordered = sorted(values)

    index = round((len(ordered) - 1) * percentile_value)

    return ordered[index]


def print_latency_summary(
    name: str,
    durations: list[float],
) -> None:
    milliseconds = [duration * 1000 for duration in durations]

    print()
    print(f"=== {name} ===")
    print(f"Runs: {len(milliseconds)}")
    print("Mean: " f"{statistics.mean(milliseconds):.1f} ms")
    print("Median: " f"{statistics.median(milliseconds):.1f} ms")
    print("P95: " f"{percentile(milliseconds, 0.95):.1f} ms")
    print(f"Min: {min(milliseconds):.1f} ms")
    print(f"Max: {max(milliseconds):.1f} ms")


def measure_call(
    function,
) -> tuple[object, float]:
    synchronize_device()

    start = time.perf_counter()

    result = function()

    synchronize_device()

    duration = time.perf_counter() - start

    return result, duration


def main() -> None:
    cases = load_retrieval_cases(CASES_PATH)

    if CASE_LIMIT is not None:
        cases = cases[:CASE_LIMIT]

    print(f"Benchmark cases: {len(cases)}")

    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )

    if torch.cuda.is_available():
        print(
            "CUDA device:",
            torch.cuda.get_device_name(0),
        )

    repository_rag = PythonRepositoryRAG(
        repository_path=(REPOSITORY_PATH),
        show_progress_bar=False,
    )

    repository_rag.ensure_index(index_directory=(INDEX_DIRECTORY))

    frozen_expander = FrozenQueryExpander.from_json(EXPANSIONS_PATH)

    candidate_retriever = MultiQueryRetriever(
        base_retriever=(repository_rag.hybrid_retriever),
        query_expander=(frozen_expander),
        rrf_k=60,
    )

    reranker = CrossEncoderReranker(
        model_name=("BAAI/bge-reranker-v2-m3"),
        batch_size=8,
        max_length=512,
        show_progress_bar=False,
    )

    full_retriever = RerankingRetriever(
        base_retriever=(candidate_retriever),
        reranker=reranker,
        candidate_count=(CANDIDATE_COUNT),
    )

    # ----------------------------------
    # 1. Candidate Retrieval
    # ----------------------------------

    candidate_durations: list[float] = []

    candidate_sets = []

    for case in cases:
        candidates, duration = measure_call(
            lambda case=case: (
                candidate_retriever.retrieve(
                    query=case.query,
                    top_k=(CANDIDATE_COUNT),
                )
            )
        )

        candidate_sets.append(
            (
                case,
                candidates,
            )
        )

        candidate_durations.append(duration)

    print_latency_summary(
        "MultiQuery+Hybrid " "Candidate Retrieval Top30",
        candidate_durations,
    )

    # ----------------------------------
    # 2. Cold Reranker
    # ----------------------------------

    first_case, first_candidates = candidate_sets[0]

    memory_before = process_memory_mb()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    _, cold_duration = measure_call(
        lambda: reranker.rerank(
            query=first_case.query,
            results=first_candidates,
        )
    )

    memory_after = process_memory_mb()

    print()
    print("=== Reranker Cold Start ===")
    print("Cold load + rerank Top30: " f"{cold_duration * 1000:.1f} ms")

    if memory_before is not None and memory_after is not None:
        print("Process RSS before: " f"{memory_before:.1f} MB")
        print("Process RSS after: " f"{memory_after:.1f} MB")
        print("RSS increase: " f"{memory_after - memory_before:.1f} MB")

    else:
        print("CPU RSS: unavailable " "(install psutil if needed)")

    if torch.cuda.is_available():
        print(
            "CUDA peak allocated: "
            f"{torch.cuda.max_memory_allocated() / 1024**2:.1f} MB"
        )
        print(
            "CUDA peak reserved: "
            f"{torch.cuda.max_memory_reserved() / 1024**2:.1f} MB"
        )

    # ----------------------------------
    # 3. Warm Reranker
    # ----------------------------------

    warm_rerank_durations: list[float] = []

    # 第一个 case 已经用于 cold，
    # 这里仍然可以再次运行，
    # 此时模型已经 warm。
    for case, candidates in candidate_sets:
        _, duration = measure_call(
            lambda case=case, candidates=candidates: (
                reranker.rerank(
                    query=case.query,
                    results=candidates,
                )
            )
        )

        warm_rerank_durations.append(duration)

    print_latency_summary(
        "v2-m3 Warm Rerank Top30",
        warm_rerank_durations,
    )

    # ----------------------------------
    # 4. Full Retrieval Pipeline
    # ----------------------------------

    full_durations: list[float] = []

    for case in cases:
        _, duration = measure_call(
            lambda case=case: (
                full_retriever.retrieve(
                    query=case.query,
                    top_k=FINAL_TOP_K,
                )
            )
        )

        full_durations.append(duration)

    print_latency_summary(
        "MultiQuery+Hybrid" "+v2-m3 Full Top8",
        full_durations,
    )

    # ----------------------------------
    # 5. Incremental cost
    # ----------------------------------

    candidate_mean = statistics.mean(candidate_durations) * 1000

    rerank_mean = statistics.mean(warm_rerank_durations) * 1000

    full_mean = statistics.mean(full_durations) * 1000

    print()
    print("=== Cost Summary ===")

    print("Candidate retrieval mean: " f"{candidate_mean:.1f} ms")

    print("Warm reranker mean: " f"{rerank_mean:.1f} ms")

    print("Full pipeline mean: " f"{full_mean:.1f} ms")

    if candidate_mean > 0:
        print(
            "Warm reranker / candidate "
            "latency ratio: "
            f"{rerank_mean / candidate_mean:.2f}x"
        )


if __name__ == "__main__":
    main()
