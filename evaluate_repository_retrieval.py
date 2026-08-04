from pathlib import Path

from evaluation.retrieval import (
    RetrievalEvaluationSummary,
    RetrievalEvaluationCase,
    RetrievalEvaluator,
    load_retrieval_cases,
)
from rag.python_repository_rag import (
    PythonRepositoryRAG,
)

from evaluation.query_expander import (
    FrozenQueryExpander,
)

from rag.multi_query_retriever import (
    MultiQueryRetriever,
)

from rag.reranker import RerankingRetriever

from rag.result_diversifier import DiversifiedRetriever

REPOSITORY_PATH = Path(".")


def print_summary(
    summary: RetrievalEvaluationSummary,
) -> None:
    print()
    print(f"=== {summary.retriever_name} ===")

    print(f"Cases: {summary.case_count}")
    print("Top-1 Accuracy: " f"{summary.top1_accuracy:.3f}")
    print("HitRate@3: " f"{summary.hit_rate_at_3:.3f}")
    print("HitRate@5: " f"{summary.hit_rate_at_5:.3f}")
    print("HitRate@8: " f"{summary.hit_rate_at_8:.3f}")
    print("Recall@3: " f"{summary.recall_at_3:.3f}")
    print("Recall@5: " f"{summary.recall_at_5:.3f}")
    print("Recall@8: " f"{summary.recall_at_8:.3f}")
    print("Complete@3: " f"{summary.complete_rate_at_3:.3f}")
    print("Complete@5: " f"{summary.complete_rate_at_5:.3f}")
    print("Complete@8: " f"{summary.complete_rate_at_8:.3f}")
    print("Candidate Recall@30: " f"{summary.candidate_recall_at_30:.3f}")
    print("Candidate Complete@30: " f"{summary.candidate_complete_at_30:.3f}")
    print(f"MRR: {summary.mrr:.3f}")


def print_failures(
    summary: RetrievalEvaluationSummary,
    max_results: int = 5,
) -> None:
    print()
    print(f"--- {summary.retriever_name} " "misses ---")

    for result in summary.case_results:
        if result.hit_at(5):
            continue

        print()
        print(f"[{result.case_id}]")
        print(result.query)

        for search_result in result.retrieved[:max_results]:
            chunk = search_result.chunk

            print(
                f"  #{search_result.rank} "
                f"{chunk.source} :: "
                f"{chunk.metadata.get('symbol')}"
            )


def print_case_details(
    summary: RetrievalEvaluationSummary,
    cases: list[RetrievalEvaluationCase],
    max_results: int = 5,
) -> None:
    case_by_id = {case.id: case for case in cases}

    print()
    print(f"--- {summary.retriever_name} " "incomplete cases ---")

    for result in summary.case_results:
        if result.complete_at(8):
            continue

        case = case_by_id[result.case_id]

        print()
        print(f"[{result.case_id}]")
        print(result.query)

        print("Expected groups:")

        for index, group in enumerate(case.relevant_groups):
            rank = result.group_first_ranks[index]

            print(f"  [{group.name}] " f"rank={rank}")

            for target in group.alternatives:
                print("    - " f"{target.source} :: " f"{target.symbol}")

        print("Retrieved:")

        for search_result in result.retrieved[:max_results]:
            chunk = search_result.chunk

            print(
                f"  #{search_result.rank} "
                f"{chunk.source} :: "
                f"{chunk.metadata.get('symbol')}"
            )


def debug_case_matching(
    *,
    cases,
    summary,
    case_id: str,
) -> None:
    case = next(case for case in cases if case.id == case_id)

    result = next(
        result for result in summary.case_results if result.case_id == case_id
    )

    print()
    print(f"=== DEBUG {case_id} ===")

    for search_result in result.retrieved[:5]:
        chunk = search_result.chunk

        print()
        print(f"Result #{search_result.rank}")

        print(
            "source:",
            repr(chunk.source),
        )

        print(
            "symbol:",
            repr(chunk.metadata.get("symbol")),
        )

        for group in case.relevant_groups:
            print(f"group={group.name}")

            for target in group.alternatives:
                print(
                    "  target source:",
                    repr(target.source),
                )

                print(
                    "  target symbol:",
                    repr(target.symbol),
                )

                print(
                    "  matches:",
                    target.matches(chunk),
                )


def print_query_expansions(
    repository_rag,
    cases,
) -> None:
    print()
    print("=== Query Expansions ===")

    for case in cases:
        rewrites = repository_rag.query_expander.expand(case.query)

        print()
        print(f"[{case.id}]")

        print(f"Original: {case.query}")

        for index, rewrite in enumerate(
            rewrites,
            start=1,
        ):
            print(f"Rewrite {index}: " f"{rewrite}")


def main() -> None:
    cases = load_retrieval_cases("evaluation/" "repository_retrieval_cases.json")

    repository_rag = PythonRepositoryRAG(
        repository_path=(REPOSITORY_PATH),
        show_progress_bar=False,
    )

    repository_rag.ensure_index(index_directory=".rag_index")

    frozen_expander = FrozenQueryExpander.from_json(
        "evaluation/" "repository_query_expansions.json"
    )

    frozen_multi_query_retriever = MultiQueryRetriever(
        base_retriever=(repository_rag.hybrid_retriever),
        query_expander=(frozen_expander),
        rrf_k=60,
    )

    frozen_reranking_retriever = RerankingRetriever(
        base_retriever=(frozen_multi_query_retriever),
        reranker=(repository_rag.cross_encoder_reranker),
        candidate_count=30,
    )

    frozen_reranked_diversified = DiversifiedRetriever(
        base_retriever=(frozen_reranking_retriever),
        diversifier=(repository_rag.result_diversifier),
        candidate_count=30,
    )

    evaluator = RetrievalEvaluator(top_k=30)

    retrievers = {
        # "Vector": (repository_rag.vector_retriever),
        # "MultiQuery+Vector": repository_rag.multi_query_dense_retriever,
        # "BM25": (repository_rag.bm25_retriever),
        # "Hybrid": (repository_rag.hybrid_retriever),
        # "Hybrid+Diversity": (repository_rag.retriever),
        "MultiQuery+Hybrid": frozen_multi_query_retriever,
        # "MultiQuery+Hybrid+Diversity": (
        #     repository_rag.multi_query_diversified_retriever
        # ),
        "MultiQuery+Hybrid+Reranker": frozen_reranking_retriever,
        "MultiQuery+Hybrid+Reranker+Diversity": frozen_reranked_diversified,
    }

    for name, retriever in retrievers.items():
        summary = evaluator.evaluate(
            retriever_name=name,
            retriever=retriever,
            cases=cases,
        )

        print_summary(summary)
        print_case_details(
            summary,
            cases,
        )
        print_failures(summary)
        # debug_case_matching(cases=cases, summary=summary, case_id="repository-state")

    print_query_expansions(
        repository_rag,
        cases,
    )


if __name__ == "__main__":
    main()
