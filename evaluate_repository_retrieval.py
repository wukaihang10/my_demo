from pathlib import Path

from evaluation.retrieval import (
    RetrievalEvaluationSummary,
    RetrievalEvaluator,
    load_retrieval_cases,
)
from rag.python_repository_rag import (
    PythonRepositoryRAG,
)

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
    print("Recall@3: " f"{summary.recall_at_3:.3f}")
    print("Recall@5: " f"{summary.recall_at_5:.3f}")
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


def main() -> None:
    cases = load_retrieval_cases("evaluation/" "repository_retrieval_cases.json")

    repository_rag = PythonRepositoryRAG(
        repository_path=(REPOSITORY_PATH),
        show_progress_bar=False,
    )

    repository_rag.ensure_index(index_directory=".rag_index")

    evaluator = RetrievalEvaluator(top_k=10)

    retrievers = {
        "Vector": (repository_rag.vector_retriever),
        "BM25": (repository_rag.bm25_retriever),
        "Hybrid": (repository_rag.retriever),
    }

    for name, retriever in retrievers.items():
        summary = evaluator.evaluate(
            retriever_name=name,
            retriever=retriever,
            cases=cases,
        )

        print_summary(summary)
        print_failures(summary)


if __name__ == "__main__":
    main()
