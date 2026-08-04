from __future__ import annotations

from pathlib import Path

from evaluation.retrieval import load_retrieval_cases
from rag.multi_query_retriever import MultiQueryRetriever
from rag.python_repository_rag import PythonRepositoryRAG
from rag.reranker import CrossEncoderReranker, RerankingRetriever
from evaluation.query_expander import FrozenQueryExpander

REPOSITORY_PATH = Path(".").resolve()

CASES_PATH = Path("evaluation/repository_retrieval_cases.json")

EXPANSIONS_PATH = Path("evaluation/repository_query_expansions.json")

INDEX_DIRECTORY = REPOSITORY_PATH / ".rag_index"

RETRIEVAL_TOP_K = 8


def main() -> None:
    cases = load_retrieval_cases(CASES_PATH)

    # 这里只利用 production RAG 中已经稳定的：
    # loader / index / vector / BM25 / hybrid /
    # ContextBuilder。
    #
    # Frozen Query Expansion 和 reranker
    # 由 Evaluation 自己组合，
    # 保证结果可复现。
    repository_rag = PythonRepositoryRAG(
        repository_path=REPOSITORY_PATH,
        # 与 RepositoryKnowledgeManager
        # production 默认 Evidence 预算保持一致。
        max_context_characters=8000,
        max_context_items=5,
        show_progress_bar=False,
        # 避免这里额外构造 production reranker。
        # Evaluation 自己在下面创建。
        retrieval_mode="fast",
    )

    repository_rag.ensure_index(INDEX_DIRECTORY)

    frozen_expander = FrozenQueryExpander.from_json(EXPANSIONS_PATH)

    multi_query_retriever = MultiQueryRetriever(
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

    quality_retriever = RerankingRetriever(
        base_retriever=(multi_query_retriever),
        reranker=reranker,
        candidate_count=30,
    )

    total_recall = 0.0
    hit_count = 0
    complete_count = 0

    total_selected_items = 0
    total_context_characters = 0

    incomplete_cases: list[str] = []

    for case in cases:
        results = quality_retriever.retrieve(
            query=case.query,
            top_k=RETRIEVAL_TOP_K,
        )

        context = repository_rag.context_builder.build(results)

        chunks = [item.chunk for item in context.items]

        group_matches = [
            any(group.matches(chunk) for chunk in chunks)
            for group in case.relevant_groups
        ]

        matched_count = sum(group_matches)

        group_count = len(case.relevant_groups)

        recall = matched_count / group_count

        hit = matched_count > 0
        complete = matched_count == group_count

        total_recall += recall
        hit_count += int(hit)
        complete_count += int(complete)

        total_selected_items += len(context.items)

        total_context_characters += context.character_count

        if not complete:
            lines = [
                "",
                f"[{case.id}]",
                case.query,
                ("Matched groups: " f"{matched_count}/" f"{group_count}"),
                "Final evidence:",
            ]

            for item in context.items:
                chunk = item.chunk

                symbol = chunk.metadata.get("symbol")

                lines.append(
                    "  "
                    f"retrieval_rank="
                    f"{item.retrieval_rank} "
                    f"{chunk.source} "
                    f":: {symbol}"
                )

            incomplete_cases.append("\n".join(lines))

    case_count = len(cases)

    print()
    print("=== Final Evidence Evaluation ===")

    print(f"Cases: {case_count}")

    print("Final Evidence Hit Rate: " f"{hit_count / case_count:.3f}")

    print("Final Evidence Recall: " f"{total_recall / case_count:.3f}")

    print("Final Evidence Complete Rate: " f"{complete_count / case_count:.3f}")

    print("Average selected items: " f"{total_selected_items / case_count:.2f}")

    print("Average context characters: " f"{total_context_characters / case_count:.1f}")

    if incomplete_cases:
        print()
        print("--- Incomplete final evidence ---")

        for text in incomplete_cases:
            print(text)


if __name__ == "__main__":
    main()
