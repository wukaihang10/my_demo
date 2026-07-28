from rag import (
    InMemoryVectorStore,
    PythonASTChunker,
    PythonDocumentLoader,
    RAGIndexer,
    SentenceTransformerEmbeddingClient,
    TextChunker,
    VectorRetriever,
)


def main() -> None:
    loader = PythonDocumentLoader()

    chunker = PythonASTChunker(
        fallback_chunker=TextChunker(
            chunk_size=1200,
            chunk_overlap=150,
        )
    )

    embedding_client = SentenceTransformerEmbeddingClient(
        show_progress_bar=True,
    )

    vector_store = InMemoryVectorStore(dimension=embedding_client.dimension)

    indexer = RAGIndexer(
        loader=loader,
        chunker=chunker,
        embedding_client=embedding_client,
        vector_store=vector_store,
    )

    index_result = indexer.rebuild_directory(".")

    print()
    print(f"Indexed documents: " f"{index_result.document_count}")
    print(f"Indexed chunks: " f"{index_result.chunk_count}")

    retriever = VectorRetriever(
        embedding_client=embedding_client,
        vector_store=vector_store,
    )

    queries = [
        "execute_tool 如何处理工具内部异常？",
        "Agent 的运行主循环在哪里？",
        "哪个方法负责更新 trace？",
        "RepositoryState 保存了哪些状态？",
    ]

    for query in queries:
        results = retriever.retrieve(
            query=query,
            top_k=5,
        )

        print()
        print("=" * 80)
        print(f"Query: {query}")

        for result in results:
            chunk = result.chunk
            metadata = chunk.metadata

            print()
            print("-" * 80)
            print(f"Rank: {result.rank}")
            print(f"Score: {result.score:.6f}")
            print(f"Source: {chunk.source}")
            print("Symbol: " f"{metadata.get('symbol')}")
            print(
                "Lines: " f"{metadata.get('start_line')}-" f"{metadata.get('end_line')}"
            )
            print()
            print(chunk.content[:800])


if __name__ == "__main__":
    main()
