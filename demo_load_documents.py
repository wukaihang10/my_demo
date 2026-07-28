import numpy as np

from rag import (
    SentenceTransformerEmbeddingClient,
    TextChunker,
    TextDocumentLoader,
)


def main() -> None:
    loader = TextDocumentLoader()

    documents = loader.load_directory("knowledge")

    chunker = TextChunker(
        chunk_size=120,
        chunk_overlap=30,
    )

    chunks = chunker.split_documents(documents)

    embedding_client = SentenceTransformerEmbeddingClient(
        show_progress_bar=True,
    )

    document_vectors = embedding_client.embed_documents(
        [chunk.content for chunk in chunks]
    )

    query = "工具执行结果会如何返回给 Agent？"

    query_vector = embedding_client.embed_query(query)

    print()
    print("=" * 70)
    print(f"Model: " f"{embedding_client.model_name}")
    print(f"Embedding dimension: " f"{embedding_client.dimension}")
    print(f"Chunk count: {len(chunks)}")
    print(f"Document matrix shape: " f"{document_vectors.shape}")
    print(f"Query vector shape: " f"{query_vector.shape}")
    print(f"Document dtype: " f"{document_vectors.dtype}")
    print(f"Query dtype: " f"{query_vector.dtype}")

    print()
    print("Vector norms:")

    for index, vector in enumerate(document_vectors):
        norm = np.linalg.norm(vector)

        print(f"Chunk {index}: " f"{norm:.6f}")

    query_norm = np.linalg.norm(query_vector)

    print(f"Query: {query_norm:.6f}")

    print()
    print("First 8 query vector values:")
    print(query_vector[:8])

    # 这里只是提前观察相似度结果。
    # 下一轮会把它正式封装到 VectorStore 中。
    scores = document_vectors @ query_vector

    print()
    print("=" * 70)
    print(f"Query: {query}")
    print()

    for chunk, score in zip(
        chunks,
        scores,
        strict=True,
    ):
        print("-" * 70)
        print(f"Chunk index: {chunk.index}")
        print(f"Score: {float(score):.6f}")
        print(f"Source: {chunk.source}")
        print()
        print(chunk.content)


if __name__ == "__main__":
    main()
