"""
retriever.py
<<<<<<< HEAD

Phase 4: Retrieve the most relevant chunks for a user query.
=======
------------
Retrieves the most relevant chunks for a user query.
>>>>>>> f2d8605 (Release DocuBuddy V2 with improved RAG pipeline and UI)

Improvements:
- Handles short queries better.
- Slightly increases recall (TOP_K = 8).
- Prevents too many chunks from the same page.
- Preserves page/source metadata for citations.
"""

from dataclasses import dataclass

from src.vector_store import query_collection

TOP_K = 8
MAX_CHUNKS_PER_PAGE = 2


@dataclass
class RetrievedChunk:
    """One retrieved chunk with everything the UI needs."""
    chunk_id: int
    text: str
    similarity_score: float
    page: int
    source: str
    word_count: int
    rank: int


def preprocess_query(query: str) -> str:
    """
    Improve very short queries without affecting normal ones.
    """

    query = query.strip()

    if len(query.split()) < 3:
        query = f"Information about {query}"

    return query


def retrieve(
    pdf_hash: str,
    query: str,
    k: int = TOP_K,
) -> list[RetrievedChunk]:
    """
    Retrieve top-k chunks for one document.
    """

    query = preprocess_query(query)

    results = query_collection(
        pdf_hash=pdf_hash,
        query=query,
        k=k,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    chunks = []
    page_counts = {}

    rank = 1

    for doc, meta, dist in zip(documents, metadatas, distances):

        page = int(meta["page"])

        if page_counts.get(page, 0) >= MAX_CHUNKS_PER_PAGE:
            continue

        page_counts[page] = page_counts.get(page, 0) + 1

        similarity = max(
            0.0,
            min(
                1.0,
                1.0 - dist,
            ),
        )

        chunks.append(
            RetrievedChunk(
                chunk_id=int(meta["chunk_id"]),
                text=doc,
                similarity_score=round(similarity, 4),
                page=page,
                source=meta["source"],
                word_count=int(meta["word_count"]),
                rank=rank,
            )
        )

        rank += 1

<<<<<<< HEAD
        for i in range(len(results["documents"][0])):

            distance = results["distances"][0][i]

            # Convert distance → cosine similarity
            similarity = round(1 - distance, 4)

            metadata = results["metadatas"][0][i]

            # Hallucination guardrail
            if similarity >= MIN_SIMILARITY:

                retrieved_chunks.append(
                    {
                        "chunk_id": metadata["chunk_id"],
                        "text": results["documents"][0][i],
                        "similarity_score": similarity,
                        "word_count": metadata.get("word_count", 0),
                    }
                )

        # Sort after collecting ALL chunks
        retrieved_chunks.sort(
            key=lambda x: x["similarity_score"],
            reverse=True,
        )

        return retrieved_chunks

    def format_context(self, retrieved_chunks: list[dict]) -> str:
        """
        Formats retrieved chunks into a single context string
        for injection into the prompt.
        """

        context_parts = []

        for i, chunk in enumerate(retrieved_chunks, start=1):
            context_parts.append(
                f"[Source {i}]\n{chunk['text']}"
            )

        return "\n\n".join(context_parts)




if __name__ == "__main__":

    print("\nRetriever Test\n")

    retriever = Retriever()

    test_queries = [
        "Can users opt out of arbitration?",
        "What is the eBay Money Back Guarantee?",
        "Can eBay terminate my account?",
        "What happens if a seller does not ship an item?",
    ]

    for query in test_queries:

        print(f"\nQuery: {query}")

        results = retriever.retrieve(query)

        print(f"Retrieved {len(results)} chunks")

        for chunk in results:
            print(
                f"  [Chunk {chunk['chunk_id']}] "
                f"Score: {chunk['similarity_score']:.4f} | "
                f"{chunk['text'][:100]}..."
            )

    print("\nRetriever test complete.\n")
=======
    return chunks
>>>>>>> f2d8605 (Release DocuBuddy V2 with improved RAG pipeline and UI)
