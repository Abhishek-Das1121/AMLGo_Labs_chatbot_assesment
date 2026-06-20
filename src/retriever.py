"""
retriever.py
------------
Retrieves the most relevant chunks for a user query.

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

    return chunks
