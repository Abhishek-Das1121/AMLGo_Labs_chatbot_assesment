"""
vector_store.py
----------------
Owns all ChromaDB interaction.

Architecture:
PDF chunks

to

embed_documents()

to

store embeddings in Chroma

to

embed_query()

tp

query_embeddings

to

similarity search

Each PDF gets its own collection named after its hash.
"""

from pathlib import Path

import chromadb

from src.embed import embed_documents, embed_query

CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
BATCH_SIZE = 32

_client_cache = None


def _get_client():
    """
    Singleton persistent ChromaDB client.
    """
    global _client_cache

    if _client_cache is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client_cache = chromadb.PersistentClient(path=str(CHROMA_DIR))

    return _client_cache


def _collection_name(pdf_hash: str) -> str:
    """
    Collection name for one document.
    """
    return f"doc_{pdf_hash}"


def collection_exists(pdf_hash: str) -> bool:
    """
    Check whether this document has already been indexed.
    """
    client = _get_client()
    existing = [c.name for c in client.list_collections()]

    return _collection_name(pdf_hash) in existing


def build_collection(pdf_hash: str, chunks: list[dict]):
    """
    Build a NEW collection for one document.

    Embeddings are generated explicitly instead of using
    Chroma's embedding_function.
    """
    client = _get_client()

    name = _collection_name(pdf_hash)

    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )

    documents = [c["text"] for c in chunks]

    embeddings = embed_documents(documents)

    ids = [str(c["chunk_id"]) for c in chunks]

    metadatas = [
        {
            "chunk_id": c["chunk_id"],
            "page": c["page"],
            "source": c["source"],
            "word_count": c["word_count"],
        }
        for c in chunks
    ]

    total = len(chunks)

    for start in range(0, total, BATCH_SIZE):

        end = min(start + BATCH_SIZE, total)

        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    print(
        f"[vector_store] Indexed {collection.count()} chunks "
        f"into collection '{name}'."
    )

    return collection


def get_collection(pdf_hash: str):
    """
    Return an existing collection.
    """
    client = _get_client()

    return client.get_collection(
        name=_collection_name(pdf_hash)
    )


def query_collection(
    pdf_hash: str,
    query: str,
    k: int = 4,
) -> dict:
    """
    Perform similarity search using explicit query embeddings.
    """
    collection = get_collection(pdf_hash)

    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    return results


def delete_collection(pdf_hash: str):
    """
    Delete one document collection.
    Useful during debugging.
    """
    client = _get_client()

    client.delete_collection(
        name=_collection_name(pdf_hash)
    )

    print(
        f"[vector_store] Deleted collection "
        f"'{_collection_name(pdf_hash)}'"
    )