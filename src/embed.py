"""
embed.py
---------
Handles all embedding operations.

We explicitly embed documents and queries ourselves instead of letting
Chroma do it internally. This gives better retrieval quality with BGE.
"""

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

_model_cache = None


def get_model():
    global _model_cache

    if _model_cache is None:
        print(f"[embedder] Loading {EMBEDDING_MODEL} on CPU...")
        _model_cache = SentenceTransformer(
            EMBEDDING_MODEL,
            device="cpu"
        )

    return _model_cache


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Embed document chunks.
    """
    model = get_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    Embed a query using the BGE instruction prefix.
    """
    model = get_model()

    query = (
        "Represent this sentence for searching relevant passages: "
        + query
    )

    embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    return embedding.tolist()