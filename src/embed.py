"""
embed.py
--------
Generating embeddings and store them in ChromaDB.

What this file does:
1. Loads chunks from data/chunks.json
2. Loads the embedding model (BAAI/bge-small-en-v1.5)
3. Generates vector embeddings for every chunk
4. Stores embeddings + metadata in a persistent ChromaDB collection
"""

import os
import json
import chromadb
from sentence_transformers import SentenceTransformer




CHUNKS_PATH = os.path.join("data", "chunks.json")
VECTORDB_PATH = os.path.join("vectordb")
COLLECTION_NAME = "ebay_user_agreement"

# Primary model: fast, lightweight, great for legal retrieval

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"



# function



def load_chunks(path: str) -> list[dict]:
    """
    Loads chunks from the JSON file created by chunking.py
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Chunks not found at '{path}'.\n"
            "Run chunking.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks.")
    return chunks


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Loads the sentence-transformer embedding model.
    First run will download the model (~33MB for bge-small).
    Subsequent runs will use the cached version.
    """
    print(f"Loading embedding model: {model_name}")
    print("    (First run downloads the model. This may take 1-2 minutes.)")
    model = SentenceTransformer(model_name)
    print(f"Embedding model loaded.")
    return model


def generate_embeddings(
    chunks: list[dict],
    model: SentenceTransformer
) -> list[list[float]]:
    """
    Generates an embedding vector for each chunk's text.
    Returns a list of embedding vectors (list of floats).

    BGE models work best with a query instruction prefix,
    but for document indexing we use plain text.
    """
    texts = [chunk["text"] for chunk in chunks]

    print(f"[→] Generating embeddings for {len(texts)} chunks...")
    print("    (This runs on CPU. May take 30-60 seconds.)")

    embeddings = model.encode(
        texts,
        batch_size=32,        # Process 32 chunks at a time to save RAM
        show_progress_bar=True,
        normalize_embeddings=True,  # Needed for cosine similarity
    )

    print(f"Embeddings generated. Shape: {embeddings.shape}")
    return embeddings.tolist()


def store_in_chromadb(
    chunks: list[dict],
    embeddings: list[list[float]],
    vectordb_path: str,
    collection_name: str,
) -> chromadb.Collection:
    """
    Creates (or resets) a ChromaDB collection and stores:
    - The embedding vectors
    - The chunk text (as documents)
    - Metadata: chunk_id, word_count, char_count
    """
    # Create persistent ChromaDB client
    client = chromadb.PersistentClient(path=vectordb_path)

    # Delete existing collection if it exists (fresh rebuild)
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)
        print(f"[→] Deleted existing collection '{collection_name}'.")

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # Use cosine similarity
    )

    # Prepare data for ChromaDB
    ids = [str(chunk["chunk_id"]) for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "chunk_id": chunk["chunk_id"],
            "word_count": chunk["word_count"],
            "char_count": chunk["char_count"],
        }
        for chunk in chunks
    ]

    # Insert in one batch
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"Stored {collection.count()} chunks in ChromaDB.")
    print(f"    Collection: '{collection_name}'")
    print(f"    Path: {vectordb_path}")
    return collection


def verify_collection(
    collection: chromadb.Collection,
    model: SentenceTransformer
) -> None:
    """
    Quick sanity check using the SAME embedding model
    that was used for indexing.
    """
    test_query = "What are seller obligations?"

    print(f"\n[→] Running verification query: '{test_query}'")

    query_embedding = model.encode(
        test_query,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2,
    )

    print("Verification passed. Sample result:")

    if results["documents"]:
        print(
            "    "
            + results["documents"][0][0][:150]
            + "..."
        )


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def run_embedding() -> chromadb.Collection:
    """
    Full embedding pipeline:
    chunks.json → embeddings → ChromaDB
    """
    print("\n Embedding + Vector Storage ===\n")

    chunks = load_chunks(CHUNKS_PATH)
    model = load_embedding_model(EMBEDDING_MODEL_NAME)
    embeddings = generate_embeddings(chunks, model)

    collection = store_in_chromadb(
        chunks=chunks,
        embeddings=embeddings,
        vectordb_path=VECTORDB_PATH,
        collection_name=COLLECTION_NAME,
    )

    verify_collection(collection, model)

    print("\n Vector database is ready.\n")
    return collection


if __name__ == "__main__":
    run_embedding()
