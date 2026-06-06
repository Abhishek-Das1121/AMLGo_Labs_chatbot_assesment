"""
retriever.py
------------
Phase 4: Retrieve the most relevant chunks for a user query.

What this file does:
1. Loads the embedding model
2. Connects to the existing ChromaDB collection
3. Embeds the user's query
4. Returns the top-K most similar chunks with similarity scores
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

VECTORDB_PATH = os.path.join("vectordb")
COLLECTION_NAME = "ebay_user_agreement"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

TOP_K = 6
MIN_SIMILARITY = 0.65


# ──────────────────────────────────────────────
# RETRIEVER CLASS
# ──────────────────────────────────────────────

class Retriever:
    """
    Handles all retrieval operations:
    - Loads model once
    - Connects to ChromaDB
    - Retrieves top-K chunks with scores
    """

    def __init__(
        self,
        vectordb_path: str = VECTORDB_PATH,
        collection_name: str = COLLECTION_NAME,
        model_name: str = EMBEDDING_MODEL_NAME,
        top_k: int = TOP_K,
    ):
        self.top_k = top_k
        self.model_name = model_name

        print("Loading retriever...")

        # Load embedding model
        self.model = SentenceTransformer(model_name)

        # Connect to ChromaDB
        if not os.path.exists(vectordb_path):
            raise FileNotFoundError(
                f"Vector database not found at '{vectordb_path}'.\n"
                "Run embed.py first to build the vector database."
            )

        self.client = chromadb.PersistentClient(path=vectordb_path)
        self.collection = self.client.get_collection(name=collection_name)

        total_chunks = self.collection.count()
        print(f"Retriever ready. Collection has {total_chunks} chunks.")

    def retrieve(self, query: str) -> list[dict]:
        """
        Retrieves the top-K most relevant chunks for a given query.

        Returns:
        [
            {
                "chunk_id": 3,
                "text": "...",
                "similarity_score": 0.87,
                "word_count": 198,
            }
        ]
        """

        if not query.strip():
            return []

        # BGE query instruction prefix
        prefixed_query = (
            f"Represent this sentence for searching relevant passages: {query}"
        )

        # Embed query
        query_embedding = self.model.encode(
            prefixed_query,
            normalize_embeddings=True,
        ).tolist()

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved_chunks = []

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


# ──────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────

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