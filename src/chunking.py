"""
chunking.py
-----------
Phase 2: Split cleaned text into sentence-aware chunks.

What this file does:
1. Reads the cleaned text from data/cleaned_text.txt
2. Splits it into overlapping chunks (150-250 words each)
3. Saves chunks to data/chunks.json for inspection
4. Returns a list of chunk dicts ready for embedding
"""

import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

CLEANED_TEXT_PATH = os.path.join("data", "cleaned_text.txt")
CHUNKS_OUTPUT_PATH = os.path.join("data", "chunks.json")

# ~1200 characters ≈ 200 words for English legal text
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# Sentence-priority separators:
# RecursiveCharacterTextSplitter tries these in order.
# It first tries to split on paragraph breaks, then sentences,
# then words — preserving legal clause integrity.
SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", " "]


# ──────────────────────────────────────────────
# FUNCTIONS
# ──────────────────────────────────────────────

def load_cleaned_text(path: str) -> str:
    """
    Loads the cleaned text file from disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cleaned text not found at '{path}'.\n"
            "Run preprocess.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded cleaned text intoo({len(text)} characters).")
    return text


def chunk_text(text: str) -> list[dict]:
    """
    Splits text into overlapping sentence-aware chunks.

    Returns a list of dicts:
    [
        {
            "chunk_id": 0,
            "text": "...",
            "char_count": 123,
            "word_count": 45
        },
        ...
    ]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )

    raw_chunks = splitter.split_text(text)

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        word_count = len(chunk_text.split())
        chunks.append({
            "chunk_id": i,
            "text": chunk_text.strip(),
            "char_count": len(chunk_text),
            "word_count": word_count,
        })

    print(f"[✓] Created {len(chunks)} chunks.")
    print(f"    Avg word count per chunk: "
          f"{sum(c['word_count'] for c in chunks) // len(chunks)}")
    return chunks


def save_chunks(chunks: list[dict], output_path: str) -> None:
    """
    Saves chunks to a JSON file for inspection.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Chunks saved in: {output_path}")


def preview_chunks(chunks: list[dict], n: int = 3) -> None:
    """
    Prints the first n chunks for a quick sanity check.
    """
    print(f"\n--- Preview: First {n} chunks ---")
    for chunk in chunks[:n]:
        print(f"\n[Chunk {chunk['chunk_id']}] "
              f"({chunk['word_count']} words, {chunk['char_count']} chars)")
        print(chunk["text"][:200] + "...")
    print("\n--- End Preview ---\n")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def run_chunking() -> list[dict]:
    """
    Full chunking pipeline:
    cleaned_text.txt → chunks → chunks.json
    """
    print("\n=== Phase 2: Chunking ===\n")

    text = load_cleaned_text(CLEANED_TEXT_PATH)
    chunks = chunk_text(text)
    save_chunks(chunks, CHUNKS_OUTPUT_PATH)
    preview_chunks(chunks)

    print("Phase 2 complete. Ready for embedding.\n")
    return chunks


if __name__ == "__main__":
    run_chunking()
