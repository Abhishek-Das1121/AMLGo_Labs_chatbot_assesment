"""
chunking.py
-----------
Section-aware chunking.

Pipeline:
Page
↓
Split into semantic sections 
↓
Only split large sections further with RecursiveCharacterTextSplitter
↓
Preserve page metadata

Benefits:
- Better retrieval quality
- Cleaner embeddings
- Same latency
- Same metadata format
- Works for arbitrary PDFs
"""

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    length_function=len,
)


def split_into_sections(text: str) -> list[str]:
    """
    Split text on numbered headings like:

    1. Introduction
    2. About eBay
    3. Using eBay

    Keeps the heading attached to its section.
    """

    pattern = r"\n(?=\d+\.\s+[A-Z])"

    sections = re.split(pattern, text)

    return [section.strip() for section in sections if section.strip()]


def chunk_pages(pages: list[dict], source_filename: str) -> list[dict]:
    """
    Create semantic chunks while preserving page metadata.

    Returns:
    [
        {
            "chunk_id": 0,
            "text": "...",
            "page": 1,
            "source": "...",
            "word_count": 180
        }
    ]
    """

    chunks = []
    chunk_id = 0


    for page_entry in pages:


        page_num = page_entry["page"]
        page_text = page_entry["text"]

        # Skip very small pages
        if len(page_text.split()) < 5:
            continue

        # -------- Stage 1: section split --------
        sections = split_into_sections(page_text)

        # -------- Stage 2: split large sections --------
        for section in sections:

            if len(section) <= CHUNK_SIZE:
                section_chunks = [section]
            else:
                section_chunks = _splitter.split_text(section)

            # -------- Stage 3: process final chunks --------
            for chunk_text in section_chunks:


                chunk_text = chunk_text.strip()


                if not chunk_text:
                    continue


                # Ignore tiny chunks (titles etc.)
                if len(chunk_text.split()) < 20:
                    continue


                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "page": page_num,
                        "source": source_filename,
                        "word_count": len(chunk_text.split()),
                    }
                )

                chunk_id += 1

    print(
        f"[chunking] Created {len(chunks)} chunks across "
        f"{len(pages)} pages for '{source_filename}'."
    )

    return chunks
