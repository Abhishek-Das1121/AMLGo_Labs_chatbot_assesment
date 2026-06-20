"""
pdf_preprocessor.py
-----------------
Handles any uploaded PDF (not just eBay).

What this file does:
1. Computes an MD5 hash of the uploaded file so we know if it's a NEW
   document or one we've already indexed (rerender protection).
2. Extracts text PAGE BY PAGE (so we can attach page numbers to chunks
   later — this is what powers "Source: page 14").
3. Cleans the text per page.
4. Returns a list of {page_number, text} dicts ready for chunking.
"""

import re
import hashlib
from pathlib import Path
from pypdf import PdfReader


UPLOADS_DIR = Path(__file__).resolve().parent.parent / "uploads"


def compute_pdf_hash(file_bytes: bytes) -> str:
    """
    Compute MD5 hash of the raw file bytes.

    Why hash instead of filename?
    Two different files can share a name (e.g. user re-uploads an edited
    'policy.pdf'). Hashing the actual content guarantees we only skip
    re-indexing when the document is *truly* unchanged.
    """
    return hashlib.md5(file_bytes).hexdigest()


def save_uploaded_pdf(file_bytes: bytes, filename: str, pdf_hash: str) -> Path:
    """
    Save the uploaded PDF to disk under uploads/, named by its hash.
    Using the hash as the filename means duplicate uploads never collide
    and we can always find the right file again for the PDF viewer.
    """
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"{pdf_hash}_{filename}"
    save_path = UPLOADS_DIR / safe_name
    if not save_path.exists():
        save_path.write_bytes(file_bytes)
    return save_path


def clean_text(raw_text: str) -> str:
    """
    Clean text extracted from a single page.
    Same cleaning logic as the original preprocess.py, applied per-page
    instead of to the whole document (so we never lose page boundaries).
    """
    text = raw_text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"-\n", "", text)               # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)            # collapse repeated spaces

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if re.fullmatch(r"\d{1,3}", line) or len(line) < 3:
            continue
        lines.append(line)

    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return cleaned.strip()


def extract_pages(pdf_path: Path) -> list[dict]:
    """
    Extract and clean text from every page, preserving page numbers.

    Returns:
        [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]

    Pages with no extractable text (e.g. blank or scanned images) are
    skipped automatically.
    """
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        raw = page.extract_text()
        if not raw:
            continue
        cleaned = clean_text(raw)
        if cleaned:
            pages.append({"page": page_num, "text": cleaned})

    print(f"[pdf_processor] Extracted {len(pages)} non-empty pages "
          f"out of {len(reader.pages)} total.")
    return pages, len(reader.pages)


def process_pdf(file_bytes: bytes, filename: str) -> dict:
    """
    Main entry point — call this once per uploaded file.

    Returns:
        {
            "pdf_hash": "...",
            "filename": "...",
            "saved_path": Path(...),
            "pages": [{"page": 1, "text": "..."}, ...],
            "total_pages": int,
        }
    """
    pdf_hash = compute_pdf_hash(file_bytes)
    saved_path = save_uploaded_pdf(file_bytes, filename, pdf_hash)
    pages, total_pages = extract_pages(saved_path)

    return {
        "pdf_hash"   : pdf_hash,
        "filename"   : filename,
        "saved_path" : saved_path,
        "pages"      : pages,
        "total_pages": total_pages,
    }