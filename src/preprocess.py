"""
preprocess.py
-------------
Phase 1: Extract and clean text from the eBay User Agreement PDF.

What this file does:
1. Reads the PDF from the data/ folder
2. Extracts all text page by page
3. Cleans the text (removes junk, extra spaces, etc.)
4. Saves cleaned text to data/cleaned_text.txt
"""

import re
import os
from pypdf import PdfReader




PDF_PATH = os.path.join("data", "ebay_user_agreement.pdf")
OUTPUT_PATH = os.path.join("data", "cleaned_text.txt")




def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Opens a PDF and extracts raw text from every page.
    Returns one big string with all the text.
    """
    reader = PdfReader(pdf_path)
    pages_text = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages_text.append(text)
        else:
            print(f"  [Warning] Page {page_num + 1} returned no text.")

    full_text = "\n".join(pages_text)
    print(f"Extracted text from {len(reader.pages)} pages.")
    return full_text


def clean_text(raw_text: str) -> str:
    """
    Cleans raw extracted PDF text:
    - Removes non-ASCII characters
    - Collapses multiple blank lines
    - Removes excessive whitespace
    - Strips leading/trailing spaces per line
    """

    # Remove non-ASCII characters (like weird PDF symbols)
    text = raw_text.encode("ascii", errors="ignore").decode("ascii")

    # Replace multiple spaces with a single space
    text = re.sub(r" {2,}", " ", text)

    # Replace Windows line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse 3+ blank lines into 2 blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip trailing spaces from each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Strip overall leading/trailing whitespace
    text = text.strip()

    print(f"Text cleaned. Total characters: {len(text)}")
    return text


def save_text(text: str, output_path: str) -> None:
    """
    Saves the cleaned text string to a .txt file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f" Cleaned text saved to: {output_path}")


#main

def run_preprocessing():
    """
    Full preprocessing pipeline:
    PDF → Extract → Clean → Save
    """
    print("\n=== Phase 1: Preprocessing ===\n")

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"PDF not found at '{PDF_PATH}'.\n"
            "Please place your eBay User Agreement PDF in the data/ folder "
            "and name it 'ebay_user_agreement.pdf'."
        )

    raw_text = extract_text_from_pdf(PDF_PATH)
    cleaned = clean_text(raw_text)
    save_text(cleaned, OUTPUT_PATH)

    print("\n[✓] Phase 1 complete. Ready for chunking.\n")
    return cleaned


if __name__ == "__main__":
    run_preprocessing()
