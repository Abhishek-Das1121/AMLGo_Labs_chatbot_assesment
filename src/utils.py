"""
utils.py
--------
Small helper functions:

- Conversation history formatting
- Session reset helpers
- Citation formatting
- Evidence display helpers

Intentionally simple.
"""

import html


# Conversation History


def get_recent_history(messages: list[dict], max_turns: int = 3) -> str:
    """
    Format the last few chat turns for follow-up questions.
    """

    if not messages:
        return ""

    recent = messages[-(max_turns * 2):]

    lines = []

    for msg in recent:
        role = msg["role"].capitalize()
        lines.append(f"{role}: {msg['content']}")

    return "\n".join(lines)



# Session Helpers


def reset_session(session_state):
    """
    Clear chat messages while keeping the current PDF indexed.
    """

    session_state["messages"] = []


def clear_document(session_state):
    """
    Full reset when switching PDFs.
    """

    session_state["messages"] = []
    session_state["active_pdf"] = None
    session_state["pdf_filename"] = None
    session_state["pdf_hash"] = None
    session_state["vector_ready"] = False
    session_state["current_page"] = 1



# Citations


def format_citation(source: str, page: int) -> str:
    """
    Format source information for display.
    """

    return f"{source} · Page {page}"


def extract_source_metadata(chunk) -> dict:
    """
    Extract metadata from RetrievedChunk.
    """

    return {
        "chunk_id": chunk.chunk_id,
        "source": chunk.source,
        "page": chunk.page,
        "score": chunk.similarity_score,
    }


 
# Evidence Display
 

def highlight_answer(evidence: str, answer: str) -> str:
    """
    Escape evidence safely.

    Placeholder for future semantic highlighting.
    For V2 we simply display the whole evidence chunk.
    """

    return html.escape(evidence)