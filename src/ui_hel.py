"""
ui_helpers.py
-------------
Reusable Streamlit UI components for DocuBuddy.

Contains:
- PDF viewer
- Page jump buttons
- Indexing status
- Query status
- Evidence cards
"""

from pathlib import Path

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer



# PDF Viewer

def render_pdf_viewer(pdf_path: Path, page: int = 1):
    """
    Render PDF and keep the current page highlighted.
    """

    if not pdf_path.exists():
        st.warning("PDF file not found.")
        return

    page = max(1, page)

    st.info(f"📍 Current page: {page}")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    pdf_viewer(
        input=pdf_bytes,
        width="100%",
        height=850,
        key=f"pdf_viewer_page_{st.session_state.viewer_key}",
    )



# Page Navigation


def render_page_jump_button(
        page: int,
        key_suffix: str = "",
):
    """
    Render source-page button.
    """

    return st.button(
        f"📄 View Source — Page {page}",
        key=f"jump_{page}_{key_suffix}",
    )


# Indexing Status

def show_indexing_status(step: str):

    icons = {
        "reading": "📄 Reading PDF...",
        "extracting": "✅ Extracting text...",
        "chunking": "✂️ Creating chunks...",
        "embedding": "🧠 Generating embeddings...",
        "storing": "🗂️ Building vector database...",
        "done": "✅ DocuBuddy is ready!",
    }

    st.write(icons.get(step, step))


# Query Status

def show_query_status(step: str):

    icons = {
        "retrieving": "🔍 Retrieving relevant chunks...",
        "generating": "🤖 Generating answer...",
        "evidence": "📚 Collecting evidence...",
        "done": "✅ Response ready.",
    }

    st.write(icons.get(step, step))


# Evidence Card

def render_evidence_card(
        chunk_text_html: str,
        source: str,
        page: int,
        score: float,
):
    """
    Render evidence block.
    """

    if score >= 0.75:
        score_icon = "🟢"
    elif score >= 0.50:
        score_icon = "🟡"
    else:
        score_icon = "🔴"

    st.markdown(
        f"""
        <div
            style="
                background:#1E293B;
                padding:14px;
                border-radius:10px;
                border-left:4px solid #4A6FA5;
                margin-bottom:12px;
            "
        >

            <div
                style="
                    color:#E2E8F0;
                    line-height:1.6;
                    font-size:0.90em;
                "
            >
                {chunk_text_html}
            </div>

            <hr>

            <div
                style="
                    color:#94A3B8;
                    font-size:0.82em;
                "
            >
                {score_icon}
                <b>{source}</b>
                • Page {page}
                • Relevance: {score:.0%}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )