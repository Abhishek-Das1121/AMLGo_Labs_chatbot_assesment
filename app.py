"""
app.py
------
Enterprise Document Intelligence Assistant V2.

Two-column layout:
  LEFT  -> PDF viewer (jumps to source page on click)
  RIGHT -> Chat interface with evidence panel + citations

Rerender protection: PDF is only processed when its hash changes —
never on chat interactions or button clicks.
"""

import streamlit as st
from src.rag_pipeline import RAGPipeline
from src.utils import (
    format_citation,
    highlight_answer,
    reset_session,
    clear_document,
)
from src.ui_hel import (
    render_pdf_viewer,
    render_evidence_card,
)

st.set_page_config(
    page_title="DocuBuddy",
    page_icon="📄",
    layout="wide",
)

#  Session State Initialisation 
defaults = {
    "messages"    : [],
    "active_pdf"  : None,     # saved file path
    "pdf_filename": None,
    "pdf_hash"    : None,
    "vector_ready": False,
    "current_page": 1,
    "viewer_key":0
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def load_pipeline() -> RAGPipeline:
    return RAGPipeline()


pipeline = load_pipeline()

#  Header 

st.markdown(
"""
<div style='background:#181818;
border:1px solid #242424;
padding:30px;
border-radius:20px;
margin-bottom:25px;
box-shadow:0 4px 18px rgba(0,0,0,0.25);'>

<h1 style='color:white;
margin:0;
font-size:2.3rem;'>
📄 DocuBuddy
</h1>

<p style='color:#C6C6C6;
font-size:1rem;
margin-top:12px;
margin-bottom:0;'>
Upload any PDF • Ask questions • Get grounded answers with citations
</p>

</div>
""",
unsafe_allow_html=True
)
#  Sidebar: Upload + Controls 
with st.sidebar:
    st.markdown("### 📤 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()

        # Only re-process if this is a genuinely new/changed file.
        # index_document() itself also checks the hash, but checking here
        # too avoids re-reading the file unnecessarily on every rerun.
        from src.pdf_preprocess import compute_pdf_hash
        incoming_hash = compute_pdf_hash(file_bytes)

        if incoming_hash != st.session_state.pdf_hash:
            with st.status("Processing document...", expanded=True) as status:
                st.write("📄 Reading PDF...")
                result = pipeline.index_document(file_bytes, uploaded_file.name)

                if result["was_already_indexed"]:
                    st.write("✅ Document already indexed — loading instantly.")
                else:
                    st.write(f"✅ Extracted {result['total_pages']} pages")
                    st.write(f"✅ Created {result['total_chunks']} chunks")
                    st.write("✅ Embeddings stored in ChromaDB")

                status.update(label="✅ Ready!", state="complete")

            st.session_state.active_pdf   = result["saved_path"]
            st.session_state.pdf_filename = result["filename"]
            st.session_state.pdf_hash     = result["pdf_hash"]
            st.session_state.vector_ready = True
            st.session_state.current_page = 1
            st.session_state.messages     = []   # new doc -> fresh conversation

    st.markdown("---")

    if st.session_state.vector_ready:
        st.markdown("### 📋 Active Document")
        st.info(st.session_state.pdf_filename)

        if st.button("🗑️ Clear Chat", use_container_width=True):
            reset_session(st.session_state)
            st.rerun()

        if st.button("📤 Upload Different PDF", use_container_width=True):
            clear_document(st.session_state)
            st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ System Info")
    st.caption("Project : DocuBuddy")
    st.caption("LLM : Llama 3.1 8B Instant (Groq)")
    st.caption("Embeddings : BAAI/bge-small-en-v1.5")
    st.caption("Vector DB : ChromaDB")

#  Main Area 
if not st.session_state.vector_ready:
    st.markdown(
        """
        <div style='text-align:center; padding:60px 20px; color:#718096;'>
            <h3>👈 Upload a PDF to get started</h3>
            <p>Employee handbooks, research papers, policy documents — anything works.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    left_col, right_col = st.columns([1, 1.3])

    #  LEFT: PDF Viewer 
    with left_col:
        st.markdown(
            "<h3 style='margin-bottom:10px;'>📖 Document Viewer</h3>",
            unsafe_allow_html=True
        )
        render_pdf_viewer(
            st.session_state.active_pdf,
            page=st.session_state.current_page,
        )

    #  RIGHT: Chat Interface 
    with right_col:
        st.markdown(
            "<h3 style='margin-bottom:10px;'>💬 Ask About This Document</h3>",
             unsafe_allow_html=True
        )

        chat_container = st.container(height=550)
        with chat_container:
            for msg_idx, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

                    if message["role"] == "assistant" and "chunks" in message:
                        with st.expander("📚 Evidence & Sources", expanded=False):
                            for chunk in message["chunks"]:
                                highlighted = highlight_answer(
                                    chunk.text, message["content"]
                                )
                                render_evidence_card(
                                    highlighted, chunk.source, chunk.page,
                                    chunk.similarity_score,
                                )
                                if st.button(
                                    f"📄 View Source — Page {chunk.page}",
                                    key=f"history_{msg_idx}_{chunk.chunk_id}"
                                ):
                                    st.session_state.current_page = chunk.page
                                    st.session_state.viewer_key += 1
                                    st.rerun()

        if prompt := st.chat_input("Ask a question about the document..."):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    try:
                        chunks, token_stream = pipeline.ask(
                            st.session_state.pdf_hash,
                            prompt,
                            st.session_state.messages[:-1],   # history excludes current Q
                        )
                        full_answer = st.write_stream(token_stream)

                        with st.expander("📚 Evidence & Sources", expanded=True):
                            for chunk in chunks:
                                highlighted = highlight_answer(
                                    chunk.text,
                                    full_answer
                                )

                                render_evidence_card(
                                    highlighted,
                                    chunk.source,
                                    chunk.page,
                                    chunk.similarity_score,
                                )

                                if st.button(
                                    f"📄 View Source — Page {chunk.page}",
                                    key=f"current_{chunk.chunk_id}"
                                ):
                                    st.session_state.current_page = chunk.page
                                    st.session_state.viewer_key += 1
                                    st.rerun()

                    except Exception as e:
                        full_answer = f"❌ Error: {e}"
                        st.error(full_answer)
                        chunks = []

            st.session_state.messages.append({
                "role"   : "assistant",
                "content": full_answer,
                "chunks" : chunks,
            })
            st.rerun()
