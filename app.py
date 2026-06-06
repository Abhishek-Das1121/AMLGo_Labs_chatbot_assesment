"""
app.py
------
Streamlit chatbot interface.

Features:
- Chat interface with streaming responses
- Source chunk display with similarity scores
- Sidebar with system information
- Clear chat button
"""

import streamlit as st

from src.rag_pipe import RAGPipeline




st.set_page_config(
    page_title="eBay Agreement Chatbot",
    page_icon="🤖",
    layout="wide",
)




@st.cache_resource(show_spinner="Loading RAG pipeline...")
def load_pipeline():
    return RAGPipeline()


pipeline = load_pipeline()



def render_sources(chunks):
    """
    Display retrieved source chunks.
    """

    if not chunks:
        return

    with st.expander("📚 Retrieved Source Passages", expanded=False):

        for chunk in chunks:

            similarity = chunk["similarity_score"]

            if similarity >= 0.75:
                score_icon = "🟢"
            elif similarity >= 0.65:
                score_icon = "🟡"
            else:
                score_icon = "🔴"

            st.markdown(
                f"""
**{score_icon} Chunk {chunk["chunk_id"]}**

- Similarity: `{similarity:.2%}`
- Word Count: `{chunk["word_count"]}`
"""
            )

            st.markdown(
                f"""
<div style="
padding:10px;
border-radius:8px;
background-color:#f4f4f4;
font-size:0.9em;
">
{chunk["text"]}
</div>
""",
                unsafe_allow_html=True,
            )

            st.markdown("---")


#sidebar


with st.sidebar:

    st.title("⚙️ System Information")

    st.markdown("---")

    st.markdown("### 🤖 LLM")
    st.code(pipeline.generator.model)

    st.markdown("### 🔤 Embedding Model")
    st.code(pipeline.retriever.model_name)

    st.markdown("### 🗄️ Vector Database")
    st.code("ChromaDB")

    st.markdown("### 📄 Indexed Chunks")
    st.metric(
        label="Total Chunks",
        value=pipeline.retriever.collection.count()
    )

    st.markdown("---")

    st.markdown(
        """
### About

This chatbot answers questions using only the
**eBay User Agreement** document.

Architecture:

PDF → Chunks → BGE Embeddings →
ChromaDB → Retrieval → Groq Llama 3
"""
    )

    st.markdown("---")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# MAIN 


st.title("🤖 eBay User Agreement Chatbot")

st.markdown(
    """
Ask questions about the eBay User Agreement.

Answers are generated using Retrieval-Augmented
Generation (RAG) and grounded strictly in the document.
"""
)

st.markdown("---")



# Chat history


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and "chunks" in message
        ):
            render_sources(message["chunks"])


# input


prompt = st.chat_input(
    "Ask about the eBay User Agreement..."
)

if prompt:

    # User message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant message

    with st.chat_message("assistant"):

        try:

            chunks, stream = pipeline.query(prompt)

            full_answer = st.write_stream(stream)

            render_sources(chunks)

        except Exception as e:

            full_answer = f"❌ Error: {str(e)}"

            st.error(full_answer)

            chunks = []

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_answer,
            "chunks": chunks,
        }
    )
