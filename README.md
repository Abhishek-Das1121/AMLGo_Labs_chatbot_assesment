# 🤖 eBay Agreement RAG Chatbot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Local%20Vector%20DB-orange?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-LLaMA%203.1-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**A production-grade Retrieval-Augmented Generation (RAG) chatbot built for the Amlgo Labs AI Engineer Assessment.**  
Ask any question about the eBay User Agreement and get grounded, cited, streamed answers — powered by semantic search and a cloud-hosted LLM.

### 🌐 Live Demo

https://abhishek-amlgolabs-docs-chatbot1121.streamlit.app/

### 📂 GitHub Repository

https://github.com/Abhishek-Das1121/AMLGo_Labs_chatbot_assesment

[🚀 Quick Start](#-quick-start) · [🏗️ Architecture](#️-architecture) · [📁 Project Structure](#-project-structure) · [💡 Features](#-features) · [🧠 Tech Decisions](#-technology-decisions)

</div>

---

##  The Problem

Legal documents like the eBay User Agreement (~10,500 words, 20 pages) are dense, complex, and difficult for users to navigate. Finding a specific clause about **arbitration**, **seller fees**, **return policies**, or **liability limits** typically requires reading the entire document.

**The challenge:** Build an AI chatbot that can answer natural language questions about this document — accurately, instantly, and without hallucinating information that isn't there.

**The hard constraints:**
- Must run on an **8GB RAM laptop with integrated GPU only** (no cloud GPU, no fine-tuning)
- Must stream answers in real time
- Must display which document passages were used
- Must refuse to answer from outside the document

---

##  Our Approach

Rather than fine-tuning a large language model (expensive, hardware-intensive), we use **Retrieval-Augmented Generation (RAG)** a technique that separates *finding information* from *generating answers*.

```
Instead of:  "Train the LLM to memorize the document"
We do:       "Find the relevant parts of the document, then ask the LLM to answer from them"
```

This gives us:
-  No GPU required
-  Grounded, verifiable answers
-  Source citations built-in
-  Easy to update if the document changes
-  Fast inference via Groq's cloud API

---

##  Architecture

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        INDEXING PIPELINE                        │
│                        (runs once only)                         │
│                                                                 │
│     PDF Document                                                │
│       │                                                         │
│       ▼                                                         │
│     Text Extraction       (pypdf)                               │
│       │                                                         │
│       ▼                                                         │
│     Text Cleaning         (regex, normalisation)                │
│       │                                                         │
│       ▼                                                         │
│     Sentence-Aware         (RecursiveCharacterTextSplitter)     │
│     Chunking              1200 chars / 200 overlap              │
│       │                                                         │
│       ▼                                                         │
│    Embedding Generation   (BAAI/bge-small-en-v1.5, CPU)         │
│       │                                                         │
│       ▼                                                         │
│    Vector Storage          (ChromaDB, local persistence)        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE                           │
│                     (runs on every question)                    │
│                                                                 │
│     User Question                                               │
│       │                                                         │
│       ▼                                                         │
│     Query Embedding       (same model as indexing)              │
│       │                                                         │
│       ▼                                                         │
│     Semantic Search       (cosine similarity, top-k=4)          │
│       │                                                         │
│       ▼                                                         │
│     Prompt Construction   (retrieved chunks + question)         │
│       │                                                         │
│       ▼                                                         │
│     LLM Generation        (Llama 3.1 8B via Groq API)           │
│       │                                                         │
│       ▼                                                         │
│    Streamed Response     (token-by-token via st.write_stream)   │
│       │                                                         │
│       ▼                                                         │
│      Streamlit UI          (answer + sources + scores)          │
└─────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | File | Responsibility |
|---|---|---|
| **Preprocessor** | `src/preprocess.py` | PDF → clean UTF-8 text |
| **Chunker** | `src/chunking.py` | Text → overlapping sentence-aware chunks |
| **Embedder** | `src/embed.py` | Chunks → vectors → ChromaDB |
| **Retriever** | `src/retriever.py` | Query → top-K chunks + similarity scores |
| **Generator** | `src/generator.py` | Chunks + query → streaming LLM answer |
| **Pipeline** | `src/rag_pipe.py` | Ties all components into one interface |
| **UI** | `app.py` | Streamlit chat interface |

---

##  Features

-  **Semantic Search** — finds relevant passages even if you don't use the exact document wording
-  **Real-Time Streaming** — see answers generated token-by-token, not after a long wait
-  **Source Citations** — every answer shows which document passages were used
-  **Similarity Scores** — colour-coded relevance scores (🟢 high / 🟡 medium / 🔴 low) for each retrieved passage
-  **Hallucination Guard** — system prompt forces the LLM to refuse questions it can't answer from the document
-  **Clean UI** — chat history, sidebar system info, and a one-click clear button
-  **Persistent Vector DB** — ChromaDB is built once and reused across sessions

---

##  Technology Decisions

Every choice was driven by the **8GB RAM / CPU-only constraint**. Here's why we picked what we picked:

### Embedding Model: `BAAI/bge-small-en-v1.5`

| Why this? | Why not alternatives? |
|---|---|
| Only 33M parameters | `all-mpnet-base-v2` is 4× slower on CPU |
| 384-dim vectors — fast similarity search | `text-embedding-ada-002` costs money per call |
| Top-ranked on MTEB retrieval benchmark | `all-MiniLM-L6-v2` — good fallback but slightly lower quality |
| Runs in ~200ms per batch on CPU | Large models like `bge-large` OOM on 8GB RAM |

### Vector Database: `ChromaDB`

| Why this? | Why not alternatives? |
|---|---|
| Zero-config local persistence | Pinecone requires cloud account + paid tier |
| Native cosine similarity support | Weaviate/Qdrant overkill for one document |
| Runs fully in-process (no server) | FAISS lacks built-in metadata storage |
| Built-in embedding function wrapper | |

### LLM: `Llama 3.1 8B via Groq`

| Why this? | Why not alternatives? |
|---|---|
| ~500 tokens/sec — near-instant streaming | Running Mistral 7B locally needs ~5GB RAM + slow CPU |
| Free tier with generous limits | GPT-4 costs money; overkill for doc QA |
| Native streaming API support | Ollama + local models: too slow on 8GB RAM |
| Instruction-tuned — follows system prompts precisely | |

### Chunking: `RecursiveCharacterTextSplitter` (1200 chars / 200 overlap)

```
Why 1200 characters?
  → ≈ 150–250 words per chunk
  → Fits a complete legal clause without splitting mid-sentence
  → Small enough for precise retrieval, large enough for full context

Why 200-character overlap?
  → Legal text has clauses that span paragraph boundaries
  → Overlap ensures no clause is accidentally split across two chunks
  → Prevents the "edge case" where the answer sits between two chunks

Why RecursiveCharacterTextSplitter?
  → Tries \n\n → \n → ". " → " " in order
  → Respects paragraph and sentence boundaries naturally
  → Much better than naive fixed-character splitting for legal documents
```

### Retrieval: Top-K = 6, Cosine Similarity

```
Why K=6?
  → Enough context for complex legal questions
  → Small enough to stay within LLM context window comfortably
  → Empirically: legal questions rarely need more than 6 passages

Why cosine similarity?
  → Measures semantic direction, not raw magnitude
  → Works well with normalized embeddings from bge-small
  → ChromaDB native support — no extra computation
```

---

## 📁 Project Structure

```
amlgo_labs/
│
├── 📂 data/
│   └── ebay_user_agreement.pdf       ← input document (place here)
│
├── 📂 chunks/
│   ├── cleaned_text.txt              ← generated by preprocess.py
│   └── chunks.json                   ← generated by chunking.py
│
├── 📂 vectordb/                      ← generated by embed.py (ChromaDB)
│
├── 📂 notebooks/                     ← for exploration / analysis
│
├── 📂 src/
│   ├── __init__.py
│   ├── preprocess.py                 ← Phase 1: PDF → clean text
│   ├── chunking.py                   ← Phase 2: text → chunks
│   ├── embed.py                      ← Phase 3: chunks → ChromaDB
│   ├── retriever.py                  ← Phase 4: query → top-k chunks
│   ├── generator.py                  ← Phase 5: chunks → streamed LLM answer
│   └── rag_pipe.py                   ← Phase 6: unified pipeline interface
│
├── app.py                            ← Phase 7: Streamlit UI
├── .env                              ← your GROQ_API_KEY goes here
├── pyproject.toml                    ← dependencies
├── uv.lock
└── README.md
```

---

##  Quick Start

### Prerequisites
- Python 3.11+
- `uv` package manager (or `pip`)
- Free [Groq API key](https://console.groq.com/)

### Step 1 — Clone the repository
```bash
git clone https://github.com/yourusername/amlgo-labs-rag.git
cd amlgo-labs-rag
```

### Step 2 — Place the document
```
Put ebay_user_agreement.pdf inside the data/ folder.
```

### Step 3 — Add your API key
```bash
# Open .env and replace the placeholder:
GROQ_API_KEY=your_actual_key_here
```

### Step 4 — Install dependencies
```bash
uv sync
# or
pip install -r requirements.txt
```

### Step 5 — Build the vector database (one-time setup)
```bash
# Run these three commands in order:
python -m src.preprocess    # PDF → cleaned text
python -m src.chunking      # text → chunks
python -m src.embed         # chunks → ChromaDB
```

You'll see progress logs. The whole setup takes ~1–2 minutes.

### Step 6 — Launch the app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501** 🎉

---

##  Example Questions to Try

```
"What is the return policy for buyers?"
"How does the arbitration process work?"
"What are eBay's limitations of liability?"
"Can I sell recalled products on eBay?"
"What happens if a seller fails performance standards?"
"Who handles vehicle transactions on eBay?"
"What is the eBay Money Back Guarantee?"
"Can I transfer my eBay account to someone else?"
```

---

##  Prompt Design

The system prompt is the key to preventing hallucination:

```
You are a precise document question-answering assistant.
Your ONLY job is to answer questions using the provided context
from the eBay User Agreement.

Rules:
1. Answer ONLY using the provided context.
2. If the answer is not in the context, say:
   "I could not find this information in the provided document."
3. Be concise and direct.
4. Do not speculate or add outside knowledge.
```

This ensures the LLM cannot invent information — if the answer isn't in the retrieved chunks, it says so.

---

##  Hardware Compatibility

Designed and tested for:

| Spec | Value |
|---|---|
| RAM | 8GB |
| GPU | None (CPU only) |
| OS | Windows / Linux / macOS |
| Python | 3.11 |

---

##  Acknowledgements

- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) — embedding model
- [ChromaDB](https://www.trychroma.com/) — vector database
- [Groq](https://groq.com/) — LLM inference
- [LangChain](https://www.langchain.com/) — text splitting utilities
- [Streamlit](https://streamlit.io/) — UI framework

---

## Images

<img width="1920" height="1027" alt="Screenshot (469)" src="https://github.com/user-attachments/assets/4a984c5e-a7f0-45dd-aa70-4c3d3fba74f0" />
<img width="1920" height="1011" alt="Screenshot (474)" src="https://github.com/user-attachments/assets/ae5fc1d6-5411-4af0-a0ef-73417dd72568" />
<img width="1920" height="1080" alt="Screenshot (475)" src="https://github.com/user-attachments/assets/19278389-a34a-4178-93a7-ac1c2c475c67" />



