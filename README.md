# 📄 DocuBuddy — Enterprise Document Intelligence Assistant

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Local%20Vector%20DB-orange?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-LLaMA%203.1-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**An AI-powered Retrieval-Augmented Generation (RAG) application that lets you upload any PDF and ask natural language questions and get grounded, cited, streamed answers with the exact source evidence.**

Originally built as a single-document chatbot for the Amlgo Labs AI Engineer Assessment, now evolved into a general-purpose document intelligence assistant.

### 🌐 Live Demo

https://docubuddy-ai.streamlit.app/

### 📂 GitHub Repository

https://github.com/Abhishek-Das1121/DocuBuddy

[🚀 Quick Start](#-quick-start) · [🏗️ Architecture](#-architecture) · [📁 Project Structure](#-project-structure) · [💡 Features](#-features) · [🧠 Tech Decisions](#-technology-decisions) · [✨ What's New](#-whats-new-in-version-2)

</div>

---

## The Problem

Documents like employee handbooks, legal agreements, policies, and research papers are dense and difficult to navigate. Finding a specific clause about **arbitration**, **leave policy**, **refunds**, or **liability limits** typically requires reading the entire document and that problem doesn't go away once you've solved it for one document. It exists for *every* document.

**The challenge:** Build an AI assistant that can answer natural language questions about any uploaded document accurately, instantly, without hallucinating, and while showing exactly where each answer came from.

**The hard constraints:**
- Must run on an **8GB RAM laptop with integrated GPU only** (no cloud GPU, no fine-tuning)
- Must stream answers in real time
- Must display which document passages and page numbers were used
- Must refuse to answer from outside the document
- Must work with *any* PDF, not just one fixed document

---

## Our Approach

Rather than fine-tuning a large language model (expensive, hardware-intensive), we use **Retrieval-Augmented Generation (RAG)** a technique that separates *finding information* from *generating answers*.

```
Instead of:  "Train the LLM to memorize the document"
We do:       "Find the relevant parts of the document, then ask the LLM to answer from them"
```

This gives us:
- No GPU required
- Grounded, verifiable answers
- Source citations with page numbers built-in
- Works with any document, instantly
- Fast inference via Groq's cloud API

---

## Architecture

### Architecture Diagramatic representation

<img width="1536" height="1024" alt="ChatGPT Image Jun 20, 2026, 01_08_47 PM" src="https://github.com/user-attachments/assets/97683a03-50f3-4c32-ace5-d35b77041d23" />


### End-to-End Pipeline

```
                          PDF Upload
                              │
                              ▼
                Page-wise Text Extraction (pypdf)
                              │
                              ▼
                  Cleaning & Normalization
                              │
                              ▼
                  Semantic Chunking (sentence-aware,
                   1200 chars / 200 overlap, page-tagged)
                              │
                              ▼
              BAAI/bge-small-en-v1.5 Embeddings (CPU)
                              │
                              ▼
                   ChromaDB Vector Store
                              │
              ┌───────────────┴───────────────┐
              │    (runs on every question)   │
               ─────────────────────────────── 
                              │
                              ▼                                 
                        User Question                          
                              │                                 
                              ▼                                 
                Query Embedding (same model)            
                              │                                 
                              ▼                                 
              Semantic Search (cosine, top-K)         
                              │                                 
                              ▼                                 
          Conversation History + Prompt Build     
                              │                                 
                              ▼                                 
                Groq Llama 3.1 8B Instant               
                              │                                 
                              ▼                                 
            Streamed, Grounded Answer + Evidence    
                              │                                 
                              ▼
              Streamlit UI (PDF Viewer + Chat +       
                Evidence Panel + Citations)
```

### Component Responsibilities

| Component | File | Responsibility |
|---|---|---|
| **PDF Processor** | `src/pdf_preprocess.py` | PDF → page-tagged clean text + hash check |
| **Chunker** | `src/chunking.py` | Text → overlapping sentence-aware chunks (page-aware) |
| **Embedder** | `src/embed.py` | Chunks → vectors |
| **Vector Store** | `src/vector_store.py` | Embeddings → ChromaDB collections |
| **Retriever** | `src/retriever.py` | Query → top-K chunks + similarity scores |
| **Generator** | `src/generator.py` | Chunks + query + history → streaming LLM answer |
| **Pipeline** | `src/rag_pipeline.py` | Ties all components into one interface |
| **UI Helpers** | `src/ui_hel.py` | PDF viewer, status indicators, page navigation |
| **Utils** | `src/utils.py` | Chat history, citations, text highlighting |
| **UI** | `app.py` | Streamlit two-column interface |

---

## Features

- **Upload Any PDF** : employee handbooks, policies, research papers, agreements, reports, contracts
- **Semantic Search** : finds relevant passages even if you don't use the exact document wording
- **Real-Time Streaming** : see answers generated token-by-token, not after a long wait
- **Evidence & Source Citations** : every answer shows the supporting chunk, document name, page number, and similarity score
- **Built-in PDF Viewer** : view the uploaded PDF inside the app and inspect source pages directly
- **Conversation Memory** : recent chat history is used to understand follow-up questions
- **Hallucination Guard** : system prompt forces the LLM to refuse questions it can't answer from the document
- **Rerender Protection** : documents are only re-processed when the uploaded PDF actually changes (hash-based)
- **Persistent Vector DB** : ChromaDB is built once per document and reused across sessions

---

## Technology Decisions

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
| Native cosine similarity support | Weaviate/Qdrant overkill for single-document workflows |
| Runs fully in-process (no server) | FAISS lacks built-in metadata storage |
| Built-in embedding function wrapper | |

### LLM: `Llama 3.1 8B via Groq`

| Why this? | Why not alternatives? |
|---|---|
| ~500 tokens/sec : near-instant streaming | Running Mistral 7B locally needs ~5GB RAM + slow CPU |
| Free tier with generous limits | GPT-4 costs money; overkill for doc QA |
| Native streaming API support | Ollama + local models: too slow on 8GB RAM |
| Instruction-tuned : follows system prompts precisely | |

### Chunking: `RecursiveCharacterTextSplitter` (1200 chars / 200 overlap, page-aware)

```
Why 1200 characters?
  → ≈ 150–250 words per chunk
  → Fits a complete clause without splitting mid-sentence
  → Small enough for precise retrieval, large enough for full context

Why 200-character overlap?
  → Clauses often span paragraph boundaries
  → Overlap ensures no clause is accidentally split across two chunks
  → Prevents the "edge case" where the answer sits between two chunks

Why page-aware chunking?
  → Each chunk retains its source page number
  → Enables accurate citations and PDF viewer page-jumping

Why RecursiveCharacterTextSplitter?
  → Tries \n\n → \n → ". " → " " in order
  → Respects paragraph and sentence boundaries naturally
  → Much better than naive fixed-character splitting for structured documents
```

### Retrieval: Top-K = 6, Cosine Similarity

```
Why K=6?
  → Enough context for complex questions
  → Small enough to stay within LLM context window comfortably
  → Empirically: most questions rarely need more than 6 passages

Why cosine similarity?
  → Measures semantic direction, not raw magnitude
  → Works well with normalized embeddings from bge-small
  → ChromaDB native support — no extra computation
```

---

## 📁 Project Structure

```
.
├── app.py                            ← Streamlit two-column UI
├── src/
│   ├── pdf_preprocess.py             ← PDF → page-tagged clean text
│   ├── chunking.py                   ← text → page-aware chunks
│   ├── embed.py                      ← chunks → embeddings
│   ├── vector_store.py               ← embeddings → ChromaDB collections
│   ├── retriever.py                  ← query → top-K chunks
│   ├── generator.py                  ← chunks + history → streamed LLM answer
│   ├── rag_pipeline.py               ← unified pipeline interface
│   ├── ui_hel.py                     ← PDF viewer, status, navigation
│   └── utils.py                      ← chat history, citations, highlighting
├── uploads/                          ← uploaded PDFs (runtime)
├── vectordb/                         ← ChromaDB persistence (runtime)
├── pyproject.toml                    ← dependencies
└── README.md
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- `uv` package manager (or `pip`)
- Free [Groq API key](https://console.groq.com/)

### Step 1 —> Clone the repository
```bash
git clone https://github.com/Abhishek-Das1121/AMLGo_Labs_chatbot_assesment.git
cd AMLGo_Labs_chatbot_assesment
```

### Step 2 —> Add your API key
```bash
# Open .env and replace the placeholder:
GROQ_API_KEY=your_actual_key_here
```

### Step 3 —> Install dependencies
```bash
uv sync
# or
pip install -r requirements.txt
```

### Step 4 —> Launch the app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501** 🎉

### Step 5 —> Use it
1. Upload any PDF from the sidebar.
2. Wait for the status box to show **Ready**.
3. Ask a question in the chat box on the right.
4. Expand **Evidence & Sources** to see the highlighted passage, page number, and similarity score.
5. Click **View Source — Page N** to jump the PDF viewer to that exact page.

---

## Example Questions to Try

```
"What is the refund policy?"
"How many casual leaves are employees entitled to?"
"What are the limitations of liability in this agreement?"
"What happens if a party fails to meet the agreed standards?"
"Summarize the key obligations in this document."
"How many of those leaves are paid?"   ← follow-up, uses conversation memory
```

---

## Prompt Design

The system prompt is the key to preventing hallucination:

```
You are a precise document question-answering assistant.
Your ONLY job is to answer questions using the provided context
from the uploaded document.

Rules:
1. Answer ONLY using the provided context.
2. If the answer is not in the context, say:
   "I could not find this information in the provided document."
3. Be concise and direct.
4. Do not speculate or add outside knowledge.
5. Use recent conversation history only to resolve references
   (e.g. "those", "it") — never to answer from memory alone.
```

This ensures the LLM cannot invent information, if the answer isn't in the retrieved chunks, it says so.

---

## Hardware Compatibility

Designed and tested for:

| Spec | Value |
|---|---|
| RAM | 8GB |
| GPU | None (CPU only) |
| OS | Windows / Linux / macOS |
| Python | 3.11 |

---

## ✨ What's New in Version 2

Version 1 was a single-document chatbot built specifically for the eBay User Agreement. Version 2 — **DocuBuddy** — generalizes the same core RAG pipeline into a document-agnostic intelligence assistant.

| Area | Version 1 | Version 2 (DocuBuddy) |
|---|---|---|
| **Document support** | Fixed eBay PDF only | Any uploaded PDF |
| **UI layout** | Single-column chat | Two-column dashboard: PDF viewer + chat |
| **Citations** | Source chunk text only | Document name + **page number** + similarity score |
| **PDF access** | None | Built-in in-app PDF viewer with page jumping |
| **Chunk metadata** | Chunk ID only | Page number preserved through chunking → retrieval → citation |
| **Conversation** | Stateless, single-turn | Conversation memory for follow-up questions |
| **Re-processing** | Re-indexed on every run | Hash-based check — skips re-indexing unchanged PDFs |
| **Code structure** | 6 flat modules | Modular: `pdf_preprocess`, `vector_store`, `ui_hel`, and `utils` split out for clarity |
| **Response streaming** | ✅ Already present | ✅ Retained |
| **Hallucination guard** | ✅ Already present | ✅ Retained, extended to respect conversation context |

The underlying RAG fundamentals — sentence-aware chunking, BAAI/bge-small-en-v1.5 embeddings, ChromaDB retrieval, and Groq-powered streaming generation — are unchanged. V2 is an evolution of the same proven architecture, not a rewrite.

---

## ⚠ Current Limitations (V2)

Known issues currently being improved:

- Source page jump inside the PDF viewer is inconsistent.
- Evidence cards occasionally expand unexpectedly.
- Single-document support only.
- No hybrid search yet.
- No reranker.
- No evaluation metrics.
- No query decomposition.
- No multi-PDF retrieval.

These improvements are planned for future versions.

---


## Acknowledgements

- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) — embedding model
- [ChromaDB](https://www.trychroma.com/) — vector database
- [Groq](https://groq.com/) — LLM inference
- [LangChain](https://www.langchain.com/) — text splitting utilities
- [Streamlit](https://streamlit.io/) — UI framework

---

## Images

<img width="1920" height="1080" alt="Screenshot (540)" src="https://github.com/user-attachments/assets/96017c7e-95d1-4df6-99b8-bf21605e55ae" />
<img width="1920" height="1080" alt="Screenshot (541)" src="https://github.com/user-attachments/assets/7ba44e36-e700-4419-9f16-f71f66d05897" />
<img width="1920" height="1080" alt="Screenshot (542)" src="https://github.com/user-attachments/assets/56099a4f-2e4f-4f7f-95b5-dede6144f55b" />
<img width="1920" height="1080" alt="Screenshot (543)" src="https://github.com/user-attachments/assets/c2e9db03-ccec-4778-83c2-d6ae81a0bdb6" />
<img width="1920" height="1080" alt="Screenshot (544)" src="https://github.com/user-attachments/assets/2096004e-7c5d-498f-8079-89f9f36f9bdc" />
<img width="1920" height="1080" alt="Screenshot (545)" src="https://github.com/user-attachments/assets/916b0639-9284-4783-be38-e34adec3e493" />
<img width="1920" height="1080" alt="Screenshot (546)" src="https://github.com/user-attachments/assets/6662627f-ee13-4bb0-9bd9-690bdff06037" />
<img width="1920" height="1080" alt="Screenshot (547)" src="https://github.com/user-attachments/assets/887e8f85-1ed6-43db-8ceb-5948e43110ca" />


---

## Author

**Abhishek Das**

AI/ML Engineer | Generative AI | LLMs | RAG Systems | Python

🌐 LinkedIn : https://www.linkedin.com/in/abhishek-das1121/
🌐 Live Demo: https://docubuddy-ai.streamlit.app/
📂 Repository: https://github.com/Abhishek-Das1121/DocuBuddy

---

⭐ If you found this project useful, consider starring the repository.
