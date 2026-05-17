# RAG-Project — Local PDF Question-Answering with LangChain, ChromaDB & Ollama

> Built by [kranthi-prog](https://github.com/kranthi-prog) · May 2026

## Overview

RAG-Project is a fully local Retrieval-Augmented Generation (RAG) system that lets you ask natural-language questions about your own PDF documents and get accurate, grounded answers — without sending any data to the cloud. It ships with both a command-line interface (`query.py`) and an interactive Streamlit web UI (`app.py`), making it easy to demo and extend.

Built hands-on using Claude Code, this project demonstrates core AI engineering skills: document ingestion, semantic chunking, vector search, prompt engineering, and local LLM integration.

---

## Architecture

```
                    INDEXING  (ingest.py — run once)
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  documents/         Chunk            Embed                   │
│  (.pdf/.txt/.docx) ──────► 500-char ──────► all-MiniLM-L6  │
│                             chunks           (sentence-       │
│                               │              transformers)    │
│                               └──────────────► ChromaDB      │
│                                               (persisted to   │
│                                                chroma_db/)    │
└──────────────────────────────────────────────────────────────┘

                    QUERYING  (query.py or app.py — every question)
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  User Question                                               │
│       │                                                      │
│       ▼                                                      │
│  all-MiniLM-L6-v2  ──► Embed question into vector           │
│       │                                                      │
│       ▼                                                      │
│  ChromaDB          ──► Cosine similarity search             │
│                    ──► Top-4 most relevant chunks            │
│       │                                                      │
│       ▼                                                      │
│  Prompt Template   ──► "Answer using ONLY this context:      │
│                         {chunks} \n Question: {question}"    │
│       │                                                      │
│       ▼                                                      │
│  llama3.2 (Ollama) ──► Generate answer                      │
│       │                                                      │
│       ▼                                                      │
│  CLI: Answer + source chunks printed to terminal             │
│  UI:  Answer + sources shown in Streamlit chat interface     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Tool | Why Chosen |
|---|---|---|
| LLM | llama3.2 3B via Ollama | Runs fully locally, no API key, fast enough for Q&A |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) | Lightweight, high-quality semantic vectors, runs on CPU |
| Vector Store | ChromaDB (persistent local) | Zero-config, file-based, no server or Docker needed |
| RAG Framework | LangChain 1.x (LCEL) | Industry standard; composable pipe syntax is readable and extensible |
| UI | Streamlit | Rapid prototyping of interactive data apps in pure Python |
| PDF Loader | pypdf | Lightweight, pure-Python, handles text-layer PDFs without dependencies |
| DOCX Loader | docx2txt | Native `.docx` support without LibreOffice or conversion tools |
| Language | Python 3.11+ | Matches LangChain and ChromaDB minimum requirements |

---

## Key Design Decisions

### Chunk size 500 / overlap 75
Five hundred characters is large enough to hold one complete idea while being small enough for `all-MiniLM-L6-v2` to encode precisely. Shorter chunks lose context; longer chunks blur multiple topics into one vector, weakening retrieval precision. The 75-character overlap ensures sentences at chunk boundaries appear in both adjacent chunks — no context is lost at the seams.

### sentence-transformers over Ollama embeddings
`all-MiniLM-L6-v2` is a bi-encoder fine-tuned specifically for semantic similarity tasks. It runs entirely on CPU, requires no additional Ollama model pull, and produces high-quality 384-dimensional vectors. Using the LLM to produce embeddings would degrade retrieval quality because general-purpose LLMs are not optimised to produce vectors where "close in space = similar in meaning."

### ChromaDB over Pinecone or Weaviate
A local prototype should have zero infrastructure dependencies. ChromaDB writes directly to a local folder (`chroma_db/`) — no account, no server, no Docker required. Pinecone and Weaviate are excellent production choices, but they require API keys and internet access, which conflicts with the offline-first goal of this project.

### llama3.2 (3B) for the LLM
A 3B parameter model fits comfortably in RAM on a standard developer laptop and responds in seconds via Ollama. The RAG pattern compensates for the model's smaller knowledge base by supplying the relevant facts as context — the LLM's job is reading comprehension and formatting, not recall.

### Two interfaces: CLI + Streamlit
`query.py` provides a fast, scriptable CLI for power users and automation. `app.py` provides a shareable, browser-based chat UI for demos and non-technical stakeholders. Both share the same underlying retrieval and generation pipeline.

### Hallucination prevention by prompt design
The prompt template explicitly constrains the LLM: *"Use only the context below to answer the question. If the answer is not in the context, say 'I don't know'."* This grounding is what prevents hallucination — when retrieved chunks contain the answer, output is accurate and cited; when they don't, the system says so rather than making something up.

---

## Project Structure

```
my-rag-project/
│
├── documents/          ← Drop your PDFs, TXTs, or DOCX files here
│
├── chroma_db/          ← NOT in git (auto-created by ingest.py)
├── venv/               ← NOT in git (create locally)
│
├── ingest.py           Indexing pipeline: load → chunk → embed → store
├── query.py            CLI interface: question → retrieve → generate → answer
├── app.py              Streamlit web UI with upload, indexing, and chat
├── requirements.txt    Python dependencies
├── .gitignore
└── README.md
```

---

## Setup Instructions

### Prerequisites

- **Python 3.11+** — [python.org/downloads](https://www.python.org/downloads/)
- **Ollama** installed and running — [ollama.com](https://ollama.com)
- **Git** — [git-scm.com](https://git-scm.com)

### 1 — Pull the Ollama model

```bash
ollama pull llama3.2
```

Verify it is available:

```bash
ollama list
```

### 2 — Clone the repository

```bash
git clone https://github.com/kranthi-prog/RAG-project.git
cd RAG-project/my-rag-project
```

### 3 — Create and activate a virtual environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

Your terminal prompt will show `(venv)` when active.

### 4 — Install dependencies

```bash
pip install -r requirements.txt
```

### 5 — Add your documents

Drop any `.pdf`, `.txt`, or `.docx` files into the `documents/` folder.

### 6 — Run the ingest pipeline

```bash
python ingest.py
```

This loads, chunks, embeds, and stores all documents in `chroma_db/`. Run this once, and again whenever you add or change documents.

---

## Usage

### Option A — Streamlit Web UI

```bash
streamlit run app.py
```

Opens in your browser at `http://localhost:8501`. Upload a PDF, click **Index PDF**, then ask questions in the chat.

### Option B — Command-Line Interface

```bash
# Ask a question directly
python query.py "What is this document about?"

# Or use interactive mode
python query.py
```

---

## What I Learned

**The relationship between embedding models and LLMs**
I assumed at the start that the LLM could just "read" the documents. Building this taught me there are two distinct models doing two different jobs: `all-MiniLM-L6-v2` converts text into vectors (numbers that capture meaning) so similarity search is possible, and `llama3.2` reads those retrieved chunks and writes a coherent answer. Using the LLM to produce embeddings would give poor retrieval results because it is not trained for that specific task.

**Why chunking strategy matters**
The chunk size and overlap directly affect retrieval quality. Chunks that are too small lose context; chunks that are too large blend multiple topics into one vector, making it harder to retrieve precisely relevant content. The 75-character overlap ensures no idea is cut off at a boundary.

**Grounding beats model size**
The most important architectural insight: RAG is not about making the LLM smarter — it is about giving it the right information at the right time. The prompt constraint *"answer using ONLY this context"* is what prevents hallucination. When retrieved chunks contain the answer, output is accurate. When they don't, the system correctly says it doesn't know rather than inventing an answer.

**How vector similarity enables fast retrieval**
ChromaDB uses HNSW (Hierarchical Navigable Small World graphs) for approximate nearest-neighbour search. Instead of comparing the question vector against every stored vector one by one, HNSW builds a graph structure at ingest time that allows retrieval in O(log n). This is why queries feel instant even with thousands of chunks.

---

## Limitations & Future Improvements

| Limitation | Impact | Planned Fix |
|---|---|---|
| No re-ranker | Retrieved chunks ranked by vector similarity only — a closer vector is not always the best answer | Add a cross-encoder re-ranker (e.g. `ms-marco-MiniLM`) as a second-pass filter |
| No hybrid search | Pure vector search can miss content when vocabulary differs from the question | Combine ChromaDB (dense) with BM25 (sparse keyword) using `EnsembleRetriever` |
| PDF figures not extracted | Content inside diagrams and charts is invisible to the pipeline | Integrate `unstructured` with OCR or a vision-capable model |
| No conversation memory | Each query is stateless — follow-up questions lose prior context | Add `ConversationBufferWindowMemory` for multi-turn Q&A |
| No evaluation framework | No systematic way to measure retrieval precision or answer accuracy | Implement RAGAs or a custom eval suite with labelled Q&A pairs |
| No incremental ingest | Re-running `ingest.py` re-embeds all documents, not just new ones | Add content-hash deduplication to skip unchanged files |

---

## License

MIT
