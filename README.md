---
title: DocuMind
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
---
# DocuMind 🧠

### Chat with your documents. Get cited answers — not hallucinations.

[![🚀 Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Hugging_Face-yellow?style=for-the-badge)](https://huggingface.co/spaces/Mohithsai10/documind)
![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-1.2-green?style=for-the-badge)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-412991?style=for-the-badge&logo=openai)

---

**Every professional has opened a 60-page report, ctrl+F'd a keyword, found nothing, and given up — knowing the answer was in there somewhere.**

DocuMind solves that. Upload any PDF, ask questions in plain English, and get precise answers with exact citations — page number and filename included — so you can verify every claim in seconds.

---

## Why DocuMind is Different

Most "chat with PDF" demos stop at making the LLM generate a plausible-sounding answer. DocuMind was built around the opposite instinct: **trust nothing the model says unless the source document says it first.**

### RAG over fine-tuning — and why that was the right call

Fine-tuning a model on your documents gives you a model that *sounds* knowledgeable about those documents. But it bakes the knowledge in permanently — you can't update it when the document changes, you can't audit which paragraph produced which claim, and you can't know when the model is confabulating versus recalling. Retrieval-Augmented Generation sidesteps all of this. The model stays stateless; the document stays authoritative. Every answer is grounded in chunks retrieved at query time, not weight-encoded approximations from training.

### Citations as a first-class engineering constraint

Most RAG systems bolt citations on at the end: retrieve chunks, answer, then list sources as a footnote. DocuMind inverts this. Each retrieved chunk is prefixed with its source label — `[Source: filename.pdf, Page N]` — **before** the context reaches the LLM. The model is instructed to copy that label verbatim as its inline citation. This means citations aren't generated — they're propagated. A hallucinated citation requires the model to hallucinate a label that was never in its input, which GPT-3.5-turbo essentially never does.

### The hallucination guard

The system prompt doesn't just say "be accurate." It gives the model a specific, unambiguous fallback: if the retrieved context doesn't contain the answer, respond with a fixed phrase. The LLM is never asked to *decide* how uncertain to sound — it either cites or it admits it can't. Confidence is computed separately from raw embedding similarity (L2 distance converted to a `[0, 1]` relevance score) and displayed as a traffic-light badge, giving users a signal that is independent of the model's own self-assessment.

---

## Features

- 📄 **Multi-PDF ingestion** — upload and index multiple PDFs in a single session
- 🔍 **Semantic search** — finds conceptually relevant chunks, not just keyword matches
- 🏷️ **Inline citations** — every fact is tagged with filename and page number, automatically
- 🟢🟡🔴 **Confidence scoring** — relevance badge derived from embedding similarity, not model confidence
- 🧹 **One-click database reset** — clear indexed documents and start fresh without restarting the server
- 📊 **Table-aware extraction** — PyMuPDF block-level parsing captures table cells that standard text extraction drops
- 🌑 **Dark-mode UI** — clean Gradio 6 interface, usable on day one without configuration

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INGESTION PIPELINE                          │
│                                                                     │
│   PDF Upload                                                        │
│       │                                                             │
│       ▼                                                             │
│   Text Extraction  ←── PyMuPDF get_text("dict"), block-level       │
│   (per page)            sorts by bbox to reconstruct table order   │
│       │                                                             │
│       ▼                                                             │
│   Chunking         ←── RecursiveCharacterTextSplitter               │
│   (300 chars,           chunk_size=300, overlap=50                  │
│    50 overlap)                                                      │
│       │                                                             │
│       ▼                                                             │
│   Embeddings       ←── OpenAI text-embedding-ada-002               │
│       │                                                             │
│       ▼                                                             │
│   ChromaDB         ←── persisted to .chroma/ on disk               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                           QUERY PIPELINE                            │
│                                                                     │
│   User Question                                                     │
│       │                                                             │
│       ▼                                                             │
│   Semantic Search  ←── similarity_search_with_score, k=8           │
│       │                  L2 distance → relevance score             │
│       │                                                             │
│       ▼                                                             │
│   Context Assembly ←── prepend [Source: file.pdf, Page N]          │
│                          to each retrieved chunk                    │
│       │                                                             │
│       ▼                                                             │
│   LLM              ←── GPT-3.5-turbo, temperature=0                │
│   (LCEL chain)          system prompt enforces citation             │
│       │                                                             │
│       ▼                                                             │
│   Cited Answer + Confidence Badge                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| UI | Gradio | 6.14 |
| LLM | OpenAI GPT-3.5-turbo | API v2 |
| Embeddings | OpenAI text-embedding-ada-002 | API v2 |
| Orchestration | LangChain + LCEL | 1.2.18 |
| Vector Store | ChromaDB | 1.5.9 |
| PDF Parsing | PyMuPDF (fitz) | 1.27.2 |
| Language | Python | 3.11+ |

---

## Run Locally

**1. Clone and install**
```bash
git clone https://github.com/Mohithsai10/documind.git
cd documind
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Add your OpenAI key**
```bash
cp .env.example .env
# open .env and set OPENAI_API_KEY=sk-...
```

**3. Launch**
```bash
python app.py
# → open http://localhost:7860
```

**4. Use it**
1. Upload one or more PDFs using the left panel
2. Click **Index Documents** — wait for the chunk count
3. Ask questions in the chat box
4. Click **Clear Database** before switching to a new document set

---

## What I Learned

**Retrieval quality is an extraction problem before it is a search problem.**
I spent hours tuning embedding thresholds and chunk sizes before realising the retrieval was failing because PyMuPDF's default `get_text()` was silently dropping table cells — not because semantic search was misconfigured. The fix was switching to `get_text("dict")`, iterating blocks individually, and sorting by bounding-box coordinates to reconstruct reading order. The lesson: validate your extracted text *before* you index it. A vector store built on bad text will rank irrelevant chunks confidently.

**LLM output is only as trustworthy as the input you give it.**
The citation problem turned out to be an input design problem. When I passed raw text chunks to the LLM and told it to "cite sources," it had no source information to cite — so it invented placeholders like `{filename}`. The fix was to prepend `[Source: filename.pdf, Page N]` to each chunk *before* it entered the prompt. Citations became propagation, not generation. That distinction — the model copying a label versus inventing one — is the difference between a reliable system and a convincing-sounding one.

**Version mismatches in AI stacks fail silently and catastrophically.**
LangChain 1.x removed `langchain.chains`, `langchain.schema`, and `langchain.text_splitter` from the top-level namespace — and `RetrievalQA` was deprecated in favour of LCEL — none of which was obvious from the error messages. Gradio 6 moved `theme`, `css`, and `js` from `gr.Blocks()` to `demo.launch()`, and changed the chatbot history format from tuples to message dicts. The app appeared to import cleanly but failed at runtime. The lesson: pin your versions, read changelogs before upgrading, and write smoke tests that exercise actual runtime behaviour — not just imports.

---

## Project Structure

```
documind/
├── app.py          # Gradio UI — layout, event handlers, theming
├── ingest.py       # PDF extraction, chunking, embedding, ChromaDB write
├── retriever.py    # ChromaDB load, similarity search, metadata retrieval
├── chain.py        # LCEL RAG chain, confidence scoring, answer assembly
├── prompts.py      # System prompt and RAG prompt templates
├── utils.py        # Source formatting, PDF validation, DB management
├── data/           # Drop PDFs here for batch ingestion (optional)
├── .chroma/        # Auto-created vector store (gitignored)
└── requirements.txt
```

---

## License

MIT

---

<p align="center">
  Built by <strong>Mohith Sai Varma Kirthipati</strong> — ECE → AI Engineer<br/>
  <a href="https://www.linkedin.com/in/mohithsai">LinkedIn</a> · <a href="https://github.com/Mohithsai10">GitHub</a>
</p>
