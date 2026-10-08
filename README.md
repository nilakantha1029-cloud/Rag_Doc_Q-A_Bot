# 🔦 Paperlight — RAG Document Q&A Bot

Paperlight is a Retrieval-Augmented Generation (RAG) chatbot that answers questions about a set of PDF documents (four reports on AI, jobs and education). It retrieves the most relevant passages from a local vector database and asks an LLM (via the Groq API) to answer **only from those passages**, with numbered citations like `[1]` that link back to the exact file and page. If the answer is not in the documents, it says so instead of guessing.

It ships with two interfaces: a **Streamlit web app** (`app.py`) and a **terminal chat** (`src/chat.py`).

---

## Table of Contents
1. [Tech Stack](#1-tech-stack)
2. [Architecture Overview](#2-architecture-overview)
3. [Chunking Strategy](#3-chunking-strategy)
4. [Embedding Model and Vector Database](#4-embedding-model-and-vector-database)
5. [Project Structure](#5-project-structure)
6. [Setup Instructions](#6-setup-instructions)
7. [Environment Variables](#7-environment-variables)
8. [Example Queries](#8-example-queries)
9. [Testing](#9-testing)
10. [Using Your Own PDFs](#10-using-your-own-pdfs)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Tech Stack

| Component | Library / Tool | Version | Purpose |
|---|---|---|---|
| Language | Python | 3.12 (3.10+ should work) | Runtime |
| PDF parsing | `pypdf` | 6.19.0 | Extract text page by page |
| Chunking | `langchain-text-splitters` | 1.1.3 | `RecursiveCharacterTextSplitter` |
| Embeddings | Chroma default: `all-MiniLM-L6-v2` (ONNX, via `onnxruntime`) | bundled with `chromadb` | Text → 384-dim vectors |
| Vector DB | `chromadb` | 1.5.9 | Persistent local vector store (cosine / HNSW) |
| LLM API client | `groq` | 1.7.0 | Calls the Groq chat completions API |
| LLM | `openai/gpt-oss-120b` hosted on Groq | — | Answer generation (temperature 0) |
| Env loading | `python-dotenv` | 1.2.4 | Reads `GROQ_API_KEY` from `.env` |
| Web UI | `streamlit` | 1.65.0 | Chat interface with citations and source cards |

All Python packages are pinned in `requirements.txt`.

---

## 2. Architecture Overview

The pipeline has two phases: an **offline ingestion phase** (run once, or whenever PDFs change) and an **online query phase** (run for every question).

```
┌──────────────────────── INGESTION  (python src/ingest.py) ────────────────────────┐
│                                                                                   │
│  data/*.pdf ──► loader.py ──► chunker.py ──► vectorstore.py ──► chroma_db/        │
│                 (pypdf,        (700 chars,    (Chroma embeds     (persistent      │
│                 strip headers, 150 overlap,   with MiniLM-L6-v2, HNSW index,      │
│                 footers, page  drop reference  batches of 100)    cosine)         │
│                 numbers)       chunks)                                            │
└───────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────── QUERY  (app.py or src/chat.py) ───────────────────────────┐
│                                                                                   │
│  User question                                                                    │
│       │                                                                           │
│       ▼                                                                           │
│  search_diverse()  ── embed question, fetch top 30 chunks ──► filter by distance  │
│  (vectorstore.py)     ► best chunk from EVERY relevant PDF first                  │
│                       ► fill remaining slots by rank (max 4 per PDF, total 8)     │
│                       ► merge chunks from the same page                           │
│       │                                                                           │
│       ▼                                                                           │
│  generate_answer()  ── numbered context [1]..[n] + strict system prompt ──►       │
│  (generator.py)        Groq LLM (openai/gpt-oss-120b, temperature 0)              │
│       │                                                                           │
│       ▼                                                                           │
│  Answer with [n] citations ──► UI shows answer + expandable source cards          │
│                                (file name, page number, passage text)             │
└───────────────────────────────────────────────────────────────────────────────────┘
```

### Stage by stage

| Stage | File | What happens |
|---|---|---|
| **Ingestion** | `src/loader.py` | Reads every `*.pdf` in `data/` with `pypdf`, page by page. Lines that repeat on ≥40% of pages (min. 3) are treated as headers/footers and removed, as are bare page numbers. Output: one record per page `{text, source, page}`. |
| **Chunking** | `src/chunker.py` | Splits each page with `RecursiveCharacterTextSplitter` (700 chars, 150 overlap). Chunks that look like bibliography/endnotes (≥2 URLs, or ≥3 `[n]` markers with ≥2 years) are discarded. Each chunk gets an ID like `file.pdf-p12-c3`. |
| **Embedding + storage** | `src/vectorstore.py` | `chromadb.PersistentClient(path="chroma_db")` with collection `docs` (cosine distance). Chroma embeds text automatically with its default `all-MiniLM-L6-v2` model. Chunks are inserted in batches of 100. |
| **Retrieval** | `src/vectorstore.py` → `search_diverse()` | Retrieves top 30 candidates, drops those more than 0.45 cosine distance worse than the best hit, guarantees the best chunk from each relevant PDF (so every document gets a voice in cross-document questions), fills up to 8 chunks (max 4 per PDF), then merges chunks from the same page. |
| **Generation** | `src/generator.py` | Builds a context block `[1] (file, page N) text…`, sends it with a strict system prompt to Groq. The prompt forces context-only answers, `[n]` citations on every claim, a fixed 3-part format for compare/agree questions, and the exact refusal sentence *"I couldn't find that in the documents."* when nothing relevant exists. |
| **Presentation** | `app.py` / `src/chat.py` | Converts `[n]` into badges, shows the cited source cards (file, page, snippet), and highlights the used documents in the sidebar. The terminal version prints a `Sources:` list instead. |

### Current knowledge base (as shipped)

| PDF | Pages | 
|---|---|
| `2024-wttc-introduction-to-ai.pdf` | 44 |
| `future_of_learning.pdf` | 26 |
| `Ai_tech_pros_and_cones.pdf` | 12 |
| `Ai Replace human jobs.pdf` | 8 |
| **Total** | **90 pages → 366 chunks** |

---

## 3. Chunking Strategy

**Strategy: recursive character splitting — 700 characters per chunk, 150 characters of overlap, applied per page.**

Why this was chosen:

- **Recursive splitting respects structure.** `RecursiveCharacterTextSplitter` tries to break on paragraph boundaries first, then lines, then sentences/words, so chunks rarely cut a sentence in half.
- **700 characters (~120 words) is precise.** Small chunks make the embedding focus on one idea, which improves retrieval for fact questions (e.g. "What market share does TSMC have?"). Larger chunks dilute the vector with unrelated text.
- **150-character overlap (~20%)** prevents a fact that straddles a chunk boundary from being lost.
- **Splitting per page** keeps an exact `page` number in metadata, which makes accurate citations possible.
- **Cleaning before splitting** (header/footer/page-number removal in the loader, bibliography filtering in the chunker) stops noise like reference lists and URLs from polluting search results.
- **Merging same-page hits at retrieval time** restores context that small chunks would otherwise lose, so the LLM sees fuller passages.

---

## 4. Embedding Model and Vector Database

**Embedding model: `all-MiniLM-L6-v2` (Chroma's built-in default, run through ONNX Runtime).**

- Free, runs **locally on CPU** — no embedding API key, no per-request cost, no data leaves your machine.
- Small (~90 MB, 384 dimensions) and fast, yet strong for English semantic search.
- No code needed: Chroma embeds both the stored chunks and the incoming question automatically, guaranteeing both use the same model.
- The model is downloaded automatically on first run and cached (see [Troubleshooting](#11-troubleshooting)).

**Vector database: ChromaDB (persistent, local).**

- Zero infrastructure — an embedded database stored in the `chroma_db/` folder; no server or Docker needed.
- Persists to disk, so you ingest once and query many times.
- Stores text, vectors and metadata (`source`, `page`) together, which is exactly what citations need.
- Uses an HNSW index with **cosine distance** (`hnsw:space = cosine`), the standard metric for text embeddings.
- Perfect for a project of this size (hundreds to tens of thousands of chunks). For a large multi-user deployment you would move to a managed store (Pinecone, Qdrant, pgvector).

**LLM: `openai/gpt-oss-120b` on Groq** — chosen for fast inference, a generous free tier, and strong instruction-following (needed to keep to context-only answers and the citation format). `temperature=0` keeps answers factual and repeatable. You can change the model by editing `MODEL` in `src/generator.py`.

---

## 5. Project Structure

```
Rag_Doc_Q&A_Bot/
├── app.py               # Streamlit web UI ("Paperlight")
├── requirements.txt     # Pinned Python dependencies
├── .env.example         # Template for your API key (safe to commit)
├── .env                 # YOUR real key (you create this; git-ignored)
├── .gitignore
├── test_rag.py          # 30-question automated evaluation
├── data/                # Source PDFs (input to ingestion)
│   ├── 2024-wttc-introduction-to-ai.pdf
│   ├── Ai Replace human jobs.pdf
│   ├── Ai_tech_pros_and_cones.pdf
│   └── future_of_learning.pdf
├── src/
│   ├── loader.py        # PDF → cleaned pages
│   ├── chunker.py       # pages → chunks
│   ├── vectorstore.py   # Chroma storage + diverse retrieval
│   ├── generator.py     # Groq LLM call + system prompt
│   ├── ingest.py        # Runs loader → chunker → vector store
│   └── chat.py          # Terminal chat interface
└── chroma_db/           # Generated by ingest.py (git-ignored)
```

---

## 6. Setup Instructions

### Prerequisites
- **Python 3.10 or newer** (developed on 3.12) — check with `python --version`
- **Git**
- A **free Groq API key** (see [section 7](#7-environment-variables))
- Internet access on first run (to install packages and download the embedding model, ~90 MB)

### Step 1 — Clone the repository
```bash
git clone <YOUR_REPO_URL>
cd "Rag_Doc_Q&A_Bot"
```
> Replace `<YOUR_REPO_URL>` with your GitHub URL. Use the actual folder name of your repo after cloning.

### Step 2 — Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell blocks the script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first. In Command Prompt use `.venv\Scripts\activate.bat`.

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```
You should now see `(.venv)` at the start of your terminal prompt.

### Step 3 — Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Add your API key
Copy the template and edit it:

```bash
# macOS / Linux
cp .env.example .env

# Windows (PowerShell)
Copy-Item .env.example .env
```
Open `.env` and replace the placeholder:
```
GROQ_API_KEY=gsk_your_real_key_here
```
Details in [section 7](#7-environment-variables).

### Step 5 — Build the vector database (ingestion)
Place your PDFs in `data/` (the four sample PDFs are already there), then run **from the project root**:
```bash
python src/ingest.py
```
Expected output (first run also downloads the embedding model):
```
90 pages → 366 chunks
Done.
```
This creates the `chroma_db/` folder.

> ⚠️ Run `ingest.py` **once**. Running it again on an existing `chroma_db/` will fail with a duplicate-ID error. To re-ingest, delete the folder first:
> `rm -rf chroma_db` (macOS/Linux) or `Remove-Item -Recurse -Force chroma_db` (PowerShell).

### Step 6 — Run the bot

**Option A — Web app (recommended):**
```bash
python -m streamlit run app.py
```
Streamlit opens `http://localhost:8501` in your browser. Click a suggested question or type your own.

**Option B — Terminal chat:**
```bash
python src/chat.py
```
Type a question and press Enter; type `q` to quit. The answer is followed by a `Sources:` list with file names and page numbers.

### Step 7 (optional) — Run the evaluation suite
```bash
python test_rag.py
```
See [section 9](#9-testing).

---

## 7. Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | API key used by `src/generator.py` to call the Groq LLM. |

That is the **only** key needed. Embeddings run locally and need no key.

### How to get and set the key
1. Go to **https://console.groq.com/keys** and sign up / log in (free tier available).
2. Click **Create API Key**, give it a name, and copy it (starts with `gsk_`). It is shown only once.
3. In the project root, create a file named exactly `.env` (copy `.env.example`).
4. Add this line, with no quotes and no spaces around `=`:
   ```
   GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
   ```
5. Save the file. The app loads it automatically through `python-dotenv`.

### Security rules
- `.env` is listed in `.gitignore` — **never commit it** or paste the key into code, issues or screenshots.
- Only `.env.example` (placeholder value) is committed.
- If a key is ever exposed, revoke it in the Groq console and create a new one.
- Alternative without a file: set it in your shell — `export GROQ_API_KEY=...` (macOS/Linux) or `$env:GROQ_API_KEY="..."` (PowerShell).

---

## 8. Example Queries

These questions are answerable from the bundled PDFs (they come from the project's own test suite):

| # | Question | Expected answer theme |
|---|---|---|
| 1 | *Which AI systems beat a chess grandmaster in 1997 and won Jeopardy in 2011?* | IBM Deep Blue (beat Kasparov, 1997) and IBM Watson (Jeopardy, 2011) — WTTC report |
| 2 | *What did the Swedish study find about AI breast cancer screening?* | AI-supported screening matched two radiologists, cut workload by ~50%, detected ~20% more cancers early |
| 3 | *What is the 5% problem in AI and education?* | Only a small, highly motivated subset of students engages productively with AI tools, which can bias study results — Future of Learning |
| 4 | *How many children are out of school globally and how many more teachers are needed by 2030?* | ~272 million children out of school; ~44 million additional teachers needed |
| 5 | *What is the difference between a data warehouse and a data lake?* | Warehouse = structured, organised data; lake = unstructured/varied raw data |
| 6 | *Do the documents agree on whether AI will replace jobs?* | Cross-document answer: all expect major change; they differ in emphasis between augmentation/transformation and job displacement risk |
| 7 | *What do the documents say about privacy and data security?* | Large data needs create privacy risks, breaches and leaks; safeguards such as guardrails and omitting personal data from prompts |
| 8 | *Will AI replace teachers according to the documents?* | No — the documents frame AI as supporting teachers and learning, not replacing instruction |
| 9 | *What do the documents say about AI training and reskilling?* | Few workers trained (e.g. 13% in a Randstad survey); reskilling/upskilling and lifelong learning are essential |
| 10 | *What is explainable AI and why does it matter?* | Making AI decisions transparent ("black box" → "glass box") to build trust and let users verify results |

**Out-of-scope questions** (the bot should refuse with *"I couldn't find that in the documents."*):
- *Who is the CEO of Anthropic?*
- *What is the weather in Hyderabad today?*
- *What is the price of a GPT-5 subscription?*

---

## 9. Testing

`test_rag.py` runs **30 automatic test cases** against the live pipeline and checks each answer for required facts, required source files, minimum number of cited PDFs, forbidden words, and correct refusals.

| Category | Cases | What it checks |
|---|---|---|
| `fact` | 9 | Exact numbers/names from a single document |
| `concept` | 4 | Explanations of concepts |
| `cross` | 8 | Answers that combine multiple documents |
| `tricky` | 4 | False premises, nuanced claims |
| `refuse` | 5 | Questions not in the PDFs must be refused |

```bash
python test_rag.py
```
Output is printed as `PASS`/`FAIL` per question with a per-category summary, and full answers are written to `test_results.txt` (git-ignored). The script pauses 3 seconds between questions to respect Groq rate limits. The last recorded run passed **28 of 30**; the two misses were wording-sensitive checks (an LLM phrasing difference), not retrieval failures.

---

## 10. Using Your Own PDFs

1. Delete the old index: remove the `chroma_db/` folder.
2. Replace or add PDFs in `data/` (text-based PDFs only; scanned image PDFs need OCR first).
3. Run `python src/ingest.py`.
4. Start the app: `streamlit run app.py`.

Note: the UI text in `app.py` (title, "four AI papers", suggested questions) and the test cases in `test_rag.py` are written for the sample documents — edit `SUGGESTED` and the hero text for your own content.

---

## 11. Troubleshooting

| Problem | Cause / Fix |
|---|---|
| `groq.AuthenticationError` / `Invalid API Key` | `.env` is missing, misnamed (must be exactly `.env`, not `.env.txt`), or the key is wrong. Check `GROQ_API_KEY=` has no quotes/spaces. |
| `Could not get an answer: ...` in the UI | Usually an API error: bad key, rate limit, or no internet. The real message is shown after the colon. |
| `chromadb.errors.DuplicateIDError` / duplicate IDs | `ingest.py` was run twice. Delete `chroma_db/` and re-run it. |
| Empty answers / `IndexError` / collection empty | Ingestion hasn't been run. Run `python src/ingest.py` first. |
| `ModuleNotFoundError` | Virtual environment not activated, or `pip install -r requirements.txt` not run. |
| Embedding model download fails | The first ingest/query downloads `all-MiniLM-L6-v2` (~90 MB) to `~/.cache/chroma/`. Check your internet/proxy and retry. |
| `rate_limit_exceeded` from Groq | Wait a minute or switch `MODEL` in `src/generator.py` to a smaller/faster model listed in the Groq console. |
| Model not found | Groq occasionally retires models; pick a current one from https://console.groq.com/docs/models and update `MODEL`. |
| Streamlit port busy | `streamlit run app.py --server.port 8502` |

---

