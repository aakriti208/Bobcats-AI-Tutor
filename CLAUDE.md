# CLAUDE.md — SimpleRAG

SimpleRAG is a Retrieval-Augmented Generation system that ingests Canvas LMS course content and provides a web-based AI tutor with side-by-side RAG vs. non-RAG answer comparison. Built for Texas State University.

---

## Build & Run Commands

### Setup
```bash
python3.11 -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.template .env            # then fill in Canvas credentials
```

### Start Ollama (required before running the app)
```bash
ollama serve                     # in one terminal
ollama pull gemma:2b             # in another; or whichever model is in .env
```

### Ingest Canvas Content
```bash
python scripts/ingest_data.py --course <COURSE_ID>   # single course
python scripts/ingest_data.py --full                  # all configured courses
python scripts/ingest_data.py --incremental           # only new/changed content
python scripts/ingest_data.py --reset --full          # wipe ChromaDB and re-ingest
python scripts/ingest_data.py --course <ID> --content-type page  # specific type
```

### Run the App
```bash
python app.py           # web server → http://localhost:8000
python rag_demo.py      # interactive CLI demo
```

### Utilities
```bash
python scripts/list_courses.py          # list available Canvas courses
python scripts/verify_canvas_data.py   # verify ingested data integrity
```

---

## Architecture Notes

### Pipeline Overview

**Ingestion:**
```
Canvas API → CanvasClient → ContentHandlers → DocumentProcessor → Embedder → ChromaDB
```

**Query:**
```
User Question → Embedder → ChromaDB → Retriever → Generator (Ollama) → Answer
```

### Key Components

| File | Role |
|------|------|
| `src/config.py` | Single source of truth for all env-driven config |
| `src/ingestion/canvas_client.py` | Canvas REST client (retry, pagination, rate-limit handling) |
| `src/ingestion/document_processor.py` | HTML/PDF/PPTX extraction + word-based chunking |
| `src/ingestion/metadata_tracker.py` | JSON state file for incremental sync tracking |
| `src/ingestion/content_handlers/` | One handler class per Canvas content type |
| `src/embedding/embedder.py` | Sentence-Transformers wrapper (`all-MiniLM-L6-v2`) |
| `src/vectorstore/chroma_manager.py` | ChromaDB CRUD + query |
| `src/retrieval/retriever.py` | Semantic search with date-keyword boosting |
| `src/generation/generator.py` | Ollama prompting — dual answers (RAG + non-RAG) |
| `app.py` | FastAPI server — endpoints: `/`, `/health`, `/query` |
| `templates/index.html` | Web UI (served by FastAPI) |
| `static/app.js` | Form handling + dark mode (persisted in localStorage) |

### Content Handlers Pattern
Each Canvas content type (`page`, `assignment`, `announcement`, `discussion`, `file`, `syllabus`) has a handler class inheriting from `BaseContentHandler(ABC)`. The module handler exists but is **disabled** — modules are navigation containers, not content.

### Chunking
- Word-based sliding window: 500 words per chunk, 50-word overlap
- HTML tables are preserved as pipe-delimited rows for schedule parsing
- Each chunk carries full source metadata

### Retrieval Enhancements
- Date keyword detection (regex for "March 26" style) boosts similarity score by 0.3
- Lowers similarity threshold for date queries
- Queries 3× `top_k` when keywords detected, then re-ranks

### Dual-Answer Design
The generator intentionally produces two answers per query:
1. **Bobcat AI Tutor** — grounded in Canvas materials + LLM knowledge
2. **Local LLM** — pure LLM knowledge (no context)

This comparison is the core demo value of the project.

---

## Debugging Insights

- **ChromaDB issues**: Run `--reset --full` to wipe and re-ingest. DB lives at `data/chroma_db/`.
- **Ollama not responding**: Ensure `ollama serve` is running before starting `app.py`. Check `OLLAMA_BASE_URL` in `.env`.
- **No results returned**: Lower `SIMILARITY_THRESHOLD` in `.env` (default 0.5). Check that content was actually ingested.
- **Canvas API auth failures**: Token must have student/TA-level read access. Check `CANVAS_API_TOKEN` and `CANVAS_BASE_URL` (must end with `/`).
- **Slow ingestion**: Increase `INGEST_MAX_WORKERS` (default 4). First run downloads/embeds everything; incremental runs are much faster.
- **Logs**: Each ingestion run writes a timestamped log to `data/logs/`.
- **Incremental sync state**: Stored in `data/metadata/ingestion_state.json`. Delete it to force full re-ingest without `--reset`.
- **Form submission bugs**: The JS in `static/app.js` uses `fetch` POST to avoid URL parameter leakage — don't revert to GET form submission.
- **PDF/PPTX not parsing**: Check `MAX_FILE_SIZE_MB` (default 50) and `PDF_EXTRACTION_TIMEOUT` (default 30s) in `.env`.

---

## Code Style Preferences

- **Classes**: `PascalCase`
- **Functions/methods**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE` (defined in `src/config.py`)
- **Private methods**: leading underscore (e.g., `_make_request()`)
- **Type hints**: used throughout (`List[Dict]`, `Optional[str]`, etc.)
- **Docstrings**: Google-style with `Args:`, `Returns:`, `Raises:`
- **Logging**: every module does `logger = logging.getLogger(__name__)` — no `print()` in src/
- **Error handling**: custom exceptions in `canvas_client.py` (`CanvasAPIError`, `RateLimitError`, `AuthenticationError`, `NotFoundError`); try/except with descriptive log messages elsewhere
- **Indentation**: 4 spaces
- **Config access**: always import from `src/config.py`, never `os.environ` directly elsewhere

---

## Workflow Habits

- Install scripts are in `installation/` — prefer those for new environment setup rather than manual steps.
- `.env` is gitignored; use `.env.template` as the canonical reference.
- Runtime artifacts (`data/chroma_db/`, `data/logs/`, `data/metadata/`, `data/raw/`, `data/processed/`) are gitignored — the `data/` directory is created at runtime.
- Adding a new Canvas content type = create a new handler in `src/ingestion/content_handlers/`, inherit `BaseContentHandler`, export from `__init__.py`.
- The embedding model (`all-MiniLM-L6-v2`) is configurable via `EMBEDDING_MODEL` in `.env` but changing it requires a full re-ingest since embeddings are model-specific.
- UI is Texas State branded (maroon `#501214`, gold `#FFC72C`) — keep consistent.
- Dark mode state persists via `localStorage` in the browser.
