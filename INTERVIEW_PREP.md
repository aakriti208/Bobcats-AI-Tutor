# Canvas RAG System — Interview Prep Guide

## 1. What is this project?

A **Retrieval-Augmented Generation (RAG) system** built on top of the Canvas Learning Management System (LMS). It lets students ask natural language questions and get answers grounded in their actual course materials — syllabus, assignments, lecture slides, discussions, etc.

**One-liner:** "I built a RAG pipeline that ingests a Canvas course's content into a vector database and answers student questions by combining semantic search with an LLM."

---

## 2. High-Level Architecture

```
Canvas LMS API
      |
      v
[Ingestion Layer]          <- Fetches & processes content
      |
      v
[Document Processor]       <- Extracts text, chunks it
      |
      v
[Embedder]                 <- Converts text to vectors
      |
      v
[ChromaDB (Vector Store)]  <- Stores embeddings persistently
      |
      v
[Retriever]                <- Semantic search on query
      |
      v
[Generator (LLM)]          <- Generates final answer
      |
      v
Answer to Student
```

---

## 3. Component Breakdown

### `src/ingestion/canvas_client.py` — Canvas API Client
- Uses `requests.Session` with a Bearer token for authentication
- Handles **pagination** automatically via Canvas's `Link` header (`rel="next"`)
- Fetches: pages, modules, assignments, announcements, discussions, files (PDFs, PPTs)
- **Rate limiting**: reads `X-Rate-Limit-Remaining` header; slows down at threshold
- **Retry logic**: 3 retries with exponential backoff `[1s, 5s, 15s]` for server errors
- Custom exceptions: `RateLimitError`, `AuthenticationError`, `NotFoundError`

### `src/ingestion/document_processor.py` — Document Processor
- **HTML** — uses `BeautifulSoup` (lxml parser) to strip scripts/styles, converts tables to pipe-separated text (critical for schedule/syllabus data), then clean text
- **PDF** — `pypdf` library, page-by-page extraction
- **PPTX** — `python-pptx`, iterates slides and shapes
- **Chunking**: word-based sliding window with configurable size (default **500 words**) and overlap (default **50 words**). Each chunk carries full metadata.

### `src/ingestion/content_handlers/` — Handler Pattern
- Abstract base class `BaseContentHandler` with methods `fetch_content()`, `extract_metadata()`, `get_content_text()`, and a concrete `process_content()` pipeline
- Separate handlers for: `page`, `assignment`, `announcement`, `discussion`, `file`, `syllabus`, `module`
- Each handler extracts rich metadata: `content_id`, `content_type`, `course_id`, `title`, `url`, `created_at`, `updated_at`, `ingested_at`

### `src/ingestion/metadata_tracker.py` — Incremental Updates
- Persists ingestion state to `data/metadata/ingestion_state.json`
- `should_process_item(content_id, updated_at)` — compares ISO 8601 timestamps to skip unchanged content
- Supports marking items as deleted for cleanup
- Enables `--incremental` flag to only re-ingest changed/new content

### `src/embedding/embedder.py` — Embedder
- Uses **`sentence-transformers`** library
- Default model: **`all-MiniLM-L6-v2`** (384-dimensional embeddings, fast, good quality)
- `embed_text()` for single queries, `embed_batch()` for bulk ingestion with progress bar
- Model is loaded once and reused

### `src/vectorstore/chroma_manager.py` — Vector Store
- Uses **ChromaDB** with `PersistentClient` (data survives restarts, stored at `data/chroma_db/`)
- Collection metric: **cosine similarity** (`hnsw:space: cosine`)
- `add_documents()` — generates embeddings then bulk-adds to ChromaDB in batches of 100
- `add_documents_with_ids()` — custom IDs for update support
- `update_content()` — delete old chunks then add new chunks (supports incremental update)
- `delete_by_content_id()` — deletes all chunk IDs for a content item
- IDs are structured as `{content_id}_chunk_{chunk_index}`

### `src/retrieval/retriever.py` — Retriever
- Embeds the query, queries ChromaDB for top-K results
- **Similarity threshold** (default 0.2 — intentionally low to capture more relevant content)
- **Date keyword boosting**: extracts date patterns (e.g., "April 23", "Mar 5") using regex, retrieves 3x more results, then boosts scores by +0.3 if the date appears in the text chunk
- Supports `course_filter` to scope queries to a specific course via metadata filtering

### `src/generation/generator.py` — Generator
- Sends prompts to **Ollama** (`http://localhost:11434/api/generate`) by default
- **Two-step RAG generation**:
  1. First asks LLM for 2-3 sentences of background context on the question
  2. Then constructs a full prompt combining: Canvas retrieved chunks + LLM background + student question
- Prompt engineering includes specific handlers for: schedule/date questions, people/instructor questions, concept questions, policy/assignment questions
- Instructs the LLM to prioritize Canvas materials over general knowledge

### `src/config.py` — Configuration
All settings via `.env` file or environment variables:

| Setting | Default | Purpose |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `CHUNK_SIZE` | 500 words | Chunk size |
| `CHUNK_OVERLAP` | 50 words | Overlap between chunks |
| `TOP_K_RESULTS` | 5 | Docs retrieved per query |
| `SIMILARITY_THRESHOLD` | 0.2 | Min similarity to include |
| `LLM_PROVIDER` | `ollama` | `ollama`, `openai`, or `groq` |
| `OLLAMA_MODEL` | `gemma:2b` | Local LLM model |

---

## 4. Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Canvas API | `requests` (REST, Bearer token) |
| HTML parsing | `BeautifulSoup` + `lxml` |
| PDF extraction | `pypdf` |
| PPTX extraction | `python-pptx` |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector DB | `ChromaDB` (local persistent) |
| LLM | Ollama (`gemma:2b`) / OpenAI GPT-4o-mini / Groq Llama-3 |
| Config | `python-dotenv` |

---

## 5. Data Flow — Full Pipeline

**Ingestion:**
1. `CanvasClient` fetches raw JSON from Canvas REST API (with pagination)
2. Content handlers extract text + metadata per content type
3. `DocumentProcessor` cleans HTML/PDF/PPTX to plain text
4. Text is chunked into 500-word overlapping windows
5. `Embedder` converts chunks to 384-dim vectors
6. `ChromaManager` stores vectors + text + metadata in ChromaDB
7. `MetadataTracker` records what was ingested

**Query:**
1. User asks a question
2. `Retriever` embeds the question, queries ChromaDB, applies threshold + date boosting
3. Top-K chunks retrieved with similarity scores
4. `Generator` builds a structured prompt and sends to Ollama
5. Answer returned to user

---

## 6. Key Design Decisions

**Why ChromaDB?**
Local, persistent, no external service needed. Easy to reset/rebuild. Good for prototyping and fully offline use.

**Why `all-MiniLM-L6-v2`?**
Fast, small (80MB), runs entirely locally, well-suited for sentence-level semantic similarity. 384-dim keeps storage small.

**Why chunking with overlap?**
Overlap (50 words) prevents answers being split across chunk boundaries — important for sentences that span the end of one chunk.

**Why tables get special treatment in HTML parsing?**
Canvas syllabi often contain weekly schedules as HTML tables. Converting them to pipe-separated text (`Week 1 | Mon, Jan 13 | Intro | Chapter 1`) preserves the structure so the LLM can parse it correctly.

**Why date boosting in the retriever?**
Semantic similarity alone struggles with exact date matching. A chunk about "April 23" topics may not be semantically close to the query "What is taught on April 23?" — the keyword boost compensates for this weakness.

**Why `--incremental` mode?**
Full ingestion of a large course is slow. Tracking `updated_at` timestamps from Canvas means only changed content needs re-embedding.

**Why multi-provider LLM support?**
The config supports Ollama (local/free), OpenAI, and Groq — making it flexible for local development vs. production deployments.

---

## 7. Project Structure

```
SimpleRAG/
├── src/
│   ├── config.py                    # All settings from .env
│   ├── embedding/embedder.py        # sentence-transformers wrapper
│   ├── vectorstore/chroma_manager.py # ChromaDB CRUD operations
│   ├── retrieval/retriever.py       # Semantic search + date boosting
│   ├── generation/generator.py      # LLM prompt + answer generation
│   └── ingestion/
│       ├── canvas_client.py         # Canvas REST API client
│       ├── document_processor.py    # HTML/PDF/PPTX -> chunks
│       ├── metadata_tracker.py      # Incremental update state
│       └── content_handlers/        # One handler per content type
│           ├── base_handler.py
│           ├── page_handler.py
│           ├── assignment_handler.py
│           ├── syllabus_handler.py
│           ├── announcement_handler.py
│           ├── discussion_handler.py
│           ├── file_handler.py
│           └── module_handler.py
├── scripts/
│   ├── ingest_data.py               # CLI ingestion script
│   ├── list_courses.py
│   └── verify_canvas_data.py
├── data/
│   ├── chroma_db/                   # Persistent vector store
│   ├── metadata/                    # Ingestion state JSON
│   └── logs/
└── README.md
```

---

## 8. Common Interview Questions

**Q: What is RAG?**
Retrieval-Augmented Generation. Instead of relying purely on an LLM's training data, you first retrieve relevant documents from a knowledge base using semantic search, then pass them as context to the LLM. This grounds answers in real, up-to-date, domain-specific data and reduces hallucinations.

**Q: How do embeddings work here?**
Both the stored chunks and the user's query are encoded into fixed-size vectors using the same sentence transformer model. Similarity is measured by cosine distance in that vector space — semantically similar text will have vectors that point in roughly the same direction.

**Q: What is cosine similarity vs. L2 distance?**
Cosine similarity measures the angle between vectors (direction, not magnitude). L2 (Euclidean) measures absolute distance. For sentence embeddings, cosine is preferred because it's scale-invariant — it compares meaning, not length of text.

**Q: How does ChromaDB store the vectors?**
It uses HNSW (Hierarchical Navigable Small World) graphs — an approximate nearest neighbor index that allows very fast similarity search at scale without scanning every vector.

**Q: What are the limitations of this system?**
1. Keyword-heavy queries may not match semantically similar content
2. Very long documents may lose context across chunk boundaries
3. The `gemma:2b` model is small — answers may be lower quality than GPT-4
4. No conversation history — each query is independent
5. Rate limits on the Canvas API slow down large course ingestion

**Q: How would you improve this?**
1. Hybrid search (BM25 keyword + semantic) for better recall
2. Re-ranking retrieved chunks with a cross-encoder
3. Conversation memory / chat history
4. Streaming responses for better UX
5. Sentence-aware or paragraph-aware chunking instead of fixed word count
6. Evaluation pipeline (e.g., RAGAS) to measure retrieval and answer quality
