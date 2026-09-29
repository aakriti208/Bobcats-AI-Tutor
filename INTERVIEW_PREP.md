# Interview Preparation: Canvas RAG AI Tutor ("Bobcat AI Tutor" / "SimpleRAG")

---

## 1. Why Was This Project Built? What Problem Does It Solve?

**Problem:** Students using Canvas LMS (the online course management platform) struggle to quickly find answers buried across dozens of pages, PDFs, PowerPoints, assignments, and announcements. Manually searching through course content is slow and frustrating.

**Solution:** An AI-powered tutor that ingests all content from a Canvas course and lets students ask natural language questions — receiving answers grounded in the actual course materials, not generic internet knowledge.

**Key differentiator:** Uses Retrieval-Augmented Generation (RAG) so the LLM answers are tied to specific, course-specific materials (syllabus, lecture slides, assignments, etc.) rather than hallucinating from general training data.

**Target institution:** Texas State University (the UI uses "Bobcat AI Tutor" branding with the Supercat logo and TXST color scheme).

---

## 2. End-to-End Pipeline / Architecture

### Ingestion Pipeline (offline, run once or incrementally):
```
Canvas API
    |
    v
CanvasClient          -- HTTP requests with pagination, rate-limit handling, retries
    |
    v
ContentHandlers       -- 6 handlers: SyllabusHandler, PageHandler, AssignmentHandler,
                         AnnouncementHandler, DiscussionHandler, FileHandler
    |
    v
DocumentProcessor     -- HTML cleaning (BeautifulSoup/lxml), PDF extraction (pypdf),
                         PPTX extraction (python-pptx), word-level text chunking
    |
    v
MetadataTracker       -- Tracks content IDs + updated_at timestamps for incremental sync
    |
    v
Embedder              -- sentence-transformers/all-MiniLM-L6-v2 (384-dim embeddings)
    |
    v
ChromaDB              -- Persistent vector store with cosine distance (HNSW index)
```

### Query Pipeline (real-time, per user question):
```
User Question (via Web UI or CLI)
    |
    v
Embedder              -- Embeds query using same model
    |
    v
ChromaDB              -- ANN search, returns top-K chunks with distances
    |
    v
Retriever             -- Converts distances to similarities, applies date keyword boost,
                         filters by similarity threshold, re-ranks and returns top-K
    |
    v
Generator             -- Two parallel Ollama calls:
                         (1) generate_without_rag: LLM answers from general knowledge
                         (2) generate_with_rag: LLM uses retrieved Canvas chunks as context
    |
    v
FastAPI /query        -- Returns structured RAGResponse JSON
    |
    v
Web UI (index.html)   -- Side-by-side comparison: "Bobcat AI Tutor" vs "Local LLM"
```

### Key architectural diagram (from README):
```
Canvas API -> CanvasClient -> ContentHandlers -> DocumentProcessor -> Embedder -> ChromaDB
User Query -> Embedder -> ChromaDB -> Retriever -> Generator -> Answer
```

---

## 3. Data: Source, Scale, Processing

**Source:** Canvas LMS API (REST API v1), authenticated via Bearer token. Connects to a specific institution's Canvas instance (Texas State University: `canvas.txstate.edu` is referenced in code).

**Content types ingested:**
| Type | Handler | What it fetches |
|---|---|---|
| Syllabus | SyllabusHandler | Course syllabus body (HTML) |
| Pages | PageHandler | All Canvas wiki pages (HTML) |
| Assignments | AssignmentHandler | Assignment descriptions + rubrics |
| Announcements | AnnouncementHandler | Instructor announcements (HTML) |
| Discussions | DiscussionHandler | Discussion topics + entries |
| Files | FileHandler | PDFs and PowerPoint files (binary) |

**Module handler was intentionally disabled** (commented out in code): "Modules are just navigation metadata, actual content is in pages" — a deliberate design decision.

**File size limit:** 50 MB per file (configurable via `MAX_FILE_SIZE_MB`). Files exceeding this are skipped.

**Supported file types:** `application/pdf`, `.pptx` (`application/vnd.openxmlformats-officedocument.presentationml.presentation`), `.ppt`. Other file types (Word docs, images, etc.) are skipped.

**Scale:** Not found in codebase — no logs, metrics, or reports showing actual document counts. The code supports multiple courses via `CANVAS_COURSE_IDS` (comma-separated). The ChromaDB `/health` endpoint reports `knowledge_base_documents` count at runtime.

**Data cleaning / processing:**
1. **HTML extraction:** BeautifulSoup with lxml parser (fallback to html.parser, then raw regex tag stripping). Removes `<script>`, `<style>`, `<meta>`, `<link>` tags. Tables are converted to pipe-delimited text (`col1 | col2 | col3`) to preserve structure for schedule/date questions.
2. **PDF extraction:** `pypdf` library, page-by-page extraction.
3. **PPTX extraction:** `python-pptx`, shape-by-shape text extraction per slide.
4. **Chunking:** Word-level sliding window — **500 words per chunk, 50-word overlap** (configurable). Short documents become a single chunk.
5. **Metadata attached to each chunk:** `content_id`, `content_type`, `course_id`, `course_name`, `title`, `source`, `url`, `created_at`, `updated_at`, `ingested_at`, `chunk_index`, `total_chunks`.

---

## 4. Technologies, Models, Algorithms, and Tools

| Component | Technology | Why |
|---|---|---|
| **Embedding model** | `sentence-transformers/all-MiniLM-L6-v2` | Lightweight (384-dim), fast, good semantic similarity for English text, runs locally |
| **Vector database** | ChromaDB (PersistentClient) | Local/embeddable, no external service needed, supports metadata filtering, HNSW ANN index |
| **Distance metric** | Cosine distance | Standard for sentence embeddings; rotation-invariant, good for semantic similarity |
| **LLM** | Ollama + `gemma:2b` | Local inference, no API costs, privacy-preserving (student data stays on-premise), Google's Gemma 2B is small enough to run on consumer hardware |
| **Web framework** | FastAPI + Uvicorn | Async, fast, auto-generates OpenAPI docs, Pydantic validation |
| **Canvas API client** | Custom `requests`-based client | Canvas REST API v1; custom pagination, retry, and rate-limit handling |
| **HTML parsing** | BeautifulSoup4 + lxml | Robust HTML extraction from Canvas rich-text content |
| **PDF parsing** | pypdf | Pure-Python PDF text extraction |
| **PPTX parsing** | python-pptx | Official Python library for PowerPoint |
| **Incremental sync** | Custom MetadataTracker (JSON file) | Tracks `updated_at` timestamps to avoid re-ingesting unchanged content |
| **Frontend** | Vanilla HTML/CSS/JS | No build toolchain needed; simple deployment |

**Algorithm — Retrieval with date boosting:**
- Standard cosine similarity ANN search in ChromaDB
- **Date keyword extraction** using regex (`Jan|Feb|...|December \d{1,2}`): if the query contains a date, `top_k` is multiplied by 3 to retrieve more candidates, then re-ranked with a +0.3 similarity boost for chunks containing that date string. The similarity threshold is also lowered by 0.1 for date queries.

**Default hyperparameters (from `config.py`):**
- `TOP_K_RESULTS = 5`
- `SIMILARITY_THRESHOLD = 0.2`
- `CHUNK_SIZE = 500` words
- `CHUNK_OVERLAP = 50` words
- `INGEST_BATCH_SIZE = 100`

---

## 5. My Specific Contribution

The entire codebase is authored by you (GitHub: `aakriti208/SimpleRAG`). Based on git history and code structure, contributions include:

- Designed and implemented the full RAG pipeline from scratch (ingestion, embedding, retrieval, generation)
- Built the Canvas API client with pagination, rate limiting (X-Rate-Limit-Remaining header), and retry logic
- Implemented 6 content-type handlers using an abstract base class (OOP design pattern)
- Designed the DocumentProcessor for multi-format extraction (HTML, PDF, PPTX)
- Implemented incremental sync via MetadataTracker (JSON state file, ISO 8601 timestamp comparison)
- Built the ChromaDB vector store manager with batch embedding, custom IDs for update support
- Engineered the retrieval logic including the date keyword boosting heuristic
- Wrote the prompt engineering for the Generator (4 question-type routing: schedule/date, people, content/concept, policy/assignment)
- Built the FastAPI web interface with side-by-side RAG vs. non-RAG comparison
- Wrote installation scripts (bash for Unix/Mac, .bat for Windows) with pyenv support
- Iterated on prompt quality (multiple commits: "Generator prompt refined", "Upgraded ollama model and improved prompt")

---

## 6. Biggest Technical Challenges and How They Were Solved

**Challenge 1: Canvas table content (schedule/date queries)**
- Problem: Canvas syllabus schedules are HTML tables. After naive HTML extraction, table structure was lost, making "What is taught on March 26?" queries fail.
- Solution: Special table handling in `DocumentProcessor.extract_html_text()` — tables are converted to pipe-delimited rows (`Week # | Day, Date | Topic | Reading | Notes`) before text extraction, preserving the structure in the vector store.
- Additional layer: Custom date keyword extraction in `Retriever._extract_date_keywords()` + +0.3 similarity boost + expanded query (top_k * 3) for date queries.
- Prompt engineering: The Generator prompt explicitly instructs: "Find the date in the table format: Week # | Day, Date | Topic... Extract ONLY the Topic column."

**Challenge 2: Canvas modules are navigation metadata, not content**
- Problem: Early versions ingested modules, but modules are just containers/links — the actual text content lives in pages.
- Solution: `ModuleHandler` was explicitly disabled (commented out in `ingest_data.py` and `__init__.py` with an explanatory comment). This was a conscious design decision after testing.

**Challenge 3: Avoiding redundant re-ingestion**
- Problem: Re-running ingestion would duplicate all documents in ChromaDB.
- Solution: `MetadataTracker` stores content IDs and `updated_at` timestamps in a JSON file. `should_process_item()` compares ISO 8601 strings lexicographically. Only new/modified content is re-ingested. Custom chunk IDs (`{content_id}_chunk_{index}`) enable `update_content()` to delete old chunks before adding new ones.

**Challenge 4: Rate limiting on Canvas API**
- Problem: Canvas API enforces rate limits (HTTP 429/403 with `Retry-After` header).
- Solution: `CanvasClient._make_request()` checks the `X-Rate-Limit-Remaining` header and slows down proactively (threshold: 100 requests remaining → 2s sleep). On 403, it reads `Retry-After` and waits. Exponential-ish retry delays: [1, 5, 15] seconds.

**Challenge 5: Form submission URL encoding bug**
- Problem: Commit `52e18d7` — "Fixed form submission JavaScript to prevent URL parameter issues." The form was likely submitting as GET with query params instead of POST with JSON body.
- Solution: JavaScript form handler was fixed (exact details in `static/app.js`).

**Challenge 6: LLM hallucinating vs. using Canvas context**
- Problem: `gemma:2b` would ignore retrieved context and answer from general knowledge.
- Solution: Prompt engineering — the prompt explicitly says "IMPORTANT: I have searched the Canvas course materials... You MUST use this Canvas information... Do NOT rely on your general knowledge if the Canvas materials contain the answer." Context is labeled as "=== CANVAS COURSE MATERIALS ===" to make it prominent.

---

## 7. Main Bottlenecks and Limitations

**Bottlenecks:**
- **LLM inference speed:** `gemma:2b` runs locally on CPU (or GPU if available). Each query requires 2 Ollama calls (with RAG + without RAG). Response time depends on hardware.
- **Ingestion time:** Downloading binary files (PDFs, PPTXs) from Canvas is sequential (no async downloads). Batch embedding is done with `sentence-transformers` which is CPU-bound without GPU.
- **ChromaDB ANN search:** HNSW index is in-memory during queries but persisted to disk. For very large collections, load time increases.

**Limitations:**
- **`gemma:2b` is a small model:** May produce lower-quality answers, misunderstand complex questions, or fail to follow prompt instructions reliably. This is a known quality tradeoff for local/privacy-preserving inference.
- **Only PDF and PPTX files are supported:** Word documents, spreadsheets, images, videos are skipped. Canvas content in non-text formats is not indexed.
- **Word-level chunking ignores semantic boundaries:** Chunks are split by word count, not by paragraphs or sections. A chunk may cut across a conceptual boundary.
- **No re-ranking:** After initial ANN retrieval, results are only re-ranked by the date boost heuristic. No cross-encoder re-ranking.
- **No user authentication:** The web app has no login — anyone with access to port 8000 can query all ingested course materials.
- **Similarity threshold of 0.2 is very low:** Risks returning low-quality/irrelevant context. This was explicitly lowered (comment: "Lowered to 0.2 to capture more relevant content") suggesting precision/recall tradeoff was a known issue.
- **No streaming:** Ollama responses are fetched with `stream: False`, so the UI waits for the full response before displaying.
- **MetadataTracker is file-based (JSON):** Not suitable for concurrent ingestion of multiple courses simultaneously.

---

## 8. Scale and Scope

- **Courses:** Multiple courses supported via `CANVAS_COURSE_IDS` environment variable.
- **Content types:** 6 types (syllabus, pages, assignments, announcements, discussions, PDF/PPTX files).
- **File size cap:** 50 MB per file.
- **Chunk batch size:** 100 chunks per ChromaDB batch insert.
- **Max workers:** 4 (configured but not actually used with threading in current code — `INGEST_MAX_WORKERS` is set but the ingestion loop is sequential).
- **Actual document counts:** Not found — no logs or reports committed to the repo.

**Scope:** Single institution (Texas State University), designed for course-specific Q&A, not a general-purpose search engine.

---

## 9. Prototype, Research Project, or Production System?

**This is a prototype / proof-of-concept.**

Evidence:
- Named "SimpleRAG" in repo (`github.com/aakriti208/SimpleRAG`) and in `app.py` docstring
- No tests (no `tests/` directory, no pytest, no test files)
- No Docker/container setup, no CI/CD pipeline
- No authentication or access control on the web interface
- `gemma:2b` is a research/demo model, not production-grade
- `INGEST_MAX_WORKERS=4` configured but not implemented (loop is single-threaded)
- Similarity threshold of 0.2 with a comment saying it was "Lowered... to capture more relevant content" — a tuning hack, not a principled setting
- No evaluation metrics, no test set, no automated quality checks
- Development timeline: ~4 months (Dec 2025 – Apr 2026) with intermittent commits
- "wip" (work in progress) commits visible in git log

---

## 10. Evaluation: Metrics and Methodology

**Not found in the codebase.** No formal evaluation was implemented.

What exists:
- A side-by-side UI comparing "RAG answer" vs. "LLM-only answer" — this is a qualitative, human-in-the-loop evaluation mechanism, not automated metrics.
- The `rag_demo.py` CLI shows similarity scores for retrieved chunks, enabling manual inspection.
- Context sources and similarity scores are displayed in the web UI ("Retrieved Canvas Materials" section with similarity values per chunk).

**What's missing (and you should acknowledge if asked):**
- No RAGAS, TruLens, or similar RAG evaluation framework
- No held-out question set with ground-truth answers
- No precision/recall measurements at any K
- No measurement of answer faithfulness, context relevance, or answer relevance
- No A/B testing between `gemma:2b` and other models

---

## 11. Actual Results

**Not found** — no benchmark results, evaluation reports, or logged performance metrics in the codebase.

If asked, be honest: "We validated it qualitatively by testing with real course questions. The side-by-side view made it clear that RAG answers were more course-specific and accurate for questions about schedules, assignments, and course policies. We didn't run formal quantitative evaluation."

---

## 12. Technical Tradeoffs and Design Decisions to Explain

**1. Local LLM (Ollama + gemma:2b) vs. OpenAI API**
- Chose local: No API costs, student data stays on-premise (FERPA compliance concern), works offline.
- Tradeoff: Lower answer quality, slower inference, no streaming.

**2. ChromaDB vs. Pinecone/Weaviate/pgvector**
- Chose ChromaDB: Embedded/local, no external service, easy to set up, Python-native.
- Tradeoff: Not horizontally scalable, single-machine only.

**3. `all-MiniLM-L6-v2` vs. larger embedding models**
- Chose MiniLM: Fast, small (22M params, 384-dim), runs on CPU, good performance on sentence similarity benchmarks.
- Tradeoff: Lower representation quality than `all-mpnet-base-v2` or OpenAI `text-embedding-3-small`.

**4. Word-level chunking vs. semantic chunking**
- Chose word-level: Simple, predictable, deterministic.
- Tradeoff: Can cut across sentence/paragraph boundaries. A semantic chunker (sentence-aware) would be better but adds complexity.

**5. Disabled module handler**
- Canvas modules are navigation containers, not content. Ingesting them added noise (duplicate/empty chunks) without improving retrieval. Removed after testing.

**6. Dual-answer comparison UI**
- Design choice to show both RAG and non-RAG answers simultaneously. Serves as both a demo feature and a manual evaluation tool — users can immediately see the benefit of RAG.

**7. Incremental sync via MetadataTracker**
- Uses ISO 8601 string comparison (lexicographic) for `updated_at` timestamps. Avoids re-ingesting unchanged content. Trades complexity for efficiency on repeat runs.

**8. Similarity threshold set to 0.2**
- Originally higher (0.5 in `.env.template`). Lowered to improve recall for course-specific queries where semantic similarity to the question may be low even for relevant content (e.g., "What is due this week?" vs. a chunk about an assignment deadline).

---

## 13. What I Would Improve

1. **Add formal evaluation:** Implement RAGAS (Retrieval-Augmented Generation Assessment) with metrics: context precision, context recall, faithfulness, answer relevance. Create a test set of representative student questions with ground-truth answers.

2. **Upgrade the LLM:** Use `llama3.1:8b` or `mistral:7b` instead of `gemma:2b` for much better instruction-following and answer quality.

3. **Semantic/sentence-aware chunking:** Use `langchain.text_splitter.RecursiveCharacterTextSplitter` or a sentence splitter instead of word-count chunking to avoid cutting sentences mid-thought.

4. **Add cross-encoder re-ranking:** After initial ANN retrieval, use a cross-encoder (e.g., `ms-marco-MiniLM-L-6-v2`) to re-rank chunks by relevance to the specific query. Improves precision significantly.

5. **Streaming responses:** Enable Ollama streaming (`stream: True`) and use FastAPI `StreamingResponse` + Server-Sent Events on the frontend for faster perceived response time.

6. **Add authentication:** Protect the web interface with Canvas OAuth or simple API key authentication, especially since course materials may contain sensitive/copyrighted content.

7. **Implement hybrid search:** Combine dense (vector) retrieval with sparse (BM25 keyword) retrieval using a reciprocal rank fusion (RRF) approach. Especially helpful for exact-match queries (names, dates, specific assignment titles).

8. **Support more file types:** Word documents (`.docx`), Excel spreadsheets (course schedules), and transcripts from video content.

9. **Use a proper task queue for ingestion:** Replace the sequential ingestion loop with Celery or concurrent.futures for parallel course/content-type ingestion.

10. **Docker containerization:** Package the app + ChromaDB as a Docker Compose setup for repeatable deployment.

---

## 14. Likely Interview Questions

### Technical — RAG and ML:

**Q: What is RAG and why did you use it instead of just fine-tuning?**
A: RAG combines retrieval with generation. Instead of fine-tuning the LLM on course data (expensive, requires retraining when content changes, risks catastrophic forgetting), RAG retrieves relevant documents at query time and injects them as context. This means the knowledge base can be updated without touching the model, and answers are grounded in specific source documents.

**Q: Walk me through what happens when a user asks "What topics are covered in week 5?"**
A: (1) The question is embedded using `all-MiniLM-L6-v2` into a 384-dim vector. (2) The Retriever checks for date keywords — finds none. (3) ChromaDB runs ANN cosine search, returns top-5 chunks with similarity scores. (4) Chunks with similarity < 0.2 are filtered. (5) Generator makes 2 Ollama calls: one with context (RAG), one without. (6) The RAG prompt tells the LLM to check the Canvas materials first. (7) Response is returned as JSON to the frontend.

**Q: How does your chunking strategy work? What are its tradeoffs?**
A: Word-level sliding window: 500-word chunks with 50-word overlap. Overlap prevents important information from being split across two chunks. Tradeoff: it can cut mid-sentence. A better approach would be recursive character splitting with paragraph-aware boundaries.

**Q: What is cosine similarity and why use it for embeddings?**
A: Cosine similarity measures the angle between two vectors, ignoring magnitude. For sentence embeddings, it captures semantic direction rather than scale. It's preferred over Euclidean distance because embeddings of similar-length sentences that are semantically different will have similar magnitudes, making Euclidean distance misleading.

**Q: Why all-MiniLM-L6-v2 specifically?**
A: It's a distilled, 6-layer MiniLM model trained on a large sentence similarity dataset. It produces 384-dimensional embeddings — compact enough for fast CPU inference and low memory footprint, while ranking well on SBERT benchmarks. It's the standard "good enough, fast" choice for RAG prototypes.

**Q: How does your incremental sync work?**
A: `MetadataTracker` stores a JSON file with each content item's `updated_at` ISO 8601 timestamp. On each ingestion run, `should_process_item()` compares the Canvas API's `updated_at` for each item against the stored timestamp. If the Canvas timestamp is newer (lexicographic comparison works because ISO 8601 is ordered), the item is re-ingested. Old chunks for that item are deleted using content ID-based lookup before new chunks are added.

**Q: How did you handle Canvas rate limiting?**
A: Two layers: (1) Proactive — checks `X-Rate-Limit-Remaining` response header; if below threshold (100), sleeps 2 seconds. (2) Reactive — on HTTP 403, reads `Retry-After` header and waits that many seconds before retrying. Additionally, exponential retry delays [1, 5, 15] seconds for server errors (5xx).

**Q: What's the difference between the two answers shown in the UI?**
A: "Bobcat AI Tutor" (RAG answer) uses retrieved Canvas course materials as context in the prompt. "Local LLM" uses no context — just the LLM's general training knowledge. Side-by-side comparison helps users see when RAG adds value.

**Q: How would you evaluate this system?**
A: I'd use RAGAS framework with metrics: (1) Context Precision — are retrieved chunks actually relevant? (2) Context Recall — are all relevant chunks retrieved? (3) Faithfulness — does the answer actually come from the retrieved context? (4) Answer Relevance — does the answer address the question? I'd create a test set of ~50-100 representative student questions with ground-truth answers, then automate evaluation in CI.

### System Design / Architecture:

**Q: How would you scale this to 100 courses?**
A: (1) Move ChromaDB to a dedicated vector database service (e.g., Qdrant, Weaviate). (2) Use a task queue (Celery + Redis) for parallel ingestion. (3) Add course-based metadata filtering in ChromaDB so queries can be scoped to specific courses. (4) Consider sharding by course or using namespaces.

**Q: How would you handle concurrent users?**
A: FastAPI with async handlers already supports concurrent HTTP requests. The main bottleneck is Ollama inference — would need to either: (a) run multiple Ollama instances behind a load balancer, (b) switch to a hosted LLM API (OpenAI, Anthropic) that handles concurrency, or (c) implement a request queue.

**Q: What security concerns exist?**
A: (1) No authentication — anyone on the network can query all course materials. (2) Canvas API token is stored in `.env` — needs proper secret management in production. (3) FERPA compliance — student discussion posts ingested into the vector store; need to ensure only instructors/authorized students can query. (4) Prompt injection — a malicious user could craft a question to manipulate the LLM's behavior.

### Behavioral:

**Q: What was the hardest part of this project?**
A: The date/schedule query problem. Canvas stores syllabi as HTML tables, and standard text extraction destroyed the tabular structure. I had to add special table-to-pipe-delimited conversion in the HTML parser, then add a date keyword detection + similarity boost heuristic in the retriever, and finally add explicit prompt instructions telling the LLM how to read the table format. It required coordination across three different components of the pipeline.

**Q: What would you do differently if starting over?**
A: I'd invest in evaluation from day one — build a test set before writing the retrieval code so I could measure whether changes actually improved quality. I'd also start with a larger model (Llama 3.1 8B instead of Gemma 2B) since small model quality issues caused a lot of prompt engineering workarounds.

---

## CHEAT SHEET — Most Important Facts

| Category | Key Fact |
|---|---|
| **Project name** | Canvas RAG System / "Bobcat AI Tutor" / "SimpleRAG" |
| **Institution** | Texas State University (canvas.txstate.edu) |
| **Type** | Prototype / proof-of-concept RAG system |
| **Timeline** | Dec 2025 – Apr 2026 (~4 months) |
| **Stack** | Python 3.11, FastAPI, ChromaDB, Ollama, sentence-transformers |
| **Embedding model** | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, cosine) |
| **LLM** | Ollama `gemma:2b` (local, no API costs) |
| **Vector DB** | ChromaDB (HNSW, cosine distance, persistent) |
| **Chunk size** | 500 words, 50-word overlap |
| **Top-K** | 5 results (default) |
| **Similarity threshold** | 0.2 (lowered from 0.5 to improve recall) |
| **Content types** | syllabus, pages, assignments, announcements, discussions, PDF/PPTX files |
| **Excluded type** | Modules (navigation metadata only — intentionally excluded) |
| **Max file size** | 50 MB |
| **Incremental sync** | JSON MetadataTracker with ISO 8601 timestamp comparison |
| **Date query hack** | Regex extraction + 3x candidate retrieval + +0.3 similarity boost |
| **Web framework** | FastAPI + Uvicorn on port 8000 |
| **UI feature** | Side-by-side RAG vs. non-RAG answer comparison |
| **Evaluation** | None (qualitative only via side-by-side UI) |
| **No tests** | No test suite exists |
| **Rate limiting** | Proactive (X-Rate-Limit-Remaining header) + reactive (Retry-After) |
| **Retries** | 3 attempts with [1, 5, 15] second delays |

### Architecture in one sentence:
Canvas API content is fetched, HTML/PDF/PPTX is cleaned and chunked into 500-word segments, embedded with MiniLM, stored in ChromaDB; at query time the question is embedded, top-5 similar chunks are retrieved via cosine ANN search, and injected into a Gemma 2B prompt served via a FastAPI web interface.

### The one thing to emphasize:
This is a privacy-preserving, fully local RAG system — no student data leaves the institution's servers, no API costs, no external dependencies beyond the Canvas API itself.
