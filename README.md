# Canvas RAG System

A Retrieval-Augmented Generation (RAG) system that ingests educational content from Canvas LMS and enables intelligent question-answering using semantic search and LLMs.

## What It Does

- **Ingests** all content from Canvas courses (pages, modules, assignments, PDFs, PowerPoints, discussions)
- **Processes** and chunks content for optimal retrieval
- **Stores** in a persistent vector database (ChromaDB) with semantic embeddings
- **Answers** questions using course material via RAG with Ollama

## Prerequisites

- Python 3.11+
- Canvas LMS API token
- Ollama installed and running (for answer generation)

## Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file from the template:

```bash
cp .env.template .env
```

Edit `.env` with your Canvas credentials:

```env
CANVAS_API_TOKEN=your_canvas_api_token_here
CANVAS_BASE_URL=https://canvas.yourinstitution.edu/
CANVAS_COURSE_IDS=course_id1,course_id2
```

#### Getting Your Canvas API Token

1. Log in to Canvas and go to **Account** (top-left avatar)
2. Click **Settings**
3. Scroll down to **Approved Integrations** and click **+ New Access Token**
4. Enter a purpose (e.g. "Bobcat AI Tutor") and an optional expiry date
5. Click **Generate Token** and copy it as you won't be able to see it again

#### Finding Your Course ID

Open the course in Canvas — the number in the URL is the course ID:

```
https://canvas.yourinstitution.edu/courses/12345
                                             ^^^^^
                                          course ID
```

### 3. Start Ollama (for answer generation)

```bash
# Install Ollama from https://ollama.ai
# Then start the service
ollama serve

# Pull the model
ollama pull gemma:2b
```

## Usage

### Ingest Canvas Content

```bash
python scripts/ingest_data.py --course YOUR_COURSE_ID --content-type page
```

**Full ingestion (all content types):**

```bash
python scripts/ingest_data.py --course YOUR_COURSE_ID --full
```

**Ingest all configured courses:**

```bash
python scripts/ingest_data.py --full
```

**Incremental update (only new/changed content):**

```bash
python scripts/ingest_data.py --incremental
```

### Query Your Knowledge Base

**Interactive mode:**

```bash
python rag_demo.py
```

**Example questions:**

- "What is this course about?"
- "Explain text preprocessing"
- "What are the assignments in this course?"
- "Tell me about the syllabus"
