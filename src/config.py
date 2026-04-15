import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

STORAGE_DIR = DATA_DIR / "storage"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHROMA_DB_DIR, STORAGE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Canvas API settings
CANVAS_API_TOKEN = os.getenv("CANVAS_API_TOKEN", "")
CANVAS_BASE_URL = os.getenv("CANVAS_BASE_URL", "https://your-institution.instructure.com")
CANVAS_COURSE_IDS = os.getenv("CANVAS_COURSE_IDS", "").split(",")

# Embedding settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ChromaDB settings
CHROMA_COLLECTION_NAME = "course_content"
CHROMA_DISTANCE_METRIC = "cosine"

# Chunking settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# LLM settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")   # "ollama" or "openai"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma:2b")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Retrieval settings
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))  # Increased from 3 to retrieve more context
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.2"))  # Lowered to 0.2 to capture more relevant content

# Ingestion settings
INGEST_BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "100"))
INGEST_MAX_WORKERS = int(os.getenv("INGEST_MAX_WORKERS", "4"))
INGEST_RATE_LIMIT_THRESHOLD = int(os.getenv("INGEST_RATE_LIMIT_THRESHOLD", "100"))

# File processing settings
PDF_EXTRACTION_TIMEOUT = int(os.getenv("PDF_EXTRACTION_TIMEOUT", "30"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

# Content type filtering (comma-separated)
# Note: 'module' removed - modules are just navigation metadata, actual content is in pages
INGEST_CONTENT_TYPES = os.getenv(
    "INGEST_CONTENT_TYPES",
    "syllabus,page,assignment,announcement,discussion,file"
).split(",")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = DATA_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Metadata tracking
METADATA_DIR = DATA_DIR / "metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)
