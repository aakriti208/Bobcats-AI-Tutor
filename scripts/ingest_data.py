#!/usr/bin/env python3
"""
Canvas Data Ingestion Script

Fetches Canvas LMS content, processes it through the existing handler pipeline,
and stores it in a LlamaIndex VectorStoreIndex backed by ChromaDB — ready for
the Streamlit app to query via Ollama.

Usage:
    python scripts/ingest_data.py --full              # Full sync
    python scripts/ingest_data.py --incremental       # Only new/updated (default)
    python scripts/ingest_data.py --course 2295372    # Single course
    python scripts/ingest_data.py --content-type page # Specific type
    python scripts/ingest_data.py --reset             # Clear index and ChromaDB first
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# --- LlamaIndex setup (must happen before other llama_index imports) ---
from llama_index.core import Settings as LlamaSettings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.config import EMBEDDING_MODEL
LlamaSettings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
LlamaSettings.llm = None  # LLM not needed during ingestion
LlamaSettings.chunk_size = 8192
LlamaSettings.chunk_overlap = 100
# ----------------------------------------------------------------------

import chromadb as chromadb_client
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.ingestion.canvas_client import CanvasClient
from src.ingestion.document_processor import DocumentProcessor
from src.ingestion.metadata_tracker import MetadataTracker
from src.ingestion.content_handlers import (
    PageHandler,
    AssignmentHandler,
    AnnouncementHandler,
    DiscussionHandler,
    FileHandler,
    SyllabusHandler
)
from src.config import (
    CANVAS_API_TOKEN,
    CANVAS_BASE_URL,
    CANVAS_COURSE_IDS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHROMA_DB_DIR,
    STORAGE_DIR,
    LOG_DIR,
    LOG_LEVEL,
)

CHROMA_COLLECTION_NAME = "course_content"


def setup_logging():
    from datetime import datetime
    log_file = LOG_DIR / f"ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('llama_index').setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")


logger = logging.getLogger(__name__)


def _get_chroma_collection():
    """Return a persistent ChromaDB collection."""
    client = chromadb_client.PersistentClient(path=str(CHROMA_DB_DIR))
    return client.get_or_create_collection(
        CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def _load_or_create_index() -> VectorStoreIndex:
    """Load existing LlamaIndex index from storage or create a fresh one."""
    vector_store = ChromaVectorStore(chroma_collection=_get_chroma_collection())
    index_store_path = STORAGE_DIR / "index_store.json"

    if index_store_path.exists():
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=str(STORAGE_DIR)
        )
        index = load_index_from_storage(storage_context)
        logger.info("Loaded existing LlamaIndex index from storage")
    else:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex([], storage_context=storage_context)
        index.storage_context.persist(persist_dir=str(STORAGE_DIR))
        logger.info("Created new LlamaIndex index")

    return index


def _chunks_to_nodes(chunks: list) -> list:
    """
    Convert SimpleRAG chunk dicts to LlamaIndex TextNode objects.

    Each node gets a stable ID (<content_id>_chunk_<chunk_index>) so that
    re-ingesting the same content is idempotent at the ChromaDB level.
    """
    nodes = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        # LlamaIndex metadata values must be strings
        clean_meta = {k: str(v) for k, v in meta.items() if v is not None and v != ""}
        node = TextNode(
            text=chunk["text"],
            metadata=clean_meta,
            id_=f"{meta['content_id']}_chunk_{meta['chunk_index']}",
        )
        nodes.append(node)
    return nodes


class CanvasIngestionPipeline:
    """Orchestrates Canvas data ingestion into a LlamaIndex + ChromaDB index."""

    def __init__(self, incremental=True):
        logger.info("Initializing Canvas ingestion pipeline...")

        self.client = CanvasClient(CANVAS_API_TOKEN, CANVAS_BASE_URL)
        self.processor = DocumentProcessor(CHUNK_SIZE, CHUNK_OVERLAP)
        self.tracker = MetadataTracker()
        self.index = _load_or_create_index()

        self.handlers = {
            'syllabus':     SyllabusHandler(self.client, self.processor),
            'page':         PageHandler(self.client, self.processor),
            'assignment':   AssignmentHandler(self.client, self.processor),
            'announcement': AnnouncementHandler(self.client, self.processor),
            'discussion':   DiscussionHandler(self.client, self.processor),
            'file':         FileHandler(self.client, self.processor),
        }

        self.incremental = incremental
        logger.info(f"Pipeline initialized (incremental={incremental})")

    def ingest_course(self, course_id: str, content_types: list = None):
        """
        Fetch and index content from a single Canvas course.

        Returns:
            Stats dict with chunk counts per content type
        """
        logger.info("=" * 70)
        logger.info(f"Ingesting course: {course_id}")
        logger.info("=" * 70)

        try:
            course_info = self.client.get_course(course_id)
            course_name = course_info.get('name', 'Unknown Course')
            logger.info(f"Course name: {course_name}")
        except Exception as e:
            logger.error(f"Error fetching course info: {e}")
            course_name = f"Course {course_id}"

        handlers_to_run = content_types or list(self.handlers.keys())
        logger.info(f"Processing content types: {', '.join(handlers_to_run)}")

        all_chunks = []
        stats = {h: 0 for h in handlers_to_run}

        for handler_name in handlers_to_run:
            if handler_name not in self.handlers:
                logger.warning(f"Unknown handler: {handler_name}")
                continue

            logger.info(f"\nProcessing {handler_name}s...")
            handler = self.handlers[handler_name]

            try:
                chunks = handler.process_content(course_id, course_name)

                if self.incremental:
                    chunks = self._filter_incremental(chunks)
                    logger.info(f"After incremental filter: {len(chunks)} chunks")

                all_chunks.extend(chunks)
                stats[handler_name] = len(chunks)

            except Exception as e:
                logger.error(f"Error processing {handler_name}s: {e}", exc_info=True)
                continue

        # Convert to LlamaIndex nodes and insert into index
        if all_chunks:
            logger.info(f"\nConverting {len(all_chunks)} chunks to LlamaIndex nodes...")
            nodes = _chunks_to_nodes(all_chunks)

            logger.info(f"Inserting {len(nodes)} nodes into index (ChromaDB)...")
            self.index.insert_nodes(nodes)
            self.index.storage_context.persist(persist_dir=str(STORAGE_DIR))
            self.tracker.update_course_sync(course_id)
            logger.info(f"Index persisted to {STORAGE_DIR}")
        else:
            logger.info("No new content to add")

        logger.info("\n" + "=" * 70)
        logger.info(f"Course {course_id} ingestion complete!")
        logger.info(f"Total chunks: {len(all_chunks)}")
        for handler, count in stats.items():
            logger.info(f"  - {handler}: {count}")
        logger.info("=" * 70)

        return stats

    def _filter_incremental(self, chunks: list) -> list:
        if not chunks:
            return []

        filtered = []
        processed_ids = set()

        for chunk in chunks:
            content_id = chunk['metadata']['content_id']
            updated_at = chunk['metadata'].get('updated_at', '')

            if self.tracker.should_process_item(content_id, updated_at):
                filtered.append(chunk)

                if content_id not in processed_ids:
                    self.tracker.mark_item_processed(
                        content_id,
                        chunk['metadata']['content_type'],
                        updated_at,
                        chunk['metadata']['total_chunks']
                    )
                    processed_ids.add(content_id)

        return filtered

    def ingest_all_courses(self, content_types: list = None):
        overall_stats = {}

        for course_id in CANVAS_COURSE_IDS:
            if not course_id or not course_id.strip():
                continue
            try:
                stats = self.ingest_course(course_id, content_types)
                overall_stats[course_id] = stats
            except Exception as e:
                logger.error(f"Error ingesting course {course_id}: {e}", exc_info=True)
                continue

        return overall_stats


def main():
    parser = argparse.ArgumentParser(description='Ingest Canvas LMS data into LlamaIndex + ChromaDB')
    parser.add_argument('--full', action='store_true',
                        help='Full sync (re-process everything)')
    parser.add_argument('--incremental', action='store_true', default=True,
                        help='Incremental sync (only new/updated, default)')
    parser.add_argument('--course', type=str,
                        help='Specific course ID to ingest')
    parser.add_argument('--content-type', type=str, nargs='+',
                        help='Specific content types (e.g. page assignment)')
    parser.add_argument('--reset', action='store_true',
                        help='Wipe ChromaDB and LlamaIndex storage before ingestion')

    args = parser.parse_args()
    setup_logging()

    logger.info("=" * 70)
    logger.info("CANVAS DATA INGESTION  (LlamaIndex + ChromaDB + Ollama)")
    logger.info("=" * 70)
    logger.info(f"Canvas Base URL: {CANVAS_BASE_URL}")
    logger.info(f"Configured Courses: {', '.join(CANVAS_COURSE_IDS)}")
    logger.info(f"Mode: {'Full' if args.full else 'Incremental'}")
    logger.info(f"Embedding model: {EMBEDDING_MODEL}")
    logger.info("=" * 70)

    # --reset: wipe ChromaDB collection and LlamaIndex storage
    if args.reset:
        logger.warning("Resetting ChromaDB collection and LlamaIndex storage...")
        client = chromadb_client.PersistentClient(path=str(CHROMA_DB_DIR))
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
            logger.info("ChromaDB collection deleted")
        except Exception:
            pass
        if STORAGE_DIR.exists():
            shutil.rmtree(STORAGE_DIR)
            STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("LlamaIndex storage directory cleared")

    pipeline = CanvasIngestionPipeline(incremental=not args.full)

    if args.reset:
        pipeline.tracker.reset()

    try:
        if args.course:
            logger.info(f"Ingesting single course: {args.course}")
            pipeline.ingest_course(args.course, args.content_type)
        else:
            logger.info("Ingesting all configured courses")
            pipeline.ingest_all_courses(args.content_type)

        # Final stats
        tracker_stats = pipeline.tracker.get_stats()
        doc_count = _get_chroma_collection().count()

        logger.info("\n" + "=" * 70)
        logger.info("INGESTION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total documents in ChromaDB: {doc_count}")
        logger.info(f"Total content items tracked: {tracker_stats['total_items']}")
        logger.info(f"Active items: {tracker_stats['active_items']}")
        logger.info(f"Last full sync: {tracker_stats['last_full_sync'] or 'Never'}")
        logger.info("\nBy content type:")
        for content_type, count in tracker_stats.get('by_type', {}).items():
            logger.info(f"  - {content_type}: {count}")
        logger.info(f"\nLlamaIndex index persisted at: {STORAGE_DIR}")
        logger.info("=" * 70)

    except KeyboardInterrupt:
        logger.info("\n\nIngestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nFatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
