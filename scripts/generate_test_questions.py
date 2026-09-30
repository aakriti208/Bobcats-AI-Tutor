#!/usr/bin/env python3
"""
Synthetic Test Question Generator

Samples chunks from ChromaDB and uses Claude to generate realistic
student question-answer pairs, producing a ready-to-use test set for
scripts/evaluate.py without any manual question writing.

How it works:
  1. Pulls all chunks from ChromaDB
  2. Filters out very short or empty chunks
  3. Samples N chunks, spread evenly across content types
  4. Sends each chunk to Claude with a prompt asking it to generate
     one realistic student question + a reference answer grounded in
     that chunk's text
  5. Saves results to data/eval/test_questions_auto.json

Usage:
    python scripts/generate_test_questions.py
    python scripts/generate_test_questions.py --num-questions 50
    python scripts/generate_test_questions.py --num-questions 30 --content-type syllabus page
    python scripts/generate_test_questions.py --output data/eval/my_questions.json
"""

import argparse
import json
import logging
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vectorstore.chroma_manager import ChromaManager
from src.generation.generator import Generator
from src.config import LOG_DIR

# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging() -> None:
    log_file = LOG_DIR / f"generate_questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger("chromadb").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

_GENERATION_PROMPT = """You are building an evaluation dataset for a university course AI tutor.

Given the following excerpt from a Canvas course, generate ONE realistic question a student might ask, and a concise reference answer based strictly on this text.

COURSE CONTENT:
{chunk_text}

Source: {title} ({content_type})
Course: {course_name}

Requirements:
- The question must be something a real student would genuinely ask about this course
- The answer must be based strictly on the provided text — do not add outside knowledge
- Keep the answer concise (1–3 sentences)
- Avoid trivial questions like "What does this text discuss?" or "What is mentioned here?"
- Prefer factual, specific questions: deadlines, policies, concepts, people, grading, topics

Respond in this exact format with nothing else:
QUESTION: [your question]
ANSWER: [your answer]"""


# ── Core logic ────────────────────────────────────────────────────────────────

def _fetch_all_chunks(chroma: ChromaManager) -> List[Dict]:
    """Pull every chunk from ChromaDB and return as a flat list of dicts."""
    raw = chroma.collection.get()
    chunks = []
    for doc, meta in zip(raw["documents"], raw["metadatas"]):
        chunks.append({"text": doc, "metadata": meta})
    return chunks


def _filter_chunks(chunks: List[Dict], min_words: int = 40) -> List[Dict]:
    """Remove chunks that are too short to generate a meaningful question from."""
    filtered = [c for c in chunks if len(c["text"].split()) >= min_words]
    logger.info(
        "Filtered chunks: %d → %d (removed %d too-short chunks)",
        len(chunks), len(filtered), len(chunks) - len(filtered)
    )
    return filtered


def _sample_evenly(chunks: List[Dict], n: int,
                   content_types: Optional[List[str]] = None) -> List[Dict]:
    """
    Sample n chunks spread evenly across content types.

    If content_types is specified, only sample from those types.
    Falls back to random sampling if a type has fewer chunks than its quota.

    Args:
        chunks: All available chunks.
        n: Total number of chunks to sample.
        content_types: Optional filter list of content type strings.

    Returns:
        Sampled list of chunk dicts.
    """
    # Group by content type
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for chunk in chunks:
        ct = chunk["metadata"].get("content_type", "unknown")
        by_type[ct].append(chunk)

    # Apply content type filter
    if content_types:
        by_type = {k: v for k, v in by_type.items() if k in content_types}
        if not by_type:
            logger.warning("No chunks found for content types: %s", content_types)
            return []

    active_types = list(by_type.keys())
    logger.info("Content types available: %s", active_types)

    # Distribute quota evenly across types
    per_type = max(1, n // len(active_types))
    remainder = n - per_type * len(active_types)

    sampled = []
    for i, ct in enumerate(active_types):
        quota = per_type + (1 if i < remainder else 0)
        pool = by_type[ct]
        take = min(quota, len(pool))
        sampled.extend(random.sample(pool, take))
        logger.info("  Sampled %d/%d from '%s'", take, len(pool), ct)

    random.shuffle(sampled)
    return sampled[:n]


def _generate_qa_pair(chunk: Dict, generator: Generator) -> Optional[Tuple[str, str]]:
    """
    Ask Claude to generate a (question, answer) pair from a single chunk.

    Args:
        chunk: Chunk dict with 'text' and 'metadata'.
        generator: Configured Generator instance.

    Returns:
        (question, answer) tuple, or None if parsing fails.
    """
    meta = chunk["metadata"]
    prompt = _GENERATION_PROMPT.format(
        chunk_text=chunk["text"],
        title=meta.get("title", "Unknown"),
        content_type=meta.get("content_type", "unknown"),
        course_name=meta.get("course_name", "Unknown Course"),
    )

    try:
        response = generator.ask_llm(prompt)
        return _parse_qa(response)
    except Exception as e:
        logger.warning("LLM call failed for chunk '%s': %s",
                       meta.get("title", "?"), e)
        return None


def _parse_qa(text: str) -> Optional[Tuple[str, str]]:
    """
    Parse QUESTION/ANSWER from Claude's response.

    Args:
        text: Raw LLM response string.

    Returns:
        (question, answer) tuple or None if either field is missing.
    """
    q_match = re.search(r"QUESTION:\s*(.+?)(?=ANSWER:|$)", text, re.DOTALL | re.IGNORECASE)
    a_match = re.search(r"ANSWER:\s*(.+)", text, re.DOTALL | re.IGNORECASE)

    if not q_match or not a_match:
        logger.warning("Could not parse QA from response: %s", text[:120])
        return None

    question = q_match.group(1).strip()
    answer = a_match.group(1).strip()

    if not question or not answer:
        return None

    return question, answer


# ── Main runner ───────────────────────────────────────────────────────────────

def generate_questions(num_questions: int = 20,
                       output_file: str = "data/eval/test_questions_auto.json",
                       content_types: Optional[List[str]] = None,
                       seed: int = 42) -> List[Dict]:
    """
    Generate synthetic QA test cases from ChromaDB chunks.

    Args:
        num_questions: How many question-answer pairs to generate.
        output_file: Path to write the resulting JSON.
        content_types: Optional list to restrict to specific content types.
        seed: Random seed for reproducible sampling.

    Returns:
        List of test case dicts in the same format as test_questions.json.
    """
    random.seed(seed)

    chroma = ChromaManager()
    generator = Generator()

    kb_count = chroma.collection.count()
    if kb_count == 0:
        logger.error("Knowledge base is empty — run ingestion first.")
        sys.exit(1)

    logger.info("Knowledge base: %d chunks", kb_count)
    logger.info("Generating %d questions...\n", num_questions)

    # Fetch, filter, sample
    all_chunks = _fetch_all_chunks(chroma)
    filtered = _filter_chunks(all_chunks)
    sampled = _sample_evenly(filtered, num_questions, content_types)

    if not sampled:
        logger.error("No chunks available after filtering/sampling.")
        sys.exit(1)

    logger.info("\nGenerating QA pairs from %d sampled chunks...", len(sampled))

    test_cases = []
    failed = 0

    for i, chunk in enumerate(sampled, 1):
        meta = chunk["metadata"]
        title = meta.get("title", "Unknown")
        ct = meta.get("content_type", "unknown")

        print(f"[{i}/{len(sampled)}] {ct} — {title[:60]}")

        result = _generate_qa_pair(chunk, generator)
        if result is None:
            failed += 1
            print("  ✗ Failed to generate\n")
            continue

        question, answer = result
        print(f"  Q: {question[:80]}")
        print(f"  A: {answer[:80]}\n")

        test_cases.append({
            "question": question,
            "reference_answer": answer,
            "source_title": title,
            "source_content_type": ct,
            "source_course": meta.get("course_name", ""),
        })

    # Save
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(test_cases, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Generated : {len(test_cases)} questions")
    print(f"Failed    : {failed}")
    print(f"Saved to  : {out_path}")
    print(f"{'='*60}")
    print(f"\nRun evaluation:")
    print(f"  python scripts/evaluate.py --questions {out_path} "
          f"--output data/eval/results_auto.json --top-k 5")

    return test_cases


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Generate synthetic QA test cases from ingested Canvas chunks")
    parser.add_argument(
        "--num-questions", type=int, default=20,
        help="Number of question-answer pairs to generate (default: 20)")
    parser.add_argument(
        "--output", default="data/eval/test_questions_auto.json",
        help="Output JSON file path (default: data/eval/test_questions_auto.json)")
    parser.add_argument(
        "--content-type", nargs="+", default=None,
        metavar="TYPE",
        help="Restrict to specific content types e.g. --content-type syllabus page assignment")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible chunk sampling (default: 42)")
    args = parser.parse_args()

    generate_questions(
        num_questions=args.num_questions,
        output_file=args.output,
        content_types=args.content_type,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
