#!/usr/bin/env python3
"""
RAG Evaluation Script

Runs the SimpleRAG pipeline on a set of test questions and computes five
quality metrics:

  Retrieval:
    context_relevance  — avg similarity of retrieved chunks to the question
    context_recall     — fraction of reference-answer claims covered by context
    context_precision  — are relevant chunks ranked before irrelevant ones?

  Generation:
    faithfulness       — fraction of answer claims grounded in the context
    answer_relevance   — does the answer actually address the question?

context_recall and context_precision require a "reference_answer" in the
test-questions JSON. The other three metrics work without one.

Usage:
    python scripts/evaluate.py --questions data/eval/test_questions.json
    python scripts/evaluate.py --questions data/eval/test_questions.json \\
                               --output data/eval/results.json \\
                               --top-k 5
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.retriever import Retriever
from src.generation.generator import Generator
from src.evaluation.metrics import RAGEvaluator
from src.config import LOG_DIR

# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging() -> None:
    log_file = LOG_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ── Core runner ───────────────────────────────────────────────────────────────

def run_evaluation(questions_file: str,
                   output_file: Optional[str] = None,
                   top_k: int = 3) -> Dict:
    """
    Run full evaluation on a JSON file of test questions.

    Args:
        questions_file: Path to JSON file (list of objects with 'question'
                        and optional 'reference_answer').
        output_file: Optional path to write the full results JSON.
        top_k: Number of context chunks to retrieve per question.

    Returns:
        Dict with 'summary' (aggregate scores) and 'results' (per-question).
    """
    # Load questions
    questions_path = Path(questions_file)
    if not questions_path.exists():
        logger.error("Questions file not found: %s", questions_file)
        sys.exit(1)

    with open(questions_path) as f:
        test_cases: List[Dict] = json.load(f)

    if not test_cases:
        logger.error("No test cases found in %s", questions_file)
        sys.exit(1)

    logger.info("Loaded %d test questions from %s", len(test_cases), questions_file)

    # Initialise pipeline components
    retriever = Retriever()
    generator = Generator()
    evaluator = RAGEvaluator()

    kb_stats = retriever.get_stats()
    if kb_stats["count"] == 0:
        logger.error("Knowledge base is empty — run ingestion first.")
        sys.exit(1)

    logger.info("Knowledge base: %d documents", kb_stats["count"])
    logger.info("Evaluating with top_k=%d\n", top_k)

    # Per-question evaluation
    all_results: List[Dict] = []
    collector: Dict[str, List[float]] = {
        "context_relevance": [],
        "context_recall": [],
        "context_precision": [],
        "faithfulness": [],
        "answer_relevance": [],
    }

    for i, test_case in enumerate(test_cases, 1):
        question: str = test_case["question"]
        reference: Optional[str] = test_case.get("reference_answer")

        print(f"\n{'='*65}")
        print(f"[{i}/{len(test_cases)}] {question}")
        if reference:
            print(f"  Reference: {reference[:80]}{'...' if len(reference) > 80 else ''}")

        # Retrieve
        retrieved_contexts = retriever.retrieve(question, top_k=top_k)
        logger.info("Retrieved %d contexts", len(retrieved_contexts))

        # Generate RAG answer
        if retrieved_contexts:
            answer_with_rag, _, _ = generator.generate_with_rag(
                question, retrieved_contexts)
        else:
            answer_with_rag = "No relevant context found in the knowledge base."
            logger.warning("No contexts retrieved for: %s", question[:60])

        # Compute all metrics
        result = evaluator.evaluate(
            question=question,
            retrieved_contexts=retrieved_contexts,
            answer_with_rag=answer_with_rag,
            reference_answer=reference,
        )
        result["generated_answer"] = answer_with_rag
        all_results.append(result)

        # Collect for aggregate summary
        for metric in collector:
            score = result.get(metric, {}).get("score")
            if score is not None:
                collector[metric].append(score)

        # Print per-question scores
        _print_scores(result)

    # Aggregate summary
    summary = _build_summary(collector, len(test_cases))
    _print_summary(summary)

    # Persist results
    output = {
        "evaluated_at": datetime.now().isoformat(),
        "questions_file": str(questions_path),
        "knowledge_base_documents": kb_stats["count"],
        "top_k": top_k,
        "num_questions": len(test_cases),
        "summary": summary,
        "results": all_results,
    }

    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("Results saved to %s", output_file)

    return output


# ── Display helpers ───────────────────────────────────────────────────────────

def _print_scores(result: Dict) -> None:
    """Print the five metric scores for a single question."""
    metrics = [
        ("Context Relevance ", result["context_relevance"]["score"]),
        ("Context Recall    ", result["context_recall"]["score"]),
        ("Context Precision ", result["context_precision"]["score"]),
        ("Faithfulness      ", result["faithfulness"]["score"]),
        ("Answer Relevance  ", result["answer_relevance"]["score"]),
    ]
    print()
    for label, score in metrics:
        value = f"{score:.4f}" if score is not None else "N/A (no reference answer)"
        print(f"  {label}: {value}")


def _build_summary(collector: Dict[str, List[float]], total: int) -> Dict:
    """Build aggregate statistics from collected per-question scores."""
    summary = {}
    for metric, scores in collector.items():
        if scores:
            summary[metric] = {
                "mean": round(sum(scores) / len(scores), 4),
                "min":  round(min(scores), 4),
                "max":  round(max(scores), 4),
                "n":    len(scores),
                "coverage": f"{len(scores)}/{total} questions",
            }
        else:
            summary[metric] = {
                "mean": None,
                "note": "No questions had a reference answer",
                "coverage": f"0/{total} questions",
            }
    return summary


def _print_summary(summary: Dict) -> None:
    """Print aggregate evaluation results table."""
    print(f"\n{'='*65}")
    print("EVALUATION SUMMARY")
    print(f"{'='*65}")
    print(f"{'Metric':<25} {'Mean':>8}  {'Min':>8}  {'Max':>8}  Coverage")
    print("-" * 65)
    for metric, stats in summary.items():
        if stats.get("mean") is not None:
            row = (f"{metric:<25} {stats['mean']:>8.4f}  "
                   f"{stats['min']:>8.4f}  {stats['max']:>8.4f}  "
                   f"{stats['coverage']}")
        else:
            row = f"{metric:<25} {'N/A':>8}  {'':>8}  {'':>8}  {stats['coverage']}"
        print(row)
    print(f"{'='*65}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Main entry point."""
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Evaluate the SimpleRAG system on a set of test questions")
    parser.add_argument(
        "--questions", required=True,
        help="Path to JSON file with test questions (see data/eval/test_questions.json)")
    parser.add_argument(
        "--output", default=None,
        help="Path to save full results JSON (optional, e.g. data/eval/results.json)")
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of context chunks to retrieve per question (default: 3)")
    args = parser.parse_args()

    run_evaluation(
        questions_file=args.questions,
        output_file=args.output,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
