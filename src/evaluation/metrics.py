"""
RAG Evaluation Metrics

Five metrics split across two categories:

Retrieval:
  - Context Relevance  : avg similarity of retrieved chunks to the question
  - Context Recall     : fraction of reference-answer claims covered by context  (needs reference)
  - Context Precision  : are relevant chunks ranked before irrelevant ones?       (needs reference)

Generation:
  - Faithfulness       : fraction of answer claims grounded in retrieved context
  - Answer Relevance   : does the answer address the question?

LLM-as-judge metrics use the existing Ollama generator so no extra dependencies
are needed. Embedding-based metrics use the existing Embedder.
"""

import logging
import re
import numpy as np
from typing import Dict, List, Optional

from src.embedding.embedder import Embedder
from src.generation.generator import Generator

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Computes retrieval and generation quality metrics for the RAG pipeline."""

    def __init__(self):
        """Initialize evaluator, reusing project's existing embedder and generator."""
        self.embedder = Embedder()
        self.generator = Generator()

    # ── RETRIEVAL METRICS ─────────────────────────────────────────────────────

    def context_relevance(self, retrieved_contexts: List[Dict]) -> Dict:
        """
        Measures whether retrieved chunks are relevant to the question.

        Uses the cosine similarity scores already computed by the retriever —
        no extra LLM or embedding call required.

        Args:
            retrieved_contexts: List of context dicts from Retriever.retrieve(),
                                 each containing a 'similarity' key.

        Returns:
            Dict with 'score' (mean similarity, 0-1), per-chunk similarities,
            and chunk count.
        """
        if not retrieved_contexts:
            return {"score": 0.0, "num_chunks": 0, "similarities": [],
                    "note": "No contexts retrieved"}

        similarities = [float(ctx["similarity"]) for ctx in retrieved_contexts]
        score = float(np.mean(similarities))

        return {
            "score": round(score, 4),
            "num_chunks": len(similarities),
            "similarities": [round(s, 4) for s in similarities],
        }

    def context_recall(self, question: str, retrieved_contexts: List[Dict],
                       reference_answer: str) -> Dict:
        """
        Measures what fraction of reference-answer claims are covered by the context.

        Method: single LLM call. The LLM breaks the reference answer into individual
        factual claims, checks each against the retrieved context, and reports the
        fraction that are supported.

        Args:
            question: The original student question.
            retrieved_contexts: Retrieved chunks from Retriever.retrieve().
            reference_answer: Ground-truth answer to compare against.

        Returns:
            Dict with 'score' (0-1) and the raw LLM response for inspection.
            Returns score=None if inputs are missing.
        """
        if not retrieved_contexts or not reference_answer:
            return {"score": None,
                    "note": "Requires both a reference answer and retrieved contexts"}

        context_text = _format_contexts(retrieved_contexts)

        prompt = f"""You are evaluating a retrieval system for an educational AI tutor.

QUESTION: {question}

REFERENCE ANSWER (ground truth):
{reference_answer}

RETRIEVED CONTEXT:
{context_text}

TASK:
Break the reference answer into individual factual claims (one per line).
For each claim, decide if it is directly supported or inferable from the retrieved context.

Use this exact format — nothing else:
CLAIMS:
1. [claim text] → SUPPORTED
2. [claim text] → NOT SUPPORTED
3. [claim text] → SUPPORTED
RECALL_SCORE: [decimal between 0.0 and 1.0]"""

        response = self.generator.ask_llm(prompt)
        score = _extract_labeled_score(response, "RECALL_SCORE")

        return {"score": score, "llm_response": response}

    def context_precision(self, question: str, retrieved_contexts: List[Dict],
                          reference_answer: str) -> Dict:
        """
        Measures whether relevant chunks are ranked higher than irrelevant ones.

        Method: single batched LLM call that judges all chunks at once, then
        average precision is computed from the resulting relevance list
        (penalises relevant chunks appearing late in the ranking).

        Args:
            question: The original student question.
            retrieved_contexts: Retrieved chunks in ranked order.
            reference_answer: Ground-truth answer used to judge relevance.

        Returns:
            Dict with 'score' (0-1), per-chunk relevance judgements, and
            the average-precision breakdown.
        """
        if not retrieved_contexts or not reference_answer:
            return {"score": None,
                    "note": "Requires both a reference answer and retrieved contexts"}

        chunks_text = "\n\n".join(
            f"CHUNK {i+1}:\n{ctx['text']}"
            for i, ctx in enumerate(retrieved_contexts)
        )

        prompt = f"""You are evaluating a retrieval system for an educational AI tutor.

QUESTION: {question}

REFERENCE ANSWER (ground truth):
{reference_answer}

RETRIEVED CHUNKS (in ranked order):
{chunks_text}

TASK:
For each chunk, answer YES if it contains information useful for answering the question
(given the reference answer), or NO if it does not.

Use this exact format — nothing else:
CHUNK 1: YES
CHUNK 2: NO
CHUNK 3: YES
...
PRECISION_SCORE: [average precision as a decimal between 0.0 and 1.0]"""

        response = self.generator.ask_llm(prompt)
        relevances = _parse_chunk_relevances(response, len(retrieved_contexts))
        score = _average_precision(relevances) if relevances else None

        return {
            "score": round(score, 4) if score is not None else None,
            "chunk_relevances": relevances,
            "llm_response": response,
        }

    # ── GENERATION METRICS ────────────────────────────────────────────────────

    def faithfulness(self, question: str, retrieved_contexts: List[Dict],
                     generated_answer: str) -> Dict:
        """
        Measures whether the generated answer's claims are grounded in the context.

        Method: single LLM call. The LLM extracts factual claims from the answer
        and checks each against the retrieved context. Does NOT require a reference
        answer — only the answer and its source context.

        Args:
            question: The original student question.
            retrieved_contexts: The context chunks the answer was generated from.
            generated_answer: The RAG-generated answer to evaluate.

        Returns:
            Dict with 'score' (0-1) and the raw LLM response for inspection.
        """
        if not retrieved_contexts or not generated_answer:
            return {"score": 0.0, "note": "No context or answer to evaluate"}

        context_text = _format_contexts(retrieved_contexts)

        prompt = f"""You are evaluating whether an AI-generated answer is faithful to its source material.

QUESTION: {question}

SOURCE CONTEXT (from Canvas course materials):
{context_text}

GENERATED ANSWER:
{generated_answer}

TASK:
1. Extract each specific factual claim made in the generated answer.
   Skip generic hedges like "I hope this helps" or "based on the context".
2. For each claim, decide if it is directly supported by the source context above.

Use this exact format — nothing else:
CLAIMS:
1. [claim text] → SUPPORTED
2. [claim text] → NOT SUPPORTED
3. [claim text] → SUPPORTED
FAITHFULNESS_SCORE: [decimal between 0.0 and 1.0]"""

        response = self.generator.ask_llm(prompt)
        score = _extract_labeled_score(response, "FAITHFULNESS_SCORE")

        return {"score": score, "llm_response": response}

    def answer_relevance(self, question: str, generated_answer: str) -> Dict:
        """
        Measures whether the generated answer actually addresses the question.

        Method: ask the LLM to generate 3 hypothetical questions that the answer
        would be a good response to, then compute the average cosine similarity
        between those reverse-questions and the original question. High similarity
        means the answer stayed on topic.

        Args:
            question: The original student question.
            generated_answer: The answer to evaluate.

        Returns:
            Dict with 'score' (0-1), the generated reverse questions,
            and per-question similarity scores.
        """
        if not generated_answer:
            return {"score": 0.0, "note": "No answer to evaluate"}

        prompt = f"""Given the following answer, write the 3 most likely questions a student
could have asked to receive this answer.

ANSWER:
{generated_answer}

Output exactly 3 questions, numbered 1-3, one per line. No explanation."""

        response = self.generator.ask_llm(prompt)
        reverse_questions = _parse_numbered_list(response)

        if not reverse_questions:
            logger.warning("Could not parse reverse questions from LLM response")
            return {"score": None, "note": "LLM did not return parseable questions",
                    "llm_response": response}

        orig_vec = self.embedder.embed_text(question)
        similarities = []
        for rq in reverse_questions:
            rq_vec = self.embedder.embed_text(rq)
            cos_sim = float(
                np.dot(orig_vec, rq_vec) /
                (np.linalg.norm(orig_vec) * np.linalg.norm(rq_vec) + 1e-10)
            )
            similarities.append(round(cos_sim, 4))

        score = float(np.mean(similarities))

        return {
            "score": round(score, 4),
            "reverse_questions": reverse_questions,
            "similarities": similarities,
        }

    # ── COMBINED RUNNER ───────────────────────────────────────────────────────

    def evaluate(self, question: str, retrieved_contexts: List[Dict],
                 answer_with_rag: str,
                 reference_answer: Optional[str] = None) -> Dict:
        """
        Run all five metrics for a single question.

        Context recall and context precision are skipped (score=None) when no
        reference answer is provided.

        Args:
            question: The student's question.
            retrieved_contexts: Contexts returned by Retriever.retrieve().
            answer_with_rag: The RAG-generated answer.
            reference_answer: Optional ground-truth answer. Enables recall + precision.

        Returns:
            Dict with keys: question, has_reference, num_contexts_retrieved,
            context_relevance, context_recall, context_precision,
            faithfulness, answer_relevance.
        """
        logger.info("Evaluating: %s", question[:80])

        result: Dict = {
            "question": question,
            "has_reference": reference_answer is not None,
            "num_contexts_retrieved": len(retrieved_contexts),
        }

        logger.info("  → context_relevance")
        result["context_relevance"] = self.context_relevance(retrieved_contexts)

        if reference_answer:
            logger.info("  → context_recall")
            result["context_recall"] = self.context_recall(
                question, retrieved_contexts, reference_answer)
            logger.info("  → context_precision")
            result["context_precision"] = self.context_precision(
                question, retrieved_contexts, reference_answer)
        else:
            result["context_recall"] = {"score": None,
                                        "note": "No reference answer provided"}
            result["context_precision"] = {"score": None,
                                           "note": "No reference answer provided"}

        logger.info("  → faithfulness")
        result["faithfulness"] = self.faithfulness(
            question, retrieved_contexts, answer_with_rag)

        logger.info("  → answer_relevance")
        result["answer_relevance"] = self.answer_relevance(question, answer_with_rag)

        return result


# ── MODULE-LEVEL HELPERS ──────────────────────────────────────────────────────

def _format_contexts(contexts: List[Dict]) -> str:
    """Format retrieved contexts into a numbered block for LLM prompts."""
    return "\n\n".join(
        f"[Source {i+1} — {ctx['metadata'].get('title', 'Unknown')} "
        f"(similarity {ctx['similarity']:.2f})]\n{ctx['text']}"
        for i, ctx in enumerate(contexts)
    )


def _extract_labeled_score(text: str, label: str) -> Optional[float]:
    """
    Extract a score from LLM output like 'FAITHFULNESS_SCORE: 0.75'.

    Args:
        text: Raw LLM response string.
        label: The label prefix to look for (e.g. 'FAITHFULNESS_SCORE').

    Returns:
        Float clamped to [0, 1], or None if not found.
    """
    pattern = rf"{re.escape(label)}:\s*([0-9]*\.?[0-9]+)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return round(min(max(float(match.group(1)), 0.0), 1.0), 4)
        except ValueError:
            pass
    logger.warning("Could not extract %s from LLM response", label)
    return None


def _parse_chunk_relevances(text: str, num_chunks: int) -> List[bool]:
    """
    Parse per-chunk YES/NO judgements from LLM output.

    Expects lines like 'CHUNK 1: YES', 'CHUNK 2: NO', etc.

    Args:
        text: Raw LLM response.
        num_chunks: Expected number of chunks.

    Returns:
        List of booleans, one per chunk. Missing entries default to False.
    """
    relevances = [False] * num_chunks
    for line in text.splitlines():
        match = re.match(r"CHUNK\s+(\d+):\s*(YES|NO)", line.strip(), re.IGNORECASE)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < num_chunks:
                relevances[idx] = match.group(2).upper() == "YES"
    return relevances


def _average_precision(relevances: List[bool]) -> float:
    """
    Compute Average Precision from an ordered list of relevance labels.

    AP = (1/R) * sum_{k} precision@k * rel_k
    where R is the total number of relevant items.

    Args:
        relevances: Boolean list in retrieval rank order (index 0 = rank 1).

    Returns:
        Average precision score in [0, 1]. Returns 0.0 if no relevant items.
    """
    if not any(relevances):
        return 0.0

    num_relevant = sum(relevances)
    running_relevant = 0
    precision_sum = 0.0

    for rank, is_relevant in enumerate(relevances, start=1):
        if is_relevant:
            running_relevant += 1
            precision_sum += running_relevant / rank

    return precision_sum / num_relevant


def _parse_numbered_list(text: str) -> List[str]:
    """
    Parse a numbered list from LLM output (handles '1.', '1)', '1 ' prefixes).

    Args:
        text: Raw LLM response.

    Returns:
        List of stripped strings for each numbered item found.
    """
    items = []
    for line in text.strip().splitlines():
        match = re.match(r"^\d+[\.\)]\s+(.+)", line.strip())
        if match:
            items.append(match.group(1).strip())
    return items
