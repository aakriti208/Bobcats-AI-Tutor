#!/usr/bin/env python3
"""
Evaluation script for the Canvas RAG pipeline using RAGAS.

Steps:
  1. Sample chunks from ChromaDB
  2. Generate synthetic Q&A pairs via Groq
  3. Run each question through the RAG pipeline (Retriever + Groq)
  4. Score with RAGAS: Faithfulness, Answer Relevancy, Context Precision, Context Recall

Usage:
  python scripts/eval.py
  python scripts/eval.py --samples 30
  python scripts/eval.py --samples 10 --output data/eval/my_run.json
"""

import sys
import json
import random
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import GROQ_API_KEY, GROQ_MODEL, TOP_K_RESULTS
from src.retrieval.retriever import Retriever
from src.vectorstore.chroma_manager import ChromaManager

from groq import Groq as GroqClient
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------------------------------------------------------------------------
# Step 1: Sample chunks from ChromaDB
# ---------------------------------------------------------------------------

def sample_chunks(n: int, manager: ChromaManager = None) -> list[dict]:
    """Pull n random chunks from ChromaDB."""
    if manager is None:
        manager = ChromaManager()
    total = manager.collection.count()

    if total == 0:
        raise ValueError("ChromaDB collection is empty — run ingestion first.")

    n = min(n, total)
    all_results = manager.collection.get(limit=total)
    indices = random.sample(range(total), n)

    return [
        {"text": all_results["documents"][i], "metadata": all_results["metadatas"][i]}
        for i in indices
    ]


# ---------------------------------------------------------------------------
# Step 2: Synthetic Q&A generation
# ---------------------------------------------------------------------------

def generate_qa_pair(client: GroqClient, chunk_text: str) -> tuple[str, str] | None:
    """Generate a question + reference answer from a single chunk."""
    prompt = f"""Given the following excerpt from a university course material, generate ONE specific question that can be answered using ONLY this text, and provide the answer.

TEXT:
{chunk_text[:1500]}

Respond in this exact JSON format:
{{"question": "...", "answer": "..."}}

Rules:
- The question must be fully answerable from the text above
- The answer must be factual, based only on the text, and 1-3 sentences
- Avoid vague questions like "What is this about?" or "What does this discuss?"
"""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    try:
        result = json.loads(response.choices[0].message.content)
        q = result.get("question", "").strip()
        a = result.get("answer", "").strip()
        if q and a:
            return q, a
    except (json.JSONDecodeError, AttributeError):
        pass
    return None


# ---------------------------------------------------------------------------
# Step 3: RAG pipeline (retrieve + generate)
# ---------------------------------------------------------------------------

def run_rag(retriever: Retriever, client: GroqClient, question: str) -> tuple[str, list[str]]:
    """Retrieve contexts and generate an answer via Groq."""
    retrieved = retriever.retrieve(question)
    contexts = [r["text"] for r in retrieved]

    if not contexts:
        return "No relevant information found in the course materials.", []

    context_str = "\n\n".join(f"[Source {i+1}]\n{c}" for i, c in enumerate(contexts))
    prompt = f"""Answer the following question based solely on the provided course material.

CONTEXT:
{context_str}

QUESTION: {question}

Answer concisely and accurately using only the context above:"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    answer = response.choices[0].message.content.strip()
    return answer, contexts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline with RAGAS")
    parser.add_argument("--samples", type=int, default=20,
                        help="Number of test samples to generate (default: 20)")
    parser.add_argument("--output", type=str, default="data/eval/results.json",
                        help="Path to save results JSON")
    args = parser.parse_args()

    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY not set in .env")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Canvas RAG Pipeline Evaluation")
    print(f"{'='*60}")

    # --- Step 1 ---
    print(f"\n[1/4] Sampling {args.samples} chunks from ChromaDB...")
    chroma_manager = ChromaManager()
    chunks = sample_chunks(args.samples, manager=chroma_manager)
    print(f"      Sampled {len(chunks)} chunks")

    # --- Step 2 ---
    print(f"\n[2/4] Generating synthetic Q&A pairs via Groq ({GROQ_MODEL})...")
    groq_client = GroqClient(api_key=GROQ_API_KEY)

    qa_pairs = []
    for i, chunk in enumerate(chunks):
        print(f"      {i+1}/{len(chunks)} generating...", end="\r")
        result = generate_qa_pair(groq_client, chunk["text"])
        if result:
            question, reference_answer = result
            qa_pairs.append({
                "question": question,
                "reference_answer": reference_answer,
                "source_chunk": chunk["text"],
            })

    print(f"\n      Generated {len(qa_pairs)} valid Q&A pairs")

    if not qa_pairs:
        print("No Q&A pairs generated — exiting.")
        sys.exit(1)

    # --- Step 3 ---
    print(f"\n[3/4] Running RAG pipeline on {len(qa_pairs)} questions...")
    retriever = Retriever()
    retriever.chroma_manager = chroma_manager

    eval_rows = []
    for i, qa in enumerate(qa_pairs):
        print(f"      {i+1}/{len(qa_pairs)} querying...", end="\r")
        answer, contexts = run_rag(retriever, groq_client, qa["question"])
        eval_rows.append({
            "question": qa["question"],
            "answer": answer,
            "contexts": contexts if contexts else [""],
            "ground_truth": qa["reference_answer"],
        })

    print(f"\n      Completed {len(eval_rows)} RAG queries")

    # --- Step 4 ---
    print(f"\n[4/4] Running RAGAS evaluation (this may take a few minutes)...")

    ragas_llm = LangchainLLMWrapper(
        ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    dataset = Dataset.from_list(eval_rows)

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    scores_df = results.to_pandas()
    mean_scores = {
        "faithfulness": float(scores_df["faithfulness"].mean()),
        "answer_relevancy": float(scores_df["answer_relevancy"].mean()),
        "context_precision": float(scores_df["context_precision"].mean()),
        "context_recall": float(scores_df["context_recall"].mean()),
    }

    print(f"\n{'='*60}")
    print("RAGAS Evaluation Results")
    print(f"{'='*60}")
    print(f"  Faithfulness:      {mean_scores['faithfulness']:.3f}  (is answer grounded in context?)")
    print(f"  Answer Relevancy:  {mean_scores['answer_relevancy']:.3f}  (is answer relevant to question?)")
    print(f"  Context Precision: {mean_scores['context_precision']:.3f}  (are retrieved docs relevant?)")
    print(f"  Context Recall:    {mean_scores['context_recall']:.3f}  (do contexts cover the answer?)")
    print(f"{'='*60}")

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "samples": len(eval_rows),
            "llm_provider": "groq",
            "model": GROQ_MODEL,
            "top_k": TOP_K_RESULTS,
        },
        "scores": mean_scores,
        "samples": eval_rows,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed results saved to: {output_path}\n")


if __name__ == "__main__":
    main()
