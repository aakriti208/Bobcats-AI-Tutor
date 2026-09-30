# 5 RAG Metrics

## Retrieval: Did we retrieve the right information?

### 1. Context Relevance: 0.57

- **What it measures:** How similar the retrieved chunks are to the question.
- **Interpretation:** Moderate score. The chunks were related to the questions but were sometimes broad or noisy.

### 2. Context Recall: 0.88

- **What it measures:** Whether the retrieved context contained the facts needed to answer the question.
- **Interpretation:** Strong result. Approximately 88% of the required facts were retrieved.

### 3. Context Precision: 0.79

- **What it measures:** Whether the most relevant chunks appeared near the top of the retrieval results.
- **Interpretation:** Good result, but some irrelevant chunks appeared among the top results.

---

## Generation: Was the answer good?

### 4. Faithfulness: 0.85

- **What it measures:** Whether the generated answer was supported by the retrieved context.
- **Interpretation:** Strong result. Approximately 85% of claims were grounded in the retrieved context, with limited hallucination.

### 5. Answer Relevance: 0.69

- **What it measures:** Whether the answer actually addressed the user's question.
- **Interpretation:** Weakest metric. Some answers were accurate but included unnecessary or tangential information.

---

## Overall

The system was generally effective at finding the necessary information and grounding its answers. The main weakness was **answer focus and relevance**, reflected by the lower Answer Relevance score of **0.69**.

## Improvement

- Improve chunking: Use smaller, more focused chunks to improve context relevance.
- Improve retrieval: Increase top-k, use hybrid search, or add a reranker to improve recall and precision.
- Reduce hallucination: Strengthen the prompt to answer only from retrieved context and say "I don't know" when evidence is missing.
- Improve answer relevance: Make the prompt more focused so answers directly address the question without unnecessary information.
- Add reranking: Reorder retrieved chunks based on relevance before sending them to the LLM.
