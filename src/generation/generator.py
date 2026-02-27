import requests
from typing import List, Dict, Tuple
from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL


class Generator:
    """Handles LLM answer generation."""

    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url

    def ask_llm(self, prompt: str) -> str:
        """Send prompt to Ollama and get response."""
        response = requests.post(
            f'{self.base_url}/api/generate',
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()['response']

    def generate_without_rag(self, question: str) -> str:
        """Generate answer without RAG."""
        prompt = f"""Answer the following question using your general knowledge. Be concise.
Question: {question}
Answer:"""
        return self.ask_llm(prompt)

    def generate_with_rag(self, question: str, retrieved_contexts: List[Dict]) -> Tuple[str, str, str]:
        """
        Generate answer with RAG using retrieved contexts.

        Returns: (answer, formatted_kb_context, llm_context)
        """
        # Get LLM general knowledge to supplement Canvas materials
        llm_context_prompt = f"""As an educational AI, provide brief background context about: {question}
Keep it concise (2-3 sentences) and focus on foundational concepts that would help a student understand this topic."""
        llm_context = self.ask_llm(llm_context_prompt)

        # Format retrieved contexts
        if retrieved_contexts:
            kb_context = "\n\n".join([
                f"[Source {i+1} - {ctx['metadata'].get('title', 'Unknown')} - Similarity: {ctx['similarity']:.2f}]\n{ctx['text']}"
                for i, ctx in enumerate(retrieved_contexts)
            ])

            prompt = f"""You are an educational AI assistant helping a student with their coursework.

IMPORTANT: I have searched the Canvas course materials and found relevant information below. You MUST use this Canvas information to answer the student's question. Do NOT rely on your general knowledge if the Canvas materials contain the answer.

=== CANVAS COURSE MATERIALS (Retrieved from course) ===
{kb_context}

=== BACKGROUND KNOWLEDGE (For context only) ===
{llm_context}

=== STUDENT'S QUESTION ===
{question}

INSTRUCTIONS:

1. FIRST, check if the Canvas materials above contain information about the student's question.
   - If YES: Use the Canvas materials to answer. This is course-specific information.
   - If NO: Then use your background knowledge.

2. QUESTION TYPE HANDLING:

   A) SCHEDULE/DATE QUESTIONS (e.g., "What will be taught on March 26?"):
      - Find the date in the table format: Week # | Day, Date | Topic | Reading | Notes
      - Extract ONLY the Topic column (3rd position)
      - Format: "The topic(s) that will be taught on [date] is/are [topic list]."
      - Ignore readings, assignments, quiz dates

   B) PEOPLE/INSTRUCTOR QUESTIONS (e.g., "Who is [name]?"):
      - Check Canvas materials for information about this person
      - Report what the course materials say (instructor, TA, guest speaker, etc.)
      - Include relevant details like title, role, contact info if available

   C) CONTENT/CONCEPT QUESTIONS (e.g., "What is text preprocessing?"):
      - Explain the concept clearly
      - Use Canvas materials as primary source
      - Supplement with background knowledge for better understanding

   D) POLICY/ASSIGNMENT QUESTIONS:
      - Refer to the specific course policies, deadlines, and requirements
      - Be precise about dates, requirements, and grading

3. GENERAL RULES:
   - Answer directly and conversationally
   - Do NOT say "I couldn't find information" if the Canvas materials contain it
   - Do NOT show your reasoning process
   - Do NOT quote raw table formatting (| separators)
   - Be helpful and student-friendly

Answer:"""
        else:
            kb_context = "No relevant context found in knowledge base"
            prompt = f"""You are an educational AI tutor. Answer this question using your general knowledge.

Question: {question}

Answer:"""

        answer = self.ask_llm(prompt)
        return answer, kb_context, llm_context
