import re
from typing import List, Dict, Optional
from src.vectorstore.chroma_manager import ChromaManager
from src.config import TOP_K_RESULTS, SIMILARITY_THRESHOLD


class Retriever:
    """Handles retrieval operations using ChromaDB."""

    def __init__(self):
        self.chroma_manager = ChromaManager()

    def retrieve(self, question: str, top_k: int = TOP_K_RESULTS,
                 course_filter: Optional[str] = None) -> List[Dict]:
        """
        Retrieve relevant context for a question.

        Returns list of dicts with 'text', 'metadata', and 'distance' keys.
        """
        # Extract potential dates from question for keyword matching
        date_keywords = self._extract_date_keywords(question)

        # Build metadata filter if course specified
        where_filter = None
        if course_filter:
            where_filter = {"course_id": course_filter}

        # Query ChromaDB with higher top_k if date detected
        query_top_k = top_k * 3 if date_keywords else top_k

        results = self.chroma_manager.query(
            query_text=question,
            n_results=query_top_k,
            where=where_filter
        )

        # Format and filter results
        retrieved_contexts = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i]
                similarity = 1 - distance
                text = results['documents'][0][i]

                # Boost score if date keywords match
                boost = 0.0
                if date_keywords:
                    for keyword in date_keywords:
                        if keyword.lower() in text.lower():
                            boost = 0.3  # Significant boost for keyword match
                            break

                adjusted_similarity = min(1.0, similarity + boost)

                # Use lower threshold for date queries with keyword match
                threshold = SIMILARITY_THRESHOLD - 0.1 if date_keywords else SIMILARITY_THRESHOLD

                if adjusted_similarity >= threshold:
                    context = {
                        'text': text,
                        'metadata': results['metadatas'][0][i],
                        'similarity': adjusted_similarity,
                        'distance': distance
                    }
                    retrieved_contexts.append(context)

        # Sort by adjusted similarity and limit to top_k
        retrieved_contexts.sort(key=lambda x: x['similarity'], reverse=True)
        return retrieved_contexts[:top_k]

    def _extract_date_keywords(self, question: str) -> List[str]:
        """Extract date-related keywords from question."""
        keywords = []

        # Match patterns like "Apr 23", "April 23", "March 5", etc.
        month_day_pattern = r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b'
        matches = re.findall(month_day_pattern, question, re.IGNORECASE)

        for match in matches:
            keywords.append(match)
            # Also add abbreviated form
            month_map = {
                'January': 'Jan', 'February': 'Feb', 'March': 'Mar',
                'April': 'Apr', 'May': 'May', 'June': 'Jun',
                'July': 'Jul', 'August': 'Aug', 'September': 'Sep',
                'October': 'Oct', 'November': 'Nov', 'December': 'Dec'
            }
            for full, abbr in month_map.items():
                if full.lower() in question.lower():
                    keywords.append(abbr)

        return list(set(keywords))

    def get_stats(self) -> Dict:
        """Get collection statistics."""
        return self.chroma_manager.get_collection_stats()
