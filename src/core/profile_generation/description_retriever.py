import logging
from typing import List, Dict, Tuple

from src.core.knowledge_graph.neo4j_manager import neo4j_manager

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

logger = logging.getLogger(__name__)

class DescriptionContextRetriever:
    """Retrieve context for document description generation with token limit management."""

    def __init__(self, max_context_tokens: int = 900000, model_name: str = "gpt-4"):
        self.neo4j_manager = neo4j_manager
        self.max_context_tokens = max_context_tokens

        if TIKTOKEN_AVAILABLE:
            try:
                self.encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                self.encoding = tiktoken.get_encoding("cl100k_base")
        else:
            self.encoding = None

        self._ensure_connection()

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            return len(text) // 4

    def _ensure_connection(self):
        if not self.neo4j_manager.is_connected():
            self.neo4j_manager.connect()

    def get_all_document_chunks(self, document_id: str) -> List[Dict]:
        query = """
        MATCH (d:Document {id: $document_id})-[*1..10]-(c:Chunk)
        RETURN DISTINCT c.id as chunk_id,
               c.chunk_number as chunk_number,
               c.text as text,
               c.summary as summary
        ORDER BY c.chunk_number ASC
        """

        try:
            results = self.neo4j_manager.execute_query(query=query, parameters={"document_id": document_id})
            chunks = []

            for record in results:
                text_content = record.get("text", "")
                summary_content = record.get("summary", "")

                chunk_data = {
                    "chunk_id": record.get("chunk_id", ""),
                    "chunk_number": record.get("chunk_number", 0),
                    "text": text_content,
                    "summary": summary_content,
                    "text_tokens": self.count_tokens(text_content),
                    "summary_tokens": self.count_tokens(summary_content)
                }
                chunks.append(chunk_data)

            return chunks
        except Exception as e:
            logger.error(f"Error retrieving chunks for document {document_id}: {e}")
            return []

    def _select_optimal_chunks(self, chunks: List[Dict]) -> Tuple[List[Dict], bool]:
        if not chunks:
            return [], False

        selected_chunks = []
        for chunk in chunks:
            text_content = chunk.get("text", "")
            text_tokens = chunk.get("text_tokens", 0)

            chunk_copy = chunk.copy()
            chunk_copy["selected_content"] = text_content
            chunk_copy["content_type"] = "full_text"
            chunk_copy["selected_tokens"] = text_tokens
            selected_chunks.append(chunk_copy)

        total_tokens = sum(chunk.get("selected_tokens", 0) for chunk in selected_chunks)
        used_summaries = False

        while total_tokens > self.max_context_tokens:
            largest_index = -1
            largest_tokens = 0

            for i, chunk in enumerate(selected_chunks):
                if (chunk.get("content_type") == "full_text" and
                    chunk.get("selected_tokens", 0) > largest_tokens and
                    chunk.get("summary", "").strip()):
                    largest_tokens = chunk.get("selected_tokens", 0)
                    largest_index = i

            if largest_index == -1:
                break

            original_chunk = chunks[largest_index]
            summary_content = original_chunk.get("summary", "")
            summary_tokens = original_chunk.get("summary_tokens", 0)

            selected_chunks[largest_index]["selected_content"] = summary_content
            selected_chunks[largest_index]["content_type"] = "summary"
            selected_chunks[largest_index]["selected_tokens"] = summary_tokens

            total_tokens = total_tokens - largest_tokens + summary_tokens
            used_summaries = True

        return selected_chunks, used_summaries

    def retrieve_document_context(self, document_id: str) -> Dict:
        all_chunks = self.get_all_document_chunks(document_id)

        if not all_chunks:
            return {
                "context": "",
                "total_chunks": 0,
                "selected_chunks": 0,
                "used_summaries": False,
                "total_tokens": 0
            }

        selected_chunks, used_summaries = self._select_optimal_chunks(all_chunks)
        selected_chunks.sort(key=lambda x: x.get("chunk_number", 0))

        context_parts = []
        for chunk in selected_chunks:
            content = chunk.get("selected_content", "")
            if content.strip():
                chunk_number = chunk.get("chunk_number", 0)
                content_type = chunk.get("content_type", "unknown")
                header = f"--- Chunk {chunk_number} ({content_type}) ---"
                context_parts.append(f"{header}\n{content}")

        final_context = "\n\n".join(context_parts)

        return {
            "context": final_context,
            "total_chunks": len(all_chunks),
            "selected_chunks": len(selected_chunks),
            "used_summaries": used_summaries,
            "total_tokens": sum(chunk.get("selected_tokens", 0) for chunk in selected_chunks)
        }


if __name__ == "__main__":
    # Test the description retriever
    retriever = DescriptionContextRetriever(max_context_tokens=10000)  # Smaller limit for testing

    # Test document ID - replace with actual document ID
    test_document_id = "aeece878-9dcb-4dc5-8e31-a6e4277ac4ec"

    print("Testing DescriptionContextRetriever...")
    result = retriever.retrieve_document_context(test_document_id)

    print(f"\nResults:")
    print(f"Total chunks: {result['total_chunks']}")
    print(f"Selected chunks: {result['selected_chunks']}")
    print(f"Used summaries: {result['used_summaries']}")
    print(f"Total length: {result['total_tokens']} tokens")
    print(f"Context preview: {result['context'][:500]}...")

    # Save context to file for inspection
    with open("description_context.txt", "w", encoding="utf-8") as f:
        f.write(result['context'])
    print("Full context saved to description_context.txt")
