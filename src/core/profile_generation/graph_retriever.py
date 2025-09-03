import logging

from src.core.knowledge_graph.neo4j_manager import neo4j_manager
from src.core.llm_client import LLMClientFactory

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GraphRetriever:
    def __init__(self):
        self.neo4j_manager = neo4j_manager
        self._ensure_connection()

        factory = LLMClientFactory()
        embedding_client_info = factory.get_client("embedding")
        self.embedding_model = embedding_client_info["model"]
        self.embedding_client = embedding_client_info["client"]

    def _ensure_connection(self):
        try:
            if not self.neo4j_manager.is_connected():
                logger.info("Connecting to Neo4j database...")
                self.neo4j_manager.connect()
                logger.info("Successfully connected to Neo4j")
            else:
                logger.info("Neo4j connection already established")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def _get_embedding(self, text: str):
        response = self.embedding_client.embeddings.create(
            model=self.embedding_model,
            input=[text]
        )
        return response.data[0].embedding

    def _get_semantic_score(self, document_id, query_vector):
        query = """
        WITH $query_vector AS queryVector
        MATCH (d:Document {id: $document_id})
        MATCH (d)-[*1..9]-(n)
        WHERE (n.embedding IS NOT NULL OR n.embeddings IS NOT NULL)
        AND (size(n.embedding) > 0 OR size(n.embeddings) > 0)
        WITH COLLECT(DISTINCT n) AS nodes, queryVector
        UNWIND nodes AS n
        WITH n, gds.similarity.cosine(coalesce(n.embedding, n.embeddings), queryVector) AS similarity
        ORDER BY similarity DESC
        RETURN n, similarity, labels(n) AS labels
        """
        params = {
            "document_id": document_id,
            "query_vector": query_vector
        }
        return self.neo4j_manager.execute_query(query=query, parameters=params)
    
    def _get_chunks_for_description(self, description_id):
        query = """
        MATCH (c:Chunk)-[*1..6]-(d:Description {id: $description_id})
        RETURN DISTINCT c
        """
        params = {"description_id": description_id}
        return self.neo4j_manager.execute_query(query=query, parameters=params)

    def retrieve(self, document_id: str, text: str, top_k: int = 5):
        query_vector = self._get_embedding(text)
        records = self._get_semantic_score(document_id, query_vector)

        full_context = ""
        seen_chunk_ids = set()
        
        for record in records:
            if len(seen_chunk_ids) >= top_k:
                break

            node = record["n"]
            node_labels = record.get("labels", [])
            if "Chunk" in node_labels:
                chunk_id = node.get("id")
                if chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    chunk_text = node.get("summary", "")
                    full_context += f"\n{chunk_text}"

            elif "Description" in node_labels:
                description_id = node.get("id")
                chunks_records = self._get_chunks_for_description(description_id)
                for chunk_rec in chunks_records:
                    chunk_node = chunk_rec["c"]
                    chunk_id = chunk_node.get("id")
                    if chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk_id)
                        chunk_text = chunk_node.get("summary", "")
                        full_context += f"\n{chunk_text}"

        return full_context

# Example usage
if __name__ == "__main__":
    retriever = GraphRetriever()
    document_id = "ad45f4b6-3860-49ba-bf22-8370c2c4df8e"
    text_query = "Extract the industry where the project is applied. Keep it simple, like healthcare, finance, education, IT, construction, manufacturing, or public sector."
    context = retriever.retrieve(document_id, text_query, top_k=10)
    print("Retrieved Context:")
    print(context)
    with open("context.txt", "w", encoding="utf-8") as f:
        f.write(context)