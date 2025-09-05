import logging
import os
import asyncio
import secrets
import json
import time
import random
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from src.core.llm_client import LLMClientFactory

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExcelNodeExtractor:
    """Extracts graph structures from documents using LLM-based transformation."""

    def __init__(self, model_name: str = "gemini-2.5-flash", embedding_model: str = "embedding"):
        try:
            factory = LLMClientFactory()
            self.llm_client = factory.get_client(model_name)
            self.embeddings = factory.get_client(embedding_model)
            logging.info(f"ExcelNodeExtractor initialized with model: {model_name}, embedding_model: {embedding_model}")
        except Exception as e:
            logging.error(f"Failed to initialize ExcelNodeExtractor: {e}")
            raise

    async def call_with_retry(self, func, *args, retries: int = 5, backoff: float = 2.0, **kwargs):
        """
        Helper function để gọi API LLM với retry khi gặp lỗi 503 UNAVAILABLE hoặc lỗi mạng.
        """
        for attempt in range(retries):
            try:
                return await asyncio.to_thread(func, *args, **kwargs)
            except Exception as e:
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    wait_time = backoff * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"503 UNAVAILABLE, retry {attempt+1}/{retries} sau {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    raise
        raise RuntimeError("Failed after max retries due to repeated 503 errors")

    async def get_chunk_summary(self, chunk_text: str, max_length: int = 100) -> str:
        """Sinh summary ngắn gọn cho 1 chunk."""
        try:
            prompt = (
                "INSTRUCTION: You are an expert summarizer.\n\n"
                f"TASK: Summarize the following text in no more than {max_length} words.\n\n"
                "Return ONLY the summary text, without any explanations, labels, or prefixes.\n\n"
                f"Text:\n{chunk_text}\n\nSummary:"
            )
            response = await self.call_with_retry(
                self.llm_client["client"].models.generate_content,
                model=self.llm_client["model"],
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config={"temperature": 0.3, "max_output_tokens": max_length + 50}
            )
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if candidate.content.parts:
                    return candidate.content.parts[0].text.strip()
            return chunk_text
        except Exception as e:
            logging.warning(f"Failed to generate summary: {e}")
            return chunk_text[:500] + "..." if len(chunk_text) > 500 else chunk_text

    def get_text_embedding(self, text: str) -> List[float]:
        try:
            if self.embeddings["type"] == "embedding_sentence_transformer":
                return self.embeddings["client"].encode([text])[0].tolist()
            else:
                response = self.embeddings["client"].embeddings.create(
                    model=self.embeddings["model"],
                    input=text
                )
                return response.data[0].embedding
        except Exception as e:
            logging.warning(f"Failed to generate embedding: {e}")
            return []

    def _generate_unique_id(self, node_type: str) -> str:
        rand_suffix = secrets.token_hex(4)
        return f"{node_type.lower()}_{rand_suffix}"

    def _embed_text(self, text: str) -> List[float]:
        return self.get_text_embedding(text)

    def normalize_graph(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        normalized_nodes = []
        normalized_edges = []
        created_nodes = set()
        id_map = {}

        seen_nodes = set()
        seen_edges = set()

        def generate_fallback_description(node_type: str, name: str, properties: Dict[str, Any]) -> str:
            if node_type == "Table":
                columns = properties.get("columns")
                table_name = properties.get("table_name", properties.get("name", name))
                if columns:
                    return f"Table '{table_name}' containing columns: {columns}"
                else:
                    return f"Table entity named '{table_name}'"
            elif node_type == "Item":
                item_type = properties.get("item_type", "item")
                return f"An {item_type} record named '{name}'"
            elif node_type == "Relationship":
                rel_type = properties.get("relationship_type", "relationship")
                source = properties.get("source")
                target = properties.get("target")
                if not source or not target:
                    return f"A {rel_type} relationship connecting entitites"
                else:
                    return f"A {rel_type} relationship connecting {source} to {target}"
            else:
                attributes = []
                for key, value in properties.items():
                    if key not in ["entity_type", "name"] and value:
                        attributes.append(f"{key}: {value}")
                if attributes:
                    return f"A {node_type} entity named '{name}' with {', '.join(attributes)}"
                else:
                    return f"A {node_type} entity named '{name}'"

        def add_node(original_id: str, node_type: str, properties: Dict[str, Any]) -> str:
            node_key = (original_id, node_type)
            if node_key in seen_nodes:
                return id_map.get(original_id)
            new_id = self._generate_unique_id(node_type)
            if new_id not in created_nodes:
                node_data = {
                    "id": new_id,
                    "name": original_id,
                    "type": node_type,
                    "properties": properties,
                }
                normalized_nodes.append(node_data)
                created_nodes.add(new_id)
            return new_id

        def add_edge(source_id: str, target_id: str, relationship_type: str, properties: Optional[Dict[str, Any]] = None):
            edge_key = (source_id, target_id, relationship_type)
            if edge_key in seen_edges:
                return
            normalized_edges.append(
                {
                    "source": source_id,
                    "target": target_id,
                    "type": relationship_type,
                    "properties": properties or {},
                }
            )
            seen_edges.add(edge_key)

        table_nodes = set()
        item_nodes = set()
        entity_nodes = set()

        for node in graph_data.get("nodes", []):
            original_id = node["id"]
            node_type = node["type"]
            properties = node.get("properties", {})

            if node_type == "Table":
                new_id = add_node(original_id, "Table", properties)
                table_nodes.add(original_id)
            elif node_type == "Item":
                new_id = add_node(original_id, "Item", properties)
                item_nodes.add(original_id)
            else:
                new_id = self._generate_unique_id("Entity")
                new_entity_node = {
                    "id": new_id,
                    "name": properties.get("name") or original_id,
                    "type": "Entity",
                    "entity_type": node_type,
                    "properties": properties,
                }
                normalized_nodes.append(new_entity_node)
                created_nodes.add(new_id)
                entity_nodes.add(original_id)

            id_map[original_id] = new_id

            description_text = properties.get("description") or properties.get("desc")
            if not description_text:
                node_name = properties.get("name") or original_id
                description_text = generate_fallback_description(node_type, node_name, properties)
                node["properties"]["description"] = description_text

            if description_text:
                desc_id = self._generate_unique_id("Description")
                if desc_id not in created_nodes:
                    normalized_nodes.append(
                        {
                            "id": desc_id,
                            "type": "Description",
                            "text": description_text,
                            "embedding": self._embed_text(description_text),
                        }
                    )
                    created_nodes.add(desc_id)
                add_edge(id_map[original_id], desc_id, "DESCRIBE")

        for rel in graph_data.get("relationships", []):
            def extract_id(val):
                if isinstance(val, dict):
                    return val.get("id")
                return val

            source_original = extract_id(rel.get("source"))
            target_original = extract_id(rel.get("target"))
            rel_type = rel.get("type")
            rel_properties = rel.get("properties", {})

            if rel_type == "CONTAINS" and source_original in table_nodes and target_original in item_nodes:
                add_edge(id_map[source_original], id_map[target_original], "CONTAINS")
                continue

            source_is_item = source_original in item_nodes
            target_is_item = target_original in item_nodes
            source_is_entity = source_original in entity_nodes
            target_is_entity = target_original in entity_nodes

            if source_is_item and target_is_entity:
                add_edge(id_map[source_original], id_map[target_original], "MENTIONS", rel_properties)
                continue
            elif source_is_entity and target_is_item:
                add_edge(id_map[target_original], id_map[source_original], "MENTIONS", rel_properties)
                continue

            rel_node_id = self._generate_unique_id("Relationship")
            rel_node_name = f"{source_original}_{rel_type}_{target_original}"
            normalized_nodes.append(
                {
                    "id": rel_node_id,
                    "name": rel_node_name,
                    "type": "Relationship",
                    "properties": {
                        "relationship_type": rel_type,
                        **rel_properties,
                    },
                }
            )
            created_nodes.add(rel_node_id)

            if source_original in id_map:
                add_edge(id_map[source_original], rel_node_id, "SOURCE")
            else:
                logger.warning(f"Source node '{source_original}' không tồn tại trong id_map khi tạo edge SOURCE cho relationship {rel_node_name}")

            if target_original in id_map:
                add_edge(rel_node_id, id_map[target_original], "TARGET")
            else:
                logger.warning(f"Target node '{target_original}' không tồn tại trong id_map khi tạo edge TARGET cho relationship {rel_node_name}")

            desc_text = rel_properties.get("description")
            if not desc_text:
                desc_text = generate_fallback_description(
                    "Relationship",
                    rel_node_name,
                    {
                        **rel_properties,
                        "relationship_type": rel_type,
                        "source": rel.get("source"),
                        "target": rel.get("target"),
                    },
                )
                rel["properties"]["description"] = desc_text

            if desc_text:
                desc_id = self._generate_unique_id("Description")
                if desc_id not in created_nodes:
                    normalized_nodes.append(
                        {
                            "id": desc_id,
                            "type": "Description",
                            "text": desc_text,
                            "embedding": self._embed_text(desc_text),
                        }
                    )
                    created_nodes.add(desc_id)
                add_edge(rel_node_id, desc_id, "DESCRIBE")

        return {
            "chunk_id": graph_data.get("chunk_id"),
            "nodes": normalized_nodes,
            "edges": normalized_edges,
            "metadata": graph_data.get("metadata", {}),
            "text": graph_data.get("text", ""),
        }

    async def extract_graph(self, chunks: List[Dict[str, Any]], include_summary: bool = True, normalize_graph: bool = True) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Processing {len(chunks)} chunks for graph extraction")
            graph_documents = []

            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")

                chunk_text = chunk.get("content", "")
                chunk_metadata = chunk.get("metadata", {})
                chunk_id = chunk.get("chunk_id")

                chunk_summary = ""
                chunk_embedding = []

                if include_summary and chunk_text:
                    logger.info(f"Generating summary for chunk {i+1}")
                    chunk_summary = await self.get_chunk_summary(chunk_text)
                    chunk_embedding = self.get_text_embedding(chunk_summary)

                prompt = (
                    "INSTRUCTION: You are an expert at extracting knowledge graphs from tabular data.\n"
                    "TASK: Given the following Excel chunk, extract a list of nodes and relationships in JSON format.\n"
                    "Return a JSON object with keys: nodes (list), relationships (list).\n"
                    "Each node should have: id, type, properties.\n"
                    "Each relationship should have: source (id), target (id), type, properties.\n"
                    f"\nExcel Chunk:\n{chunk_text}\n\nJSON:"
                )
                response = await self.call_with_retry(
                    self.llm_client["client"].models.generate_content,
                    model=self.llm_client["model"],
                    contents=[{"role": "user", "parts": [{"text": prompt}]}],
                    config={"temperature": 0.0}
                )

                graph_data = {
                    "chunk_id": chunk_id,
                    "nodes": [],
                    "relationships": [],
                    "metadata": chunk_metadata,
                    "text": chunk_text,
                }
                try:
                    if hasattr(response, "candidates") and response.candidates:
                        candidate = response.candidates[0]
                        if candidate.content.parts:
                            raw_json = candidate.content.parts[0].text.strip()
                            json_start = raw_json.find("{")
                            json_end = raw_json.rfind("}")
                            if json_start != -1 and json_end != -1:
                                json_str = raw_json[json_start:json_end+1]
                                parsed = json.loads(json_str)
                                graph_data["nodes"] = parsed.get("nodes", [])
                                graph_data["relationships"] = parsed.get("relationships", [])
                            else:
                                logger.warning(f"Could not find JSON in Gemini response for chunk {i+1}")
                    else:
                        logger.warning(f"No candidates in Gemini response for chunk {i+1}")
                except Exception as e:
                    logger.warning(f"Failed to parse Gemini response for chunk {i+1}: {e}")

                if normalize_graph:
                    graph_data = self.normalize_graph(graph_data)

                if include_summary:
                    graph_data["chunk_summary"] = chunk_summary
                    graph_data["chunk_embedding"] = chunk_embedding

                graph_documents.append(graph_data)

            logger.info(f"Successfully processed {len(graph_documents)} graph documents")
            return graph_documents

        except Exception as e:
            logger.error(f"Failed to extract graphs: {e}")
            raise


async def main():
    try:
        chunks_path = "data/output/chunks/iiPay-Global-Payroll-Request-for-Proposal-Template-1_described.json"
        output_path = "data/output/nodes/iiPay-Global-Payroll-Request-for-Proposal-Template-1_described.json"

        node_extractor = ExcelNodeExtractor()

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        extracted_graph = await node_extractor.extract_graph(chunks, include_summary=True, normalize_graph=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extracted_graph, f, indent=2, ensure_ascii=False)

        logger.info("Graph extraction, normalization, and summarization completed successfully!")

    except Exception as e:
        logger.error(f"Graph extraction failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
