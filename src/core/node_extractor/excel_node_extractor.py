import logging
import os
import asyncio
import secrets
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.core.node_extractor.prompts.excel_prompts import ADDITIONAL_INSTRUCTION
from src.core.llm_client import LLMClientFactory

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExcelNodeExtractor:
    """Extracts graph structures from documents using LLM-based transformation."""

    def __init__(self, model: str = "gpt-4.1"):
        try:
            factory = LLMClientFactory()
            client_info = factory.get_client(model)

            self.llm = ChatOpenAI(
                model=client_info["model"],
                api_key=factory.api_key,
                base_url=factory.base_url + "/v1",
            )

            embedding_client_info = factory.get_client("embedding")
            # Initialize embeddings client for Description nodes
            self.embeddings = OpenAIEmbeddings(
                model=embedding_client_info["model"],
                api_key=factory.api_key,
                base_url=factory.base_url + "/v1",
            )

            self.graph_transformer = LLMGraphTransformer(
                llm=self.llm,
                node_properties=True,
                relationship_properties=True,
                additional_instructions=ADDITIONAL_INSTRUCTION,
            )

            logger.info(f"ExcelNodeExtractor initialized with model: {model}")

        except Exception as e:
            logger.error(f"Failed to initialize ExcelNodeExtractor: {e}")
            raise

    async def get_chunk_summary(self, chunk_text: str, max_length: int = 200) -> str:
        try:
            summary_prompt = f"""
            Summarize the following text in no more than {max_length} words.
            Output only the summary text, without any explanations, labels, or prefixes.

            Text:
            {chunk_text}

            Summary:
            """

            response = await self.llm.ainvoke(summary_prompt)
            summary = response.content.strip()
            return summary

        except Exception as e:
            logger.warning(f"Failed to generate summary: {e}")
            return chunk_text[:500] + "..." if len(chunk_text) > 500 else chunk_text

    def get_text_embedding(self, text: str) -> List[float]:
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return []

    def _generate_unique_id(self, node_type: str) -> str:
        rand_suffix = secrets.token_hex(4)
        return f"{node_type.lower()}_{rand_suffix}"

    def _embed_text(self, text: str) -> List[float]:
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.warning(f"Failed to generate embedding for text: {e}")
            return []

    def normalize_graph(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        normalized_nodes = []
        normalized_edges = []
        created_nodes = set()
        id_map = {}

        seen_nodes = set()
        seen_edges = set()

        def generate_fallback_description(
            node_type: str, name: str, properties: Dict[str, Any]
        ) -> str:
            """Generate a fallback description based on node type, name, and properties."""
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

        def add_node(
            original_id: str, node_type: str, properties: Dict[str, Any]
        ) -> str:
            node_key = (original_id, node_type)
            if node_key in seen_nodes:
                # Node đã tồn tại, trả về ID đã map
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

        def add_edge(
            source_id: str,
            target_id: str,
            relationship_type: str,
            properties: Optional[Dict[str, Any]] = None,
        ):
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

        # Process nodes
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

            # Process description with fallback
            description_text = properties.get("description") or properties.get("desc")
            if not description_text:
                # Generate fallback description
                node_name = properties.get("name") or original_id
                description_text = generate_fallback_description(
                    node_type, node_name, properties
                )
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

        # Process relationships
        for rel in graph_data.get("relationships", []):
            source_original = rel["source"]["id"]
            target_original = rel["target"]["id"]
            rel_type = rel["type"]
            rel_properties = rel.get("properties", {})

            # Handle CONTAINS relationship between Table and Item
            if (
                rel_type == "CONTAINS"
                and source_original in table_nodes
                and target_original in item_nodes
            ):
                add_edge(id_map[source_original], id_map[target_original], "CONTAINS")
                continue

            # Handle mentions between Items and Entities
            source_is_item = source_original in item_nodes
            target_is_item = target_original in item_nodes
            source_is_entity = source_original in entity_nodes
            target_is_entity = target_original in entity_nodes

            if source_is_item and target_is_entity:
                add_edge(
                    id_map[source_original],
                    id_map[target_original],
                    "MENTIONS",
                    rel_properties,
                )
                continue
            elif source_is_entity and target_is_item:
                add_edge(
                    id_map[target_original],
                    id_map[source_original],
                    "MENTIONS",
                    rel_properties,
                )
                continue

            # Create relationship node for other relationships
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

            add_edge(id_map[source_original], rel_node_id, "SOURCE")
            add_edge(rel_node_id, id_map[target_original], "TARGET")

            # Process relationship description with fallback
            desc_text = rel_properties.get("description")
            if not desc_text:
                # Generate fallback description for relationship
                desc_text = generate_fallback_description(
                    "Relationship",
                    rel_node_name,
                    {
                        **rel_properties,
                        "relationship_type": rel_type,
                        "source": rel["source"],
                        "target": rel["target"],
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

    async def extract_graph(
        self,
        chunks: List[Dict[str, Any]],
        include_summary: bool = True,
        normalize_graph: bool = True,
    ) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Processing {len(chunks)} chunks for graph extraction")
            graph_documents = []

            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")

                chunk_text = chunk.get("content", "")
                chunk_metadata = chunk.get("metadata", {})
                chunk_id = chunk.get("chunk_id")

                # Generate summary and embeddings for the chunk if requested
                chunk_summary = ""
                chunk_embedding = []

                if include_summary and chunk_text:
                    logger.info(f"Generating summary for chunk {i+1}")
                    chunk_summary = await self.get_chunk_summary(chunk_text)
                    chunk_embedding = self.get_text_embedding(chunk_summary)

                # Create document for graph transformation
                document = Document(page_content=chunk_text, metadata=chunk_metadata)

                # Extract graph structure
                graph_docs = await self.graph_transformer.aconvert_to_graph_documents(
                    [document]
                )

                if graph_docs:
                    graph_doc = graph_docs[0]

                    # Build graph data structure
                    graph_data = {
                        "chunk_id": chunk_id,
                        "nodes": [node.model_dump() for node in graph_doc.nodes],
                        "relationships": [
                            rel.model_dump() for rel in graph_doc.relationships
                        ],
                        "metadata": chunk_metadata,
                        "text": chunk_text,
                    }
                    if normalize_graph:
                        graph_data = self.normalize_graph(graph_data)

                    # Add summary and embedding data if generated
                    if include_summary:
                        graph_data["chunk_summary"] = chunk_summary
                        graph_data["chunk_embedding"] = chunk_embedding

                    graph_documents.append(graph_data)
                else:
                    logger.warning(f"No graph extracted for chunk {i+1}")

            logger.info(
                f"Successfully processed {len(graph_documents)} graph documents"
            )
            return graph_documents

        except Exception as e:
            logger.error(f"Failed to extract graphs: {e}")
            raise

    async def extract_graphs_from_documents(
        self, documents: List[Document]
    ) -> List[Any]:
        try:
            logger.info(f"Extracting graphs from {len(documents)} documents")
            graph_documents = await self.graph_transformer.aconvert_to_graph_documents(
                documents
            )
            logger.info(
                f"Successfully extracted {len(graph_documents)} graph documents"
            )
            return graph_documents
        except Exception as e:
            logger.error(f"Failed to extract graphs from documents: {e}")
            raise


async def main():
    try:
        chunks_path = "data/chunk/input.json"
        output_path = chunks_path.replace(".json", "new_format_graph.json").replace(
            "chunk", "graph"
        )

        node_extractor = ExcelNodeExtractor()

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        extracted_graph = await node_extractor.extract_graph(
            chunks, include_summary=True, normalize_graph=True
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extracted_graph, f, indent=2, ensure_ascii=False)

        logger.info(
            "Graph extraction, normalization, and summarization completed successfully!"
        )

    except Exception as e:
        logger.error(f"Graph extraction failed: {e}")
        raise


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
