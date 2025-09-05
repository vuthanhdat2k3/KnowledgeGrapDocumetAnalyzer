#!/usr/bin/env python3
"""
DOCX Knowledge Graph Builder
Builder chuyên biệt cho xử lý file DOCX và tạo Knowledge Graph
"""
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import uuid
import os

from src.core.kg_builders.base_kg_builder import BaseKGBuilder
from src.core.knowledge_graph.kg_operations import kg_operations
from src.core.knowledge_graph.neo4j_manager import neo4j_manager

logger = logging.getLogger(__name__)


class DocxKGBuilder(BaseKGBuilder):
    """
    Knowledge Graph Builder cho file DOCX
    """

    def __init__(self):
        self.neo4j_manager = neo4j_manager
        self.kg_operations = kg_operations
        self._ensure_connection()

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

    def build(self, json_data_path: str, document_name: str) -> Dict[str, Any]:
        try:
            logger.info(f"Starting to build knowledge graph from {json_data_path}")

            # Load JSON data
            chunks_data = self._load_json_data(json_data_path)

            # Import data vào Neo4j
            stats = self.import_graph_data(chunks_data, document_name)
            logger.info(f"Successfully built knowledge graph. Stats: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error building knowledge graph: {str(e)}")
            raise

    def _load_json_data(self, json_data_path: str) -> List[Dict[str, Any]]:
        try:
            with open(json_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Error loading JSON data: {str(e)}")
            raise

    def import_graph_data(self, chunks_data: List[Dict[str, Any]], document_name: str) -> Dict[str, Any]:
        try:
            stats = {
                "document_nodes": 0,
                "chunk_nodes": 0,
                "entity_nodes": 0,
                "entity_description_nodes": 0,
                "relationship_nodes": 0,
                "relationship_description_nodes": 0,
                "edges_created": 0
            }

            document_id = self._create_document_node(chunks_data, document_name)
            stats["document_nodes"] = 1

            for index, chunk_data in enumerate(chunks_data):
                chunk_stats = self._process_chunk(chunk_data, document_id, index + 1)
                for key in chunk_stats:
                    if key in stats:
                        stats[key] += chunk_stats[key]

            logger.info(f"Import completed. Total stats: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error importing graph data: {str(e)}")
            raise

    def _create_document_node(self, chunks_data: List[Dict[str, Any]], document_name: str) -> str:
        try:
            doc_id = self.kg_operations.create_document(
                file_name=document_name,
                type="Document"
            )
            logger.info(f"Created Document node success!")
            return doc_id
        except Exception as e:
            logger.error(f"Error creating document node: {str(e)}")
            raise

    def _process_chunk(self, chunk_data: Dict[str, Any], document_id: str, chunk_number: int) -> Dict[str, Any]:
        try:
            chunk_stats = {
                "chunk_nodes": 0,
                "entity_nodes": 0,
                "entity_description_nodes": 0,
                "relationship_nodes": 0,
                "relationship_description_nodes": 0,
                "edges_created": 0
            }

            chunk_id = self._create_chunk_node(chunk_data, chunk_number)
            chunk_stats["chunk_nodes"] = 1

            if self.kg_operations.create_edge(document_id, chunk_id, "CONTAINS"):
                chunk_stats["edges_created"] += 1

            entities_map = {}

            for entity_data in chunk_data.get('entities', []):
                if not isinstance(entity_data, dict):
                    logger.warning(f"Invalid entity data (not dict): {entity_data}")
                    continue

                entity_id, desc_created = self._create_entity_node(entity_data)
                entities_map[entity_data.get('id')] = entity_id

                chunk_stats["entity_nodes"] += 1
                if desc_created:
                    chunk_stats["entity_description_nodes"] += 1

                if self.kg_operations.create_edge(chunk_id, entity_id, "MENTIONS"):
                    chunk_stats["edges_created"] += 1

            for rel_data in chunk_data.get('relationships', []):
                if not isinstance(rel_data, dict):
                    logger.warning(f"Invalid relationship data (not dict): {rel_data}")
                    continue

                rel_id, desc_created = self._create_relationship_node(rel_data, entities_map)
                if not rel_id:
                    continue

                chunk_stats["relationship_nodes"] += 1
                if desc_created:
                    chunk_stats["relationship_description_nodes"] += 1

            logger.info(f"Processed chunk {chunk_data.get('chunk_id', chunk_number)}: {chunk_stats}")
            return chunk_stats

        except Exception as e:
            logger.error(f"Error processing chunk {chunk_data.get('chunk_id', 'unknown')}: {str(e)}")
            raise

    def _create_chunk_node(self, chunk_data: Dict[str, Any], chunk_number: int) -> str:
        try:
            chunk_id = str(uuid.uuid4())
            chunk_text = chunk_data.get('chunk_text', '')
            summary = chunk_data.get('summary', '')
            metadata = chunk_data.get('metadata', {})
            embedding = chunk_data.get('embedding', [])

            if isinstance(embedding, str):
                embedding = []

            created_id = self.kg_operations.create_chunk(
                text=chunk_text,
                summary=summary,
                chunk_number=chunk_number,
                metadata=metadata,
                embedding=embedding,
                chunk_id=chunk_id
            )
            return created_id
        except Exception as e:
            logger.error(f"Error creating chunk node: {str(e)}")
            raise

    def _create_entity_node(self, entity_data: Dict[str, Any]) -> tuple[str, bool]:
        try:
            entity_id = self.kg_operations.create_entity(
                name=entity_data.get('name', 'Unknown'),
                entity_type=entity_data.get('type', 'Unknown'),
                properties={
                    "original_id": entity_data.get('id', '')
                }
            )

            description_created = False
            description_node = entity_data.get('description_node')

            if isinstance(description_node, dict) and description_node.get('text'):
                desc_text = description_node['text']
                desc_embedding = description_node.get('embedding', [])
                if isinstance(desc_embedding, str):
                    desc_embedding = []
                additional_properties = {
                    "type": description_node.get('type', 'unknown')
                }
                desc_el_id = self.kg_operations.create_description(
                    text=desc_text,
                    embeddings=desc_embedding,
                    additional_properties=additional_properties
                )
                description_created = True
                self.kg_operations.create_edge(
                    from_entity_id=entity_id,
                    to_entity_id=desc_el_id,
                    relationship_type="DESCRIBES"
                )

            return entity_id, description_created

        except Exception as e:
            logger.error(f"Error creating entity node {entity_data.get('name', 'unknown')}: {str(e)}")
            raise

    def _create_relationship_node(self, rel_data: Dict[str, Any], entities_map: Dict[str, str]) -> tuple[Optional[str], bool]:
        try:
            source_entity_id = entities_map.get(rel_data.get('source'))
            target_entity_id = entities_map.get(rel_data.get('target'))

            if not source_entity_id or not target_entity_id:
                logger.warning(f"Missing entities for relationship {rel_data.get('id')}: "
                               f"source='{rel_data.get('source')}', target='{rel_data.get('target')}'")
                return None, False

            relationship_type = rel_data.get('type', 'RELATED')
            rel_id = self.kg_operations.create_relationship(
                relationship_type=relationship_type,
                properties={
                    "original_id": rel_data.get('id', ''),
                    "from_entity_id": source_entity_id,
                    "to_entity_id": target_entity_id
                }
            )

            self.kg_operations.create_edge(source_entity_id, rel_id, "RELATED")
            self.kg_operations.create_edge(target_entity_id, rel_id, "RELATED")

            description_created = False
            description_node = rel_data.get('description_node')

            if isinstance(description_node, dict) and description_node.get('text'):
                desc_text = description_node['text']
                desc_embedding = description_node.get('embedding', [])
                if isinstance(desc_embedding, str):
                    desc_embedding = []
                additional_properties = {
                    "type": description_node.get('type', 'unknown')
                }
                desc_rel_id = self.kg_operations.create_description(
                    text=desc_text,
                    embeddings=desc_embedding,
                    additional_properties=additional_properties
                )
                self.kg_operations.create_edge(
                    from_entity_id=rel_id,
                    to_entity_id=desc_rel_id,
                    relationship_type="DESCRIBES"
                )
                description_created = True

            return rel_id, description_created

        except Exception as e:
            logger.error(f"Error creating relationship {rel_data.get('id', 'unknown')}: {str(e)}")
            raise


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        json_path = "src/core/knowledge_graph/docx1_final.json"
        document_name = "01-L-ROCR-002-2025-Conduit-Rd-Schoolhouse-RFP_gpt_nano (1).md"
        if not os.path.exists(json_path):
            print(f"❌ File not found: {json_path}")
            return
        builder = DocxKGBuilder()
        print("🚀 Building knowledge graph...")
        stats = builder.build(json_path, document_name)

        print("\n✅ BUILD COMPLETED!")
        print(f"📄 Documents: {stats['document_nodes']}")
        print(f"📝 Chunks: {stats['chunk_nodes']}")
        print(f"🏷️  Entities: {stats['entity_nodes']}")
        print(f"📋 Entity Descriptions: {stats['entity_description_nodes']}")
        print(f"🔗 Relationships: {stats['relationship_nodes']}")
        print(f"📝 Relationship Descriptions: {stats['relationship_description_nodes']}")
        print(f"⚡ Edges: {stats['edges_created']}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
