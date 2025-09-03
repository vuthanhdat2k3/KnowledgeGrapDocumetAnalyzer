"""
Excel Knowledge Graph Builder
Builder chuyên biệt cho xử lý file Excel và tạo Knowledge Graph
"""
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from src.core.kg_builders.base_kg_builder import BaseKGBuilder
from src.core.knowledge_graph.kg_operations import kg_operations
from src.core.knowledge_graph.neo4j_manager import neo4j_manager

logger = logging.getLogger(__name__)


class ExcelKGBuilder(BaseKGBuilder):
    """
    Knowledge Graph Builder cho file Excel
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

    def build(self, file_path: str, document_name: str = None) -> bool:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            
            logger.info(f"Loaded {len(chunks_data)} chunks from {file_path}")
            return self.import_graph_data(chunks_data, document_name)
            
        except Exception as e:
            logger.error(f"Failed to import from file {file_path}: {e}")
            return False

    def import_graph_data(self, chunks_data: List[Dict[str, Any]], document_name: str = None) -> bool:
        try:
            self._ensure_connection()
            
            # Create Document node if specified
            doc_node_id = None
            if document_name:
                logger.info(f"Creating document node: {document_name}")
                doc_node_id = self.kg_operations.create_document(file_name=document_name, type="excel")

            # Global entity tracking across all chunks
            global_entity_map: Dict[str, str] = {}

            # Process each chunk
            successful_chunks = 0
            for i, chunk_data in enumerate(chunks_data):
                try:
                    logger.info(f"Processing chunk {i+1}/{len(chunks_data)}")
                    self._process_chunk(chunk_data, doc_node_id, global_entity_map)
                    successful_chunks += 1
                except Exception as e:
                    logger.error(f"Failed to process chunk {i+1}: {e}")
                    continue

            logger.info(f"Successfully imported {successful_chunks}/{len(chunks_data)} chunks")
            return successful_chunks == len(chunks_data)

        except Exception as e:
            logger.error(f"Failed to import graph data: {e}")
            return False

    def _process_chunk(self, chunk_data: Dict[str, Any], doc_node_id: Optional[str] = None, global_entity_map: Optional[Dict[str, str]] = None):
        if global_entity_map is None:
            global_entity_map = {}
            
        chunk_number = chunk_data.get("chunk_id")
        
        # Create Chunk node
        chunk_db_id = self.kg_operations.create_chunk(
            text = chunk_data.get("text", ""),
            summary=chunk_data.get("chunk_summary", ""),
            chunk_number=chunk_number,
            embedding=chunk_data.get("chunk_embedding", []),
            metadata=chunk_data.get("metadata", {}),
        )

        # Link Document -> Chunk if document exists
        if doc_node_id:
            self.kg_operations.create_edge(doc_node_id, chunk_db_id, "CONTAINS")

        # Local node ID mapping for this chunk
        node_id_map: Dict[str, str] = {}

        # Process nodes
        for node in chunk_data.get("nodes", []):
            node_db_id = self._create_node(node, chunk_db_id, global_entity_map)
            if node_db_id:
                node_id_map[node["id"]] = node_db_id
        
        entity_has_item_link = set()
        for rel in chunk_data.get("edges", []):
            src, tgt = rel.get("source"), rel.get("target")
            src_node = next((n for n in chunk_data["nodes"] if n["id"] == src), None)
            tgt_node = next((n for n in chunk_data["nodes"] if n["id"] == tgt), None)
            if src_node and tgt_node:
                if src_node.get("type") == "Entity" and tgt_node.get("type") == "Item":
                    entity_has_item_link.add(src)
                if tgt_node.get("type") == "Entity" and src_node.get("type") == "Item":
                    entity_has_item_link.add(tgt)

        # Connect entities/tables -> chunks
        for node in chunk_data.get("nodes", []):
            node_type = node.get("type")
            node_id = node.get("id")
            if node_type == "Entity" and node_id in node_id_map:
                if node_id not in entity_has_item_link:
                    self.kg_operations.create_edge(chunk_db_id, node_id_map[node["id"]], "MENTIONS")  

            elif node_type == "Table" and node_id in node_id_map:
                self.kg_operations.create_edge(chunk_db_id, node_id_map[node["id"]], "CONTAINS")      

        # Process relationships/edges
        for rel in chunk_data.get("edges", []):
            self._create_relationship(rel, node_id_map)

    def _create_node(self, node: Dict[str, Any], chunk_db_id: str, global_entity_map: Dict[str, str]) -> Optional[str]:
        node_type = node.get("type")
        node_props = node.get("properties", {})
        if "id" in node_props:
            del node_props["id"]
        node_db_id = None

        try:
            if node_type == "Table":
                node_db_id = self._create_table_node(node, node_props)
                
            elif node_type == "Item":
                node_db_id = self._create_item_node(node, node_props)
                
            elif node_type == "Description":
                node_db_id = self._create_description_node(node, chunk_db_id)
                
            elif node_type == "Entity":
                node_db_id = self._create_entity_node(node, node_props, global_entity_map)
                
            else:
                # Handle Relationship nodes
                node_db_id = self._create_relationship_node(node, node_props)

            return node_db_id

        except Exception as e:
            logger.error(f"Failed to create {node_type} node: {e}")
            return None

    def _create_table_node(self, node: Dict[str, Any], node_props: Dict[str, Any]) -> str:
        cols = node_props.get("columns")
        if isinstance(cols, str):
            columns_list = [c.strip() for c in cols.split(",")]
        elif isinstance(cols, list):
            columns_list = cols
        else:
            columns_list = []
            
        return self.kg_operations.create_table(
            name=node_props.get("table_name", node.get("name", node.get("id"))),
            columns=columns_list,
            table_id=node.get("id", None),
        )

    def _create_item_node(self, node: Dict[str, Any], node_props: Dict[str, Any]) -> str:
        prefixed_props = {f"prop:{k}": v for k, v in node_props.items()}

        return self.kg_operations.create_item(
            name=node.get("name", node.get("id")),
            item_properties=prefixed_props,
            item_id=node.get("id", None),
        )


    def _create_description_node(self, node: Dict[str, Any], chunk_id : str) -> str:
        additional_properties = {'chunk_id': chunk_id}
        return self.kg_operations.create_description(
            text=node.get("text", ""),
            description_id=node.get("id", None),
            embeddings=node.get("embedding", None),
            additional_properties=additional_properties
        )

    def _create_entity_node(self, node: Dict[str, Any], node_props: Dict[str, Any], global_entity_map: Dict[str, str]) -> str:
        entity_name = node.get("name", node.get("id"))
        entity_type = node.get("entity_type", "Unknown")
        
        # Create unique key for entity deduplication
        entity_key = f"{entity_name}:{entity_type}"
        
        # Check if entity already exists in global map
        if entity_key in global_entity_map:
            logger.debug(f"Reusing existing entity: {entity_name} ({entity_type})")
            return global_entity_map[entity_key]
        
        # Create or get entity (this may return existing DB node)
        node_db_id = self.kg_operations.create_or_get_entity(
            name=entity_name,
            entity_type=entity_type,
            properties=node_props,
        )
        
        # Store in global map for future reference
        global_entity_map[entity_key] = node_db_id
        logger.debug(f"Created/found entity: {entity_name} ({entity_type}) -> {node_db_id}")
        
        return node_db_id

    def _create_relationship_node(self, node: Dict[str, Any], node_props: Dict[str, Any]) -> str:
        return self.kg_operations.create_relationship(
            relationship_type=node.get("type", "Unknown"),
            properties=node_props,
            relationship_id=node.get("id", None),
        )

    def _create_relationship(self, rel: Dict[str, Any], node_id_map: Dict[str, str]):
        src_id = node_id_map.get(rel["source"])
        tgt_id = node_id_map.get(rel["target"])
        
        if not src_id or not tgt_id:
            logger.warning(f"Missing node IDs for relationship: {rel}")
            return
            
        try:
            self.kg_operations.create_edge(
                src_id, tgt_id, rel["type"], rel.get("properties", {})
            )
        except Exception as e:
            logger.error(f"Failed to create relationship {rel['type']}: {e}")

    def close_connection(self):
        try:
            if self.neo4j_manager.is_connected():
                self.neo4j_manager.close()
                logger.info("Neo4j connection closed")
        except Exception as e:
            logger.error(f"Error closing Neo4j connection: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


if __name__ == "__main__":
    graph_importer = ExcelKGBuilder()
    success = graph_importer.build(
        file_path="data/graph/input.json",
        document_name="document name",
    )
    
    if success:
        logger.info("Graph import completed successfully")
    else:
        logger.error("Graph import failed")