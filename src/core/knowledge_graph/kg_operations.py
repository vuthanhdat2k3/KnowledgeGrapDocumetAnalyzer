"""
Knowledge Graph Operations
Các operations cơ bản để thao tác với Knowledge Graph
"""
import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from .neo4j_manager import neo4j_manager


logger = logging.getLogger(__name__)

@dataclass
class Document:
    id: str
    file_name: str
    created_at: Optional[datetime] = None

@dataclass
class Chunk:
    id: str
    text: str
    summary: str
    chunk_number: int
    embedding: list[float]
    metadata: Dict[str, Any]
    created_at: Optional[datetime] = None

@dataclass
class Entity:
    """Đại diện cho một thực thể trong Knowledge Graph"""
    id: str
    name: str
    type: str
    properties: Dict[str, Any]
    created_at: Optional[datetime] = None

@dataclass
class Description:
    id: str
    text: str
    chunk_id: str
    embeddings: List[float]
    created_at: Optional[datetime] = None

@dataclass
class Table:
    id: str
    name: str
    created_at: Optional[datetime] = None
    columns: Optional[List[str]] = None

@dataclass
class Item:
    id: str
    name: str
    properties: Dict[str, Any]
    created_at: Optional[datetime] = None


@dataclass  
class Relationship:
    """Đại diện cho một mối quan hệ trong Knowledge Graph"""
    id: str
    type: str
    properties: Dict[str, Any]
    created_at: Optional[datetime] = None

class Edge:
    id: str
    from_entity_id: str
    to_entity_id: str
    type: str
    properties: Dict[str, Any]
    created_at: Optional[datetime] = None


class KGOperations:
    """Các operations cơ bản cho Knowledge Graph"""
    
    def __init__(self):
        self.manager = neo4j_manager
    
    def create_entity(self, name: str, entity_type: str, 
                     properties: Optional[Dict] = None, entity_id: Optional[str] = None,
                     labels: Optional[List[str]] = None) -> str:
        """
        Tạo một entity mới trong Knowledge Graph
        
        Args:
            name: Tên entity
            entity_type: Loại entity (Product, Feature, User, etc.)
            properties: Thuộc tính bổ sung
            entity_id: ID tùy chỉnh (nếu không có sẽ tự tạo)
            labels: Labels bổ sung cho entity (ngoài Entity chính)
            
        Returns:
            str: ID của entity đã tạo
        """
        entity_id = entity_id or str(uuid.uuid4())
        properties = properties or {}
        labels = labels or []
        
        # Thêm metadata
        properties.update({
            "id": entity_id,
            "name": name,
            "type": entity_type,
            "created_at": datetime.now().isoformat()
        })
        
        # Tạo label string với Entity làm label chính
        all_labels = ["Entity"] + labels
        label_str = ":".join(all_labels)
        
        query = f"""
        CREATE (e:{label_str} $properties)
        RETURN e.id as id
        """
        
        result = self.manager.execute_query(query, {"properties": properties})
        
        if result:
            logger.info(f"Đã tạo entity: {name} ({entity_type}) với ID: {entity_id}")
            return entity_id
        else:
            raise RuntimeError(f"Không thể tạo entity: {name}")
        
    def create_document(self, file_name: str, type: Optional[str] = "unknown", document_id: Optional[str] = None) -> str:
        """
        Tạo document node
        
        Args:
            file_name: Tên file
            document_id: ID tùy chỉnh
            
        Returns:
            str: ID của document đã tạo
        """
        document_id = document_id or str(uuid.uuid4())
        properties = {}
        
        properties.update({
            "id": document_id,
            "file_name": file_name,
            "type": type,
            "created_at": datetime.now().isoformat()
        })
        
        query = """
        CREATE (d:Document $properties)
        RETURN d.id as id
        """
        
        result = self.manager.execute_query(query, {"properties": properties})
        
        if result:
            logger.info(f"Đã tạo document: {file_name} với ID: {document_id}")
            return document_id
        else:
            raise RuntimeError(f"Không thể tạo document: {file_name}")

    def create_or_get_entity(self, name: str, entity_type: str,
                           properties: Optional[Dict] = None, entity_id: Optional[str] = None,
                           labels: Optional[List[str]] = None) -> str:
        """
        Tạo entity mới hoặc lấy entity đã tồn tại (UPSERT operation)
        
        Args:
            name: Tên entity
            entity_type: Loại entity
            properties: Thuộc tính bổ sung
            entity_id: ID tùy chỉnh
            labels: Labels bổ sung
            
        Returns:
            str: ID của entity
        """
        # Tìm entity theo name và type trước
        existing_entities = self.find_entities_by_name_and_type(name, entity_type)
        
        if existing_entities:
            logger.info(f"Entity đã tồn tại: {name} ({entity_type})")
            return existing_entities[0].id
        
        # Nếu chưa tồn tại thì tạo mới
        return self.create_entity(name, entity_type, properties, entity_id, labels)
    
    def create_chunk(self, text: str, summary: str, chunk_number: int, embedding: List[float],
                    metadata: Dict[str, Any], chunk_id: Optional[str] = None) -> str:
        """
        Tạo chunk node
        
        Args:
            text: Nội dung gốc của chunk
            summary: Tóm tắt chunk
            chunk_number: Số thứ tự chunk
            embedding: Vector embedding
            metadata: Metadata của chunk
            chunk_id: ID tùy chỉnh
            
        Returns:
            str: ID của chunk đã tạo
        """
        chunk_id = chunk_id or str(uuid.uuid4())
        
        # Chuẩn bị properties
        properties = {
            "id": chunk_id,
            "text": text,
            "summary": summary,
            "chunk_number": chunk_number,
            "created_at": datetime.now().isoformat()
        }
        
        # Thêm metadata
        properties.update(metadata)
        
        query = """
        CREATE (c:Chunk $properties)
        SET c.embedding = $embedding
        RETURN c.id as id
        """
        
        result = self.manager.execute_query(query, {
            "properties": properties,
            "embedding": embedding
        })
        
        if result:
            logger.info(f"Đã tạo chunk: {chunk_number} với ID: {chunk_id}")
            return chunk_id
        else:
            raise RuntimeError(f"Không thể tạo chunk: {chunk_number}")
        
    def create_description(self, text: str, description_id: Optional[str] = None, embeddings: Optional[List[float]] = None,
                          additional_properties: Optional[Dict] = None) -> str:
        """
        Tạo description node
        
        Args:
            text: Nội dung mô tả
            description_id: ID tùy chỉnh
            properties: Thuộc tính bổ sung
            
        Returns:
            str: ID của description đã tạo
        """
        description_id = description_id or str(uuid.uuid4())
        properties = additional_properties or {}
        
        properties.update({
            "id": description_id,
            "text": text,
            "embeddings": embeddings,
            "created_at": datetime.now().isoformat()
        })
        
        query = """
        CREATE (desc:Description $properties)
        RETURN desc.id as id
        """
        
        result = self.manager.execute_query(query, {"properties": properties})
        
        if result:
            logger.info(f"Đã tạo description với ID: {description_id}")
            return description_id
        else:
            raise RuntimeError(f"Không thể tạo description")
    
    def create_table(self, name: str, columns: Optional[Any] = None,
                    table_id: Optional[str] = None, properties: Optional[Dict] = None) -> str:
        """
        Tạo table node
        
        Args:
            name: Tên table
            columns: Danh sách cột
            table_id: ID tùy chỉnh
            properties: Thuộc tính bổ sung
            
        Returns:
            str: ID của table đã tạo
        """
        table_id = table_id or str(uuid.uuid4())
        properties = properties or {}
        columns = columns or []
        
        properties.update({
            "id": table_id,
            "name": name,
            "columns": columns,
            "column_count": len(columns),
            "created_at": datetime.now().isoformat()
        })
        
        query = """
        CREATE (t:Table $properties)
        RETURN t.id as id
        """
        
        result = self.manager.execute_query(query, {"properties": properties})
        
        if result:
            logger.info(f"Đã tạo table: {name} với ID: {table_id}")
            return table_id
        else:
            raise RuntimeError(f"Không thể tạo table: {name}")
    
    
    def create_item(self, name: str, item_properties: Dict[str, Any],
                   item_id: Optional[str] = None, additional_properties: Optional[Dict] = None) -> str:
        """
        Tạo item node
        
        Args:
            name: Tên item
            item_properties: Properties của item
            item_id: ID tùy chỉnh
            additional_properties: Thuộc tính bổ sung
            
        Returns:
            str: ID của item đã tạo
        """
        item_id = item_id or str(uuid.uuid4())
        additional_properties = additional_properties or {}
        
        # Merge all properties
        all_properties = {
            "id": item_id,
            "name": name,
            "created_at": datetime.now().isoformat(),
            **item_properties,
            **additional_properties
        }
        
        query = """
        CREATE (i:Item $properties)
        RETURN i.id as id
        """
        
        result = self.manager.execute_query(query, {"properties": all_properties})
        
        if result:
            logger.info(f"Đã tạo item: {name} với ID: {item_id}")
            return item_id
        else:
            raise RuntimeError(f"Không thể tạo item: {name}")
        
    def create_relationship(self, relationship_type: str, properties: Optional[Dict] = None,
                            relationship_id: Optional[str] = None) -> str:
        """
        Tạo relationship node
        
        Args:
            relationship_type: Loại quan hệ
            properties: Thuộc tính bổ sung
            relationship_id: ID tùy chỉnh cho relationship
            
        Returns:
            str: ID của relationship.
        """
        relationship_id = relationship_id or str(uuid.uuid4())
        properties = properties or {}
        
        # Thêm metadata
        properties.update({            
            "id": relationship_id,
            "type": relationship_type,
            "created_at": datetime.now().isoformat()
        })
        
        query = """
        CREATE (r:Relationship $properties)
        RETURN r.id as id
        """
        
        result = self.manager.execute_query(query, {"properties": properties})
        
        if result:
            logger.info(f"Đã tạo relationship: {relationship_type} với ID: {relationship_id}")
            return relationship_id
        else:
            raise RuntimeError(f"Không thể tạo relationship: {relationship_type}")
                   
    
    def create_edge(self, from_entity_id: str, to_entity_id: str, 
                          relationship_type: str, properties: Optional[Dict] = None,
                          relationship_id: Optional[str] = None) -> str:
        """
        Tạo mối quan hệ giữa hai entities
        
        Args:
            from_entity_id: ID của entity nguồn
            to_entity_id: ID của entity đích
            relationship_type: Loại quan hệ (HAS_FEATURE, BELONGS_TO, etc.)
            properties: Thuộc tính bổ sung
            relationship_id: ID tùy chỉnh cho relationship
            
        Returns:
            str: ID của relationship đã tạo
        """
        relationship_id = relationship_id or str(uuid.uuid4())
        properties = properties or {}
        
        # Thêm metadata
        properties.update({
            "id": relationship_id,
            "type": relationship_type,
            "created_at": datetime.now().isoformat()
        })
        
        query = f"""
        MATCH (from {{id: $from_id}})
        MATCH (to {{id: $to_id}})
        CREATE (from)-[r:{relationship_type} $properties]->(to)
        RETURN r.id as id
        """
        
        result = self.manager.execute_query(query, {
            "from_id": from_entity_id,
            "to_id": to_entity_id,
            "properties": properties
        })
        
        if result:
            logger.info(f"Đã tạo relationship: {from_entity_id} -[{relationship_type}]-> {to_entity_id}")
            return relationship_id
        else:
            raise RuntimeError(f"Không thể tạo relationship giữa {from_entity_id} và {to_entity_id}")
    
    def find_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """Tìm entity theo ID"""
        query = "MATCH (e:Entity {id: $entity_id}) RETURN e"
        result = self.manager.execute_query(query, {"entity_id": entity_id})
        
        if result:
            node_data = result[0]["e"]
            return Entity(
                id=node_data["id"],
                name=node_data["name"],
                type=node_data["type"],
                properties=dict(node_data),
                created_at=datetime.fromisoformat(node_data.get("created_at", datetime.now().isoformat()))
            )
        return None
    
    def find_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Tìm tất cả entities theo loại"""
        query = "MATCH (e:Entity {type: $entity_type}) RETURN e"
        result = self.manager.execute_query(query, {"entity_type": entity_type})
        
        entities = []
        for record in result:
            node_data = record["e"]
            entities.append(Entity(
                id=node_data["id"],
                name=node_data["name"],
                type=node_data["type"],
                properties=dict(node_data),
                created_at=datetime.fromisoformat(node_data.get("created_at", datetime.now().isoformat()))
            ))
        
        return entities
    
    def find_entities_by_name_and_type(self, name: str, entity_type: str) -> List[Entity]:
        """Tìm entities theo tên và loại"""
        query = "MATCH (e:Entity {name: $name, type: $entity_type}) RETURN e"
        result = self.manager.execute_query(query, {"name": name, "entity_type": entity_type})
        
        entities = []
        for record in result:
            node_data = record["e"]
            entities.append(Entity(
                id=node_data["id"],
                name=node_data["name"],
                type=node_data["type"],
                properties=dict(node_data),
                created_at=datetime.fromisoformat(node_data.get("created_at", datetime.now().isoformat()))
            ))
        
        return entities

    def find_entities_by_name(self, name: str, exact_match: bool = True) -> List[Entity]:
        """Tìm entities theo tên"""
        if exact_match:
            query = "MATCH (e:Entity {name: $name}) RETURN e"
            params = {"name": name}
        else:
            query = "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($name) RETURN e"
            params = {"name": name}
            
        result = self.manager.execute_query(query, params)
        
        entities = []
        for record in result:
            node_data = record["e"]
            entities.append(Entity(
                id=node_data["id"],
                name=node_data["name"],
                type=node_data["type"],
                properties=dict(node_data),
                created_at=datetime.fromisoformat(node_data.get("created_at", datetime.now().isoformat()))
            ))
        
        return entities
    
    def get_entity_relationships(self, entity_id: str, direction: str = "both") -> List[Tuple[Entity, Relationship, Entity]]:
        """
        Lấy tất cả relationships của một entity
        
        Args:
            entity_id: ID của entity
            direction: "outgoing", "incoming", hoặc "both"
            
        Returns:
            List[Tuple[Entity, Relationship, Entity]]: (from_entity, relationship, to_entity)
        """
        if direction == "outgoing":
            query = """
            MATCH (from:Entity {id: $entity_id})-[r]->(to:Entity)
            RETURN from, r, to
            """
        elif direction == "incoming":
            query = """
            MATCH (from:Entity)-[r]->(to:Entity {id: $entity_id})
            RETURN from, r, to
            """
        else:  # both
            query = """
            MATCH (from:Entity)-[r]-(to:Entity)
            WHERE from.id = $entity_id OR to.id = $entity_id
            RETURN from, r, to
            """
        
        result = self.manager.execute_query(query, {"entity_id": entity_id})
        
        relationships = []
        for record in result:
            from_data = record["from"]
            rel_data = dict(record["r"])
            to_data = record["to"]
            
            from_entity = Entity(
                id=from_data["id"],
                name=from_data["name"],
                type=from_data["type"],
                properties=dict(from_data)
            )
            
            to_entity = Entity(
                id=to_data["id"],
                name=to_data["name"],
                type=to_data["type"],
                properties=dict(to_data)
            )
            
            relationship = Relationship(
                id=rel_data.get("id", ""),
                from_entity_id=from_data["id"],
                to_entity_id=to_data["id"],
                type=rel_data.get("type", ""),
                properties=rel_data
            )
            
            relationships.append((from_entity, relationship, to_entity))
        
        return relationships
    
    def batch_create_entities(self, entities_data: List[Dict]) -> List[str]:
        """
        Tạo hàng loạt entities
        
        Args:
            entities_data: List các dict chứa thông tin entity
            
        Returns:
            List[str]: List các entity IDs đã tạo
        """
        created_ids = []
        
        # Chuẩn bị dữ liệu
        for entity_data in entities_data:
            if "id" not in entity_data:
                entity_data["id"] = str(uuid.uuid4())
            entity_data["created_at"] = datetime.now().isoformat()
        
        query = """
        UNWIND $entities as entity_data
        CREATE (e:Entity)
        SET e = entity_data
        RETURN e.id as id
        """
        
        result = self.manager.execute_query(query, {"entities": entities_data})
        
        for record in result:
            created_ids.append(record["id"])
        
        logger.info(f"Đã tạo {len(created_ids)} entities")
        return created_ids
    
    def batch_create_relationships(self, relationships_data: List[Dict]) -> List[str]:
        """Tạo hàng loạt relationships"""
        created_ids = []
        
        for rel_data in relationships_data:
            if "id" not in rel_data:
                rel_data["id"] = str(uuid.uuid4())
            rel_data["created_at"] = datetime.now().isoformat()
            
            # Tạo từng relationship một (để handle dynamic relationship types)
            rel_id = self.create_relationship(
                from_entity_id=rel_data["from_entity_id"],
                to_entity_id=rel_data["to_entity_id"],
                relationship_type=rel_data["type"],
                properties=rel_data.get("properties", {}),
                relationship_id=rel_data["id"]
            )
            created_ids.append(rel_id)
        
        return created_ids
    
    def delete_entity(self, entity_id: str, cascade: bool = True) -> bool:
        """
        Xóa entity
        
        Args:
            entity_id: ID entity cần xóa
            cascade: Có xóa tất cả relationships không
        """
        if cascade:
            query = "MATCH (e:Entity {id: $entity_id}) DETACH DELETE e"
        else:
            query = "MATCH (e:Entity {id: $entity_id}) DELETE e"
        
        result = self.manager.execute_query(query, {"entity_id": entity_id})
        logger.info(f"Đã xóa entity: {entity_id}")
        return True
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Lấy thống kê tổng quan về Knowledge Graph"""
        query = """
        MATCH (e:Entity)
        OPTIONAL MATCH ()-[r]->()
        RETURN 
            count(DISTINCT e) as total_entities,
            count(r) as total_relationships,
            collect(DISTINCT e.type) as entity_types
        """
        
        result = self.manager.execute_query(query)
        
        if result:
            stats = result[0]
            
            # Lấy thống kê theo loại entity
            type_stats_query = """
            MATCH (e:Entity)
            RETURN e.type as entity_type, count(e) as count
            """
            type_result = self.manager.execute_query(type_stats_query)
            type_stats = {record["entity_type"]: record["count"] for record in type_result}
            
            return {
                "total_entities": stats["total_entities"],
                "total_relationships": stats["total_relationships"],
                "entity_types": stats["entity_types"],
                "entity_type_counts": type_stats
            }
        
        return {}


# Singleton instance
kg_operations = KGOperations() 