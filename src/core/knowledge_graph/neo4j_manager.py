"""
Neo4j Database Manager
Quản lý kết nối và cấu hình Neo4j database
"""
import logging
import time
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

from neo4j import GraphDatabase, Driver, Session, Transaction
from neo4j.exceptions import ServiceUnavailable, TransientError

from config.neo4j_config import neo4j_config


logger = logging.getLogger(__name__)


class Neo4jManager:
    """Quản lý kết nối Neo4j Database"""
    
    def __init__(self):
        self.driver: Optional[Driver] = None
        self.config = neo4j_config
        
    def connect(self) -> bool:
        """
        Thiết lập kết nối với Neo4j database
        
        Returns:
            bool: True nếu kết nối thành công
        """
        try:
            connection_config = self.config.get_connection_config()
            self.driver = GraphDatabase.driver(**connection_config)
            
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
                
            logger.info(f"Đã kết nối thành công với Neo4j tại {self.config.uri}")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi kết nối Neo4j: {e}")
            return False
    
    def disconnect(self):
        """Đóng kết nối Neo4j"""
        if self.driver:
            self.driver.close()
            self.driver = None
            logger.info("Đã đóng kết nối Neo4j")
    
    def is_connected(self) -> bool:
        """Kiểm tra trạng thái kết nối"""
        if not self.driver:
            return False
        
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            return True
        except:
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Kiểm tra health database
        
        Returns:
            Dict: Thông tin trạng thái database
        """
        if not self.is_connected():
            return {
                "status": "disconnected",
                "message": "Không có kết nối với database"
            }
        
        try:
            with self.driver.session() as session:
                # Kiểm tra version
                result = session.run("CALL dbms.components() YIELD name, versions")
                components = result.data()
                
                # Kiểm tra số lượng nodes và relationships
                stats_result = session.run("""
                    MATCH (n) 
                    OPTIONAL MATCH ()-[r]->() 
                    RETURN count(DISTINCT n) as nodes, count(r) as relationships
                """)
                stats = stats_result.single()
                
                return {
                    "status": "connected",
                    "components": components,
                    "stats": {
                        "nodes": stats["nodes"],
                        "relationships": stats["relationships"]
                    },
                    "database": self.config.database
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    @contextmanager
    def get_session(self, database: Optional[str] = None):
        """
        Context manager để lấy Neo4j session
        
        Args:
            database: Tên database (optional)
        """
        if not self.driver:
            raise RuntimeError("Chưa kết nối với Neo4j. Gọi connect() trước.")
        
        session = self.driver.session(database=database or self.config.database)
        try:
            yield session
        finally:
            session.close()
    
    def execute_query(self, query: str, parameters: Optional[Dict] = None, 
                     database: Optional[str] = None, retries: int = None) -> List[Dict]:
        """
        Thực thi Cypher query với retry mechanism
        
        Args:
            query: Cypher query
            parameters: Parameters cho query
            database: Database name
            retries: Số lần retry (default từ config)
            
        Returns:
            List[Dict]: Kết quả query
        """
        if retries is None:
            retries = self.config.max_retry_attempts
            
        parameters = parameters or {}
        
        for attempt in range(retries + 1):
            try:
                with self.get_session(database) as session:
                    result = session.run(query, parameters)
                    return result.data()
                    
            except (ServiceUnavailable, TransientError) as e:
                if attempt < retries:
                    wait_time = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Query failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Query failed after {retries + 1} attempts: {e}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error in query execution: {e}")
                raise
    
    def execute_transaction(self, transaction_func, database: Optional[str] = None):
        """
        Thực thi transaction function
        
        Args:
            transaction_func: Function nhận Transaction object
            database: Database name
        """
        with self.get_session(database) as session:
            return session.execute_write(transaction_func)
    
    def create_indexes(self):
        """Tạo các indexes cần thiết cho Knowledge Graph"""
        indexes = [
            "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (n:Entity) ON (n.name)",
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (n:Entity) ON (n.type)",
            "CREATE INDEX entity_id_idx IF NOT EXISTS FOR (n:Entity) ON (n.id)",
            "CREATE INDEX document_id_idx IF NOT EXISTS FOR (n:Document) ON (n.id)",
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE"
        ]
        
        for index_query in indexes:
            try:
                self.execute_query(index_query)
                logger.info(f"Tạo index thành công: {index_query}")
            except Exception as e:
                logger.warning(f"Không thể tạo index: {e}")
    
    def clear_database(self, confirm: bool = False):
        """
        Xóa tất cả dữ liệu trong database (cẩn thận!)
        
        Args:
            confirm: Phải set True để xác nhận
        """
        if not confirm:
            raise ValueError("Phải set confirm=True để xóa database")
            
        query = "MATCH (n) DETACH DELETE n"
        self.execute_query(query)
        logger.warning("Đã xóa tất cả dữ liệu trong database")


# Singleton instance
neo4j_manager = Neo4jManager() 