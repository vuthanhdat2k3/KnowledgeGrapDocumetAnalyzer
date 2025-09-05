"""
Cấu hình Neo4j Database
"""
import os
from typing import Optional
from pydantic import BaseModel


class Neo4jConfig(BaseModel):
    """Cấu hình kết nối Neo4j"""
    
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7688")
    username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "password123")
    database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    
    # Connection pool settings
    max_connection_lifetime: int = 3600  # seconds
    max_connection_pool_size: int = 50
    connection_acquisition_timeout: int = 60  # seconds
    
    # Retry settings
    max_retry_attempts: int = 3
    retry_delay: float = 1.0  # seconds
    
    def get_connection_config(self) -> dict:
        """Trả về cấu hình kết nối dưới dạng dictionary"""
        return {
            "uri": self.uri,
            "auth": (self.username, self.password),
            "database": self.database,
            "max_connection_lifetime": self.max_connection_lifetime,
            "max_connection_pool_size": self.max_connection_pool_size,
            "connection_acquisition_timeout": self.connection_acquisition_timeout,
        }


# Singleton instance
neo4j_config = Neo4jConfig() 