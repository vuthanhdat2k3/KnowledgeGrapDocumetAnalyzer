#!/usr/bin/env python3
"""
Script setup Neo4j database với indexes và constraints cần thiết
"""
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.knowledge_graph.neo4j_manager import neo4j_manager
from config.neo4j_config import neo4j_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_neo4j():
    """Setup Neo4j database với tất cả cấu hình cần thiết"""
    
    logger.info("🚀 Bắt đầu setup Neo4j database...")
    
    # 1. Kết nối với Neo4j
    logger.info("📡 Đang kết nối với Neo4j...")
    if not neo4j_manager.connect():
        logger.error("❌ Không thể kết nối với Neo4j!")
        return False
    
    # 2. Kiểm tra sức khỏe database
    health = neo4j_manager.health_check()
    if health["status"] != "connected":
        logger.error(f"❌ Database không sẵn sàng: {health}")
        return False
    
    logger.info(f"✅ Kết nối thành công với Neo4j: {health}")
    
    # 3. Tạo indexes và constraints
    logger.info("🔧 Đang tạo indexes và constraints...")
    try:
        neo4j_manager.create_indexes()
        logger.info("✅ Đã tạo indexes thành công")
    except Exception as e:
        logger.error(f"❌ Lỗi tạo indexes: {e}")
        return False
    
    # 4. Tạo thêm indexes cho performance
    additional_indexes = [
        # Indexes cho tìm kiếm text
        "CREATE INDEX entity_name_text_idx IF NOT EXISTS FOR (n:Entity) ON (n.name)",
        "CREATE INDEX entity_description_idx IF NOT EXISTS FOR (n:Entity) ON (n.description)",
        
        # Indexes cho document processing
        "CREATE INDEX document_filename_idx IF NOT EXISTS FOR (n:Document) ON (n.filename)",
        "CREATE INDEX document_type_idx IF NOT EXISTS FOR (n:Document) ON (n.file_type)",
        
        # Indexes cho timestamps
        "CREATE INDEX entity_created_idx IF NOT EXISTS FOR (n:Entity) ON (n.created_at)",
        #"CREATE INDEX relationship_created_idx IF NOT EXISTS FOR ()-[r]-() ON (r.created_at)",
        
        # Indexes cho categories
        "CREATE INDEX category_name_idx IF NOT EXISTS FOR (n:Category) ON (n.name)",
        
        # Indexes cho features
        "CREATE INDEX feature_priority_idx IF NOT EXISTS FOR (n:Feature) ON (n.priority)",
    ]
    
    for index_query in additional_indexes:
        try:
            neo4j_manager.execute_query(index_query)
            logger.info(f"✅ Tạo index: {index_query[:50]}...")
        except Exception as e:
            logger.warning(f"⚠️ Không thể tạo index: {e}")
    
    # 5. Tạo sample categories nếu chưa có
    logger.info("📝 Đang tạo sample categories...")
    create_sample_categories()
    
    # 6. Tạo sample viewpoints
    logger.info("💭 Đang tạo sample viewpoints...")
    create_sample_viewpoints()
    
    # 7. Hiển thị thống kê cuối cùng
    final_health = neo4j_manager.health_check()
    logger.info(f"📊 Setup hoàn tất! Stats: {final_health['stats']}")
    
    return True


def create_sample_categories():
    """Tạo các categories mẫu cho product classification"""
    
    categories = [
        {"name": "Mobile Application", "description": "Ứng dụng di động cho iOS/Android"},
        {"name": "Web Application", "description": "Ứng dụng web chạy trên browser"},
        {"name": "Desktop Software", "description": "Phần mềm chạy trên máy tính"},
        {"name": "API/Backend Service", "description": "Dịch vụ backend và API"},
        {"name": "Data Analytics Platform", "description": "Nền tảng phân tích dữ liệu"},
        {"name": "E-commerce Platform", "description": "Nền tảng thương mại điện tử"},
        {"name": "Content Management System", "description": "Hệ thống quản lý nội dung"},
        {"name": "Enterprise Software", "description": "Phần mềm doanh nghiệp"},
        {"name": "IoT Solution", "description": "Giải pháp Internet of Things"},
        {"name": "AI/ML Platform", "description": "Nền tảng AI và Machine Learning"}
    ]
    
    for category in categories:
        query = """
        MERGE (c:Category {name: $name})
        SET c.description = $description,
            c.created_at = datetime()
        """
        try:
            neo4j_manager.execute_query(query, category)
            logger.info(f"✅ Tạo category: {category['name']}")
        except Exception as e:
            logger.warning(f"⚠️ Không thể tạo category {category['name']}: {e}")


def create_sample_viewpoints():
    """Tạo các viewpoints mẫu cho clarification questions"""
    
    viewpoints = [
        {
            "name": "Technical Requirements",
            "description": "Yêu cầu kỹ thuật và công nghệ",
            "questions": [
                "Programming language preferences?",
                "Database requirements?", 
                "Third-party integrations needed?",
                "Performance benchmarks?"
            ]
        },
        {
            "name": "Business Logic",
            "description": "Logic nghiệp vụ và quy trình",
            "questions": [
                "Core business processes?",
                "User roles and permissions?",
                "Business rules and validations?",
                "Workflow requirements?"
            ]
        },
        {
            "name": "User Experience",
            "description": "Trải nghiệm người dùng",
            "questions": [
                "Target user demographics?",
                "UI/UX preferences?",
                "Accessibility requirements?",
                "Mobile responsiveness needed?"
            ]
        },
        {
            "name": "Security Requirements",
            "description": "Yêu cầu bảo mật",
            "questions": [
                "Authentication methods?",
                "Data encryption requirements?",
                "Compliance standards?",
                "Security audit needs?"
            ]
        },
        {
            "name": "Data Requirements",
            "description": "Yêu cầu về dữ liệu",
            "questions": [
                "Data sources and formats?",
                "Data volume expectations?",
                "Backup and recovery needs?",
                "Data migration requirements?"
            ]
        }
    ]
    
    for viewpoint in viewpoints:
        query = """
        MERGE (v:Viewpoint {name: $name})
        SET v.description = $description,
            v.questions = $questions,
            v.created_at = datetime()
        """
        try:
            neo4j_manager.execute_query(query, viewpoint)
            logger.info(f"✅ Tạo viewpoint: {viewpoint['name']}")
        except Exception as e:
            logger.warning(f"⚠️ Không thể tạo viewpoint {viewpoint['name']}: {e}")


def cleanup_database():
    """Xóa tất cả dữ liệu trong database (cẩn thận!)"""
    
    confirm = input("⚠️ BẠN CÓ CHẮC MUỐN XÓA TẤT CẢ DỮ LIỆU? (yes/no): ")
    if confirm.lower() != "yes":
        logger.info("❌ Hủy bỏ cleanup")
        return
    
    logger.warning("🗑️ Đang xóa tất cả dữ liệu...")
    try:
        neo4j_manager.clear_database(confirm=True)
        logger.warning("✅ Đã xóa tất cả dữ liệu")
    except Exception as e:
        logger.error(f"❌ Lỗi cleanup: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Neo4j database")
    parser.add_argument("--cleanup", action="store_true", help="Xóa tất cả dữ liệu")
    parser.add_argument("--setup", action="store_true", default=True, help="Setup database")
    
    args = parser.parse_args()
    
    try:
        if args.cleanup:
            cleanup_database()
        
        if args.setup:
            success = setup_neo4j()
            if success:
                logger.info("🎉 Setup Neo4j hoàn tất thành công!")
                sys.exit(0)
            else:
                logger.error("💥 Setup Neo4j thất bại!")
                sys.exit(1)
                
    except KeyboardInterrupt:
        logger.info("👋 Đã hủy bỏ setup")
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Lỗi không mong đợi: {e}")
        sys.exit(1)
    finally:
        neo4j_manager.disconnect() 