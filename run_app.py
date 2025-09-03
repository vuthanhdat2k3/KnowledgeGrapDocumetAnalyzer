#!/usr/bin/env python3
"""
Script chính để chạy Knowledge Graph Document Analyzer
"""
import os
import sys
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import configurations
from config.app_config import app_config
from config.neo4j_config import neo4j_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_environment():
    """Kiểm tra môi trường và dependencies"""
    
    logger.info("🔍 Đang kiểm tra môi trường...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("❌ Yêu cầu Python >= 3.8")
        return False
    
    # Check .env file
    env_file = Path(".env")
    if not env_file.exists():
        logger.warning("⚠️ File .env không tồn tại. Copy từ .env.example")
        try:
            import shutil
            shutil.copy(".env.example", ".env")
            logger.info("✅ Đã tạo .env từ .env.example")
        except:
            logger.error("❌ Không thể tạo file .env")
            return False
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logger.info("✅ Đã load environment variables")
    except ImportError:
        logger.warning("⚠️ python-dotenv chưa được cài đặt")
    
    # Check data directories
    data_dirs = ["data/sample_documents", "data/processed", "data/uploads"]
    for data_dir in data_dirs:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ Môi trường đã sẵn sàng")
    return True


def setup_database():
    """Setup Neo4j database"""
    
    logger.info("🗄️ Đang setup database...")
    
    try:
        # Import và chạy setup script
        from scripts.setup_neo4j import setup_neo4j
        
        if setup_neo4j():
            logger.info("✅ Database setup thành công")
            return True
        else:
            logger.error("❌ Database setup thất bại")
            return False
            
    except Exception as e:
        logger.error(f"❌ Lỗi setup database: {e}")
        return False


def download_sample_data():
    """Download sample documents"""
    
    logger.info("📥 Đang download sample data...")
    
    try:
        from scripts.download_sample_data import download_sample_documents, create_sample_text_files
        
        # Download PDFs từ URLs
        download_sample_documents()
        
        # Tạo sample text files
        create_sample_text_files()
        
        logger.info("✅ Sample data đã sẵn sàng")
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi download sample data: {e}")
        return False


def run_streamlit():
    """Chạy Streamlit application"""
    
    logger.info("🚀 Đang khởi động Streamlit...")
    
    try:
        import streamlit.web.cli as stcli
        import sys
        
        # Đường dẫn đến main app
        main_app_path = "src/ui/main_app.py"
        
        if not os.path.exists(main_app_path):
            logger.error(f"❌ Không tìm thấy {main_app_path}")
            return False
        
        # Configure Streamlit
        sys.argv = [
            "streamlit",
            "run",
            main_app_path,
            "--server.port",
            str(app_config.streamlit_port),
            "--server.address",
            app_config.streamlit_host,
            "--browser.gatherUsageStats",
            "false"
        ]
        
        # Run Streamlit
        stcli.main()
        
    except KeyboardInterrupt:
        logger.info("👋 Đã dừng Streamlit")
    except Exception as e:
        logger.error(f"❌ Lỗi chạy Streamlit: {e}")
        return False


def run_development_server():
    """Chạy development server với auto-reload"""
    
    logger.info("🔧 Chạy development mode...")
    
    # TODO: Implement development server với auto-reload
    # Có thể dùng watchdog để monitor file changes
    
    run_streamlit()


def show_status():
    """Hiển thị trạng thái hệ thống"""
    
    print("\n" + "="*60)
    print(f"📊 {app_config.name} v{app_config.version}")
    print("="*60)
    
    # Check Neo4j
    try:
        from src.core.knowledge_graph.neo4j_manager import neo4j_manager
        
        if neo4j_manager.connect():
            health = neo4j_manager.health_check()
            print(f"🟢 Neo4j: {health['status']}")
            if 'stats' in health:
                stats = health['stats']
                print(f"   📊 Entities: {stats.get('nodes', 0)}")
                print(f"   📊 Relationships: {stats.get('relationships', 0)}")
        else:
            print("🔴 Neo4j: Disconnected")
    except Exception as e:
        print(f"🔴 Neo4j: Error - {e}")
    
    # Check data directories
    data_dirs = [
        ("Sample Documents", "data/sample_documents"),
        ("Processed Data", "data/processed"),
        ("Uploads", "data/uploads")
    ]
    
    for name, path in data_dirs:
        if os.path.exists(path):
            file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            print(f"📁 {name}: {file_count} files")
        else:
            print(f"📁 {name}: Directory not found")
    
    # Check configuration
    print(f"⚙️ Config: {app_config.debug and 'Debug' or 'Production'} mode")
    print(f"🌐 URL: http://{app_config.streamlit_host}:{app_config.streamlit_port}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"{app_config.name} - Knowledge Graph Document Analyzer"
    )
    
    parser.add_argument("--setup", action="store_true", 
                       help="Setup database và download sample data")
    parser.add_argument("--dev", action="store_true",
                       help="Chạy development mode")
    parser.add_argument("--status", action="store_true",
                       help="Hiển thị trạng thái hệ thống")
    parser.add_argument("--no-setup", action="store_true",
                       help="Bỏ qua auto setup")
    
    args = parser.parse_args()
    
    try:
        # Show banner
        print(f"\n🎯 {app_config.name} v{app_config.version}")
        print("Knowledge Graph Document Analyzer Project\n")
        
        if args.status:
            show_status()
            sys.exit(0)
        
        # Check environment
        if not check_environment():
            sys.exit(1)
        
        # Auto setup (unless --no-setup)
        if not args.no_setup and (args.setup or not os.path.exists("data/sample_documents")):
            logger.info("🔧 Chạy auto setup...")
            
            # Setup database
            setup_database()
            
            # Download sample data
            download_sample_data()
        
        # Show status
        show_status()
        
        # Run application
        if args.dev:
            run_development_server()
        else:
            run_streamlit()
    
    except KeyboardInterrupt:
        logger.info("👋 Goodbye!")
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        sys.exit(1) 