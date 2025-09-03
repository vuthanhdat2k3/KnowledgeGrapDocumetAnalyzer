"""
Cấu hình ứng dụng chính
"""
import os
from typing import List
from pydantic import BaseModel


class AppConfig(BaseModel):
    """Cấu hình ứng dụng"""
    
    # App info
    name: str = os.getenv("APP_NAME", "Knowledge Graph Document Analyzer")
    version: str = os.getenv("APP_VERSION", "1.0.0")
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # File upload
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
    allowed_extensions: List[str] = os.getenv("ALLOWED_EXTENSIONS", "pdf,docx,xlsx,txt").split(",")
    upload_dir: str = "data/uploads"
    processed_dir: str = "data/processed"
    
    # Streamlit
    streamlit_port: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    streamlit_host: str = os.getenv("STREAMLIT_HOST", "0.0.0.0")
    
    # Paths
    sample_documents_path: str = "data/sample_documents"
    categories_path: str = "data/categories"
    viewpoints_path: str = "data/viewpoints"
    
    # UI Theme
    primary_color: str = os.getenv("STREAMLIT_THEME_PRIMARY_COLOR", "#FF6B6B")
    background_color: str = os.getenv("STREAMLIT_THEME_BACKGROUND_COLOR", "#FFFFFF")
    
    def get_file_size_bytes(self) -> int:
        """Trả về kích thước file tối đa tính bằng bytes"""
        return self.max_file_size_mb * 1024 * 1024
    
    def is_allowed_file(self, filename: str) -> bool:
        """Kiểm tra file có được phép upload không"""
        if not filename:
            return False
        extension = filename.rsplit('.', 1)[-1].lower()
        return extension in self.allowed_extensions


# Singleton instance
app_config = AppConfig() 