"""
Cấu hình AI/ML models và APIs
"""
import os
from typing import Optional, List
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class AIConfig(BaseModel):
    """Cấu hình AI models và APIs"""
    
    # LLM Client Configuration
    api_key: Optional[str] = os.getenv("API_KEY")
    genai_api_key: Optional[str] = os.getenv("GENAI_API_KEY")
    base_url: Optional[str] = os.getenv("BASE_URL")
    
    # Model Names
    model_gpt_41: Optional[str] = os.getenv("MODEL_NAME__GPT_41")
    model_gpt_41_nano: Optional[str] = os.getenv("MODEL_NAME__GPT_41_NANO")
    model_claude_35: Optional[str] = os.getenv("MODEL_NAME__CLAUDE_35")
    
    model_embedding: Optional[str] = os.getenv("MODEL_NAME_EMBEDDING")
    
    model_gemini_25_flash: Optional[str] = os.getenv("MODEL_NAME_GEMINI")
    
    # Default Parameters
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "2000"))
    
    
    @property
    def models(self) -> dict:
        """Trả về dictionary các model được config"""
        return {
            # OpenAI & Claude models
            "gpt-4.1": self.model_gpt_41,
            "gpt-4.1-nano": self.model_gpt_41_nano,
            "claude-3-5": self.model_claude_35,
            "embedding": self.model_embedding,
            "gemini-2.5-flash": self.model_gemini_25_flash
        }
    
    # Knowledge Graph
    kg_extraction_prompt_template: str = """
    Phân tích văn bản sau và trích xuất các thực thể và mối quan hệ để xây dựng Knowledge Graph:
    
    Văn bản: {text}
    
    Hãy trả về:
    1. Entities (các thực thể): tên, loại, thuộc tính
    2. Relationships (mối quan hệ): từ thực thể nào đến thực thể nào, loại quan hệ
    
    Format JSON:
    {{
        "entities": [
            {{"name": "...", "type": "...", "properties": {{...}}}}
        ],
        "relationships": [
            {{"from": "...", "to": "...", "type": "...", "properties": {{...}}}}
        ]
    }}
    """
    
    # Category classification
    product_categories: List[str] = [
        "Mobile Application",
        "Web Application", 
        "Desktop Software",
        "API/Backend Service",
        "Data Analytics Platform",
        "E-commerce Platform",
        "Content Management System",
        "Enterprise Software",
        "IoT Solution",
        "AI/ML Platform"
    ]
    
    # Viewpoints for clarification questions
    clarification_viewpoints: List[str] = [
        "Technical Requirements",
        "Business Logic",
        "User Experience",
        "Performance Requirements", 
        "Security Requirements",
        "Integration Requirements",
        "Data Requirements",
        "Deployment Requirements"
    ]


# Singleton instance
ai_config = AIConfig() 