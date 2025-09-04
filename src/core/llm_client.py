import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Warning: Genai package not installed. Gemini clients will not be available.")
    
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers package not installed. Local embedding models will not be available.")

from config.ai_config import ai_config

load_dotenv()

class LLMClientFactory:
    def __init__(self, config=None):
        """
        Initialize LLM Client Factory
        
        Args:
            config: AIConfig instance. Nếu None, sẽ sử dụng ai_config global
        """
        self.config = config or ai_config
        self.api_key = self.config.api_key
        self.models = self.config.models

    def get_client(self, model_key: str, **kwargs):
        """
        Tạo client cho model được chỉ định (chỉ hỗ trợ Gemini và embedding local)
        """
        model_name = self.models.get(model_key)
        if not model_name:
            raise ValueError(f"Model key '{model_key}' is not configured.")

        client_params = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens)
        }

        if model_key == "embedding":
            return self._create_embedding_client(model_name, client_params)
        elif model_key.startswith("gemini"):
            return self._create_genai_client(model_name, client_params)
        else:
            raise ValueError(f"Unsupported model key: {model_key}")

    def _create_embedding_client(self, model_name, params):
        """Tạo embedding client (dựa vào loại model)"""
        # Kiểm tra xem model_name có phải là model Sentence Transformers (local) không
        if model_name in ["MiniLM-L6-v2", "all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2"]:
            return self._create_sentence_transformer_client(model_name, params)
        else:
            raise ValueError("Only local sentence-transformers embedding is supported (OpenAI embedding removed)")
            
    def _create_sentence_transformer_client(self, model_name, params):
        """Tạo embedding client sử dụng Sentence Transformers (local)"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers package not installed. Please install with: pip install sentence-transformers")
            
        # Tiền tố "all-" thường được sử dụng trong Hugging Face, nhưng đôi khi không cần
        if model_name == "MiniLM-L6-v2":
            full_model_name = "all-MiniLM-L6-v2"
        else:
            full_model_name = model_name
            
        try:
            # Tải mô hình Sentence Transformer
            client = SentenceTransformer(full_model_name)
            return {
                "model": full_model_name,
                "client": client,
                "default_params": params,
                "type": "embedding_sentence_transformer"
            }
        except Exception as e:
            raise ValueError(f"Failed to load Sentence Transformer model '{full_model_name}': {str(e)}")


    def _create_genai_client(self, model_name, params):
        if not GENAI_AVAILABLE:
            raise ImportError("GenAI package is not installed. Please install with: pip install google-genai")
        
        client = genai.Client(api_key=self.api_key)
        return {
            "model": model_name,
            "client": client,
            "default_params": params,
            "type": "genai"
        }
        
    def chat_completion(self, model_key: str, messages: list, **kwargs):
        """
        Wrapper method để thực hiện chat completion (chỉ hỗ trợ Gemini)
        """
        client_info = self.get_client(model_key, **kwargs)
        client = client_info["client"]
        model_name = client_info["model"]
        params = client_info["default_params"]
        final_params = {**params, **kwargs}
        if client_info["type"] == "genai":
            return client.models.generate_content(
                model=model_name,
                contents=messages,
                config={
                    "temperature": final_params.get("temperature"),
                    "max_output_tokens": final_params.get("max_tokens")
                }
            )
        else:
            raise ValueError(f"Unsupported client type: {client_info['type']}")
            
    def create_embeddings(self, model_key: str, texts: list, **kwargs):
        """
        Tạo embeddings cho danh sách văn bản
        
        Args:
            model_key: Key của model embedding
            texts: Danh sách văn bản cần tạo embedding
            **kwargs: Additional parameters
            
        Returns:
            Danh sách embeddings tương ứng với mỗi văn bản
        """
        client_info = self.get_client(model_key, **kwargs)
        client = client_info["client"]
        model_name = client_info["model"]
        
        if client_info["type"] == "embedding_sentence_transformer":
            # Sentence Transformers embedding
            return client.encode(texts).tolist()
        else:
            raise ValueError(f"Unsupported embedding client type: {client_info['type']}")


# Usage example:
if __name__ == "__main__":
    try:
        factory = LLMClientFactory()
        gemini_client = factory.get_client("gemini-2.5-flash")
        print("Get gemini client successful")
        messages = "Hãy cho biết bây giờ là mấy giờ?"
        response = factory.chat_completion(model_key="gemini-2.5-flash", messages=messages)
        print(response.text)
            
    except Exception as e:
        print(f"Factory initialization error: {e}")
        print("Please check your environment variables (API_KEY, BASE_URL, MODEL_NAME__*)")