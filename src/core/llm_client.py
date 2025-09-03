import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from openai import OpenAI, AsyncOpenAI
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI package not installed. OpenAI clients will not be available.")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: Anthropic package not installed. Claude clients will not be available.")

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
        self.base_url = self.config.base_url
        self.models = self.config.models

    def get_client(self, model_key: str, **kwargs):
        """
        Tạo client cho model được chỉ định
        
        Args:
            model_key: Key của model ("gpt-4.1", "claude-3-5", etc.)
            **kwargs: Additional parameters cho client (temperature, max_tokens, etc.)
            
        Returns:
            Dict chứa client và thông tin model
        """
        model_name = self.models.get(model_key)
        if not model_name:
            raise ValueError(f"Model key '{model_key}' is not configured.")

        # Merge default config với kwargs
        client_params = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens)
        }

        if model_key.startswith("gpt"):
            return self._create_openai_client(model_name, client_params)
        elif model_key.startswith("claude"):
            return self._create_anthropic_client(model_name, client_params)
        elif model_key == "embedding":
            return self._create_embedding_client(model_name, client_params)
        else:
            raise ValueError(f"Unsupported model key: {model_key}")

    def _create_openai_client(self, model_name, params):
        """Tạo OpenAI client"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package is not installed. Please install with: pip install openai")
            
        if not self.base_url:
            raise ValueError("BASE_URL is required for OpenAI client")
            
        client = OpenAI(api_key=self.api_key, base_url=self.base_url + "/v1")
        return {
            "model": model_name,
            "client": client,
            "default_params": params,
            "type": "openai"
        }

    def _create_embedding_client(self, model_name, params):
        """Tạo embedding client (sử dụng OpenAI API)"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package is not installed. Please install with: pip install openai")
            
        if not self.base_url:
            raise ValueError("BASE_URL is required for embedding client")
            
        client = OpenAI(api_key=self.api_key, base_url=self.base_url + "/v1")
        return {
            "model": model_name,
            "client": client,
            "default_params": params,
            "type": "embedding"
        }

    def _create_anthropic_client(self, model_name, params):
        """Tạo Anthropic client"""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package is not installed. Please install with: pip install anthropic")
            
        if not self.base_url:
            raise ValueError("BASE_URL is required for Anthropic client")
            
        client = anthropic.Anthropic(api_key=self.api_key, base_url=self.base_url)
        return {
            "model": model_name,
            "client": client,
            "default_params": params,
            "type": "anthropic"
        }


    def chat_completion(self, model_key: str, messages: list, **kwargs):
        """
        Wrapper method để thực hiện chat completion
        
        Args:
            model_key: Key của model
            messages: List messages cho chat
            **kwargs: Additional parameters
            
        Returns:
            Response từ model
        """
        client_info = self.get_client(model_key, **kwargs)
        client = client_info["client"]
        model_name = client_info["model"]
        params = client_info["default_params"]
        
        # Merge params với kwargs
        final_params = {**params, **kwargs}
        
        if client_info["type"] == "openai":
            return client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=final_params.get("temperature"),
                max_tokens=final_params.get("max_tokens")
            )
        elif client_info["type"] == "anthropic":
            return client.messages.create(
                model=model_name,
                messages=messages,
                max_tokens=final_params.get("max_tokens", 1024)
            )
        else:
            raise ValueError(f"Unsupported client type: {client_info['type']}")


# Usage example:
if __name__ == "__main__":
    try:
        factory = LLMClientFactory()
        
        # Lấy clients
        gpt_client = factory.get_client("gpt-4.1")
        gpt_nano_client = factory.get_client("gpt-4.1-nano")
        claude_client = factory.get_client("claude-3-5")
        embedding_client = factory.get_client("embedding")

        # Example usage với chat_completion wrapper
        print("\n=== Chat Completion Examples ===")
        
        # GPT example
        try:
            response = factory.chat_completion(
                "gpt-4.1", 
                [{"role": "user", "content": "What is the capital of France?"}],
                temperature=0.5
            )
            print("GPT answer:", response.choices[0].message.content)
        except Exception as e:
            print(f"GPT error: {e}")
        
        # Claude example
        try:
            response = factory.chat_completion(
                "claude-3-5",
                [{"role": "user", "content": "What is the capital of France?"}]
            )
            print("Claude answer:", response.content[0].text)
        except Exception as e:
            print(f"Claude error: {e}")
            
    except Exception as e:
        print(f"Factory initialization error: {e}")
        print("Please check your environment variables (API_KEY, BASE_URL, MODEL_NAME__*)")