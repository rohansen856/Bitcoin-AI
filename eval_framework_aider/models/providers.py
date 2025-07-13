"""
Model providers for different LLM APIs.
"""

import time
import requests
from typing import Tuple
from abc import ABC, abstractmethod

from core.config import EvaluationConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelProvider(ABC):
    """Base class for model providers"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config

    @abstractmethod
    async def generate(self, prompt: str, model: str) -> Tuple[str, float, int]:
        """Generate response from model"""
        pass


class OllamaProvider(ModelProvider):
    """Ollama model provider"""
    
    async def generate(self, prompt: str, model: str) -> Tuple[str, float, int]:
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.config.ollama_base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=self.config.timeout
            )
            response.raise_for_status()
            data = response.json()
            elapsed = time.time() - start_time
            output = data.get("response", "")
            token_count = len(output.split())
            return output, elapsed, token_count
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return f"Error: {e}", time.time() - start_time, 0


class OpenAIProvider(ModelProvider):
    """OpenAI model provider"""
    
    async def generate(self, prompt: str, model: str) -> Tuple[str, float, int]:
        start_time = time.time()
        try:
            # This would require the OpenAI client implementation
            # For now, returning a placeholder
            logger.warning("OpenAI provider not fully implemented")
            return "OpenAI implementation needed", time.time() - start_time, 0
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"Error: {e}", time.time() - start_time, 0


class AnthropicProvider(ModelProvider):
    """Anthropic model provider"""
    
    async def generate(self, prompt: str, model: str) -> Tuple[str, float, int]:
        start_time = time.time()
        try:
            # This would require the Anthropic client implementation
            # For now, returning a placeholder
            logger.warning("Anthropic provider not fully implemented")
            return "Anthropic implementation needed", time.time() - start_time, 0
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return f"Error: {e}", time.time() - start_time, 0


class CustomLLMProvider(ModelProvider):
    """Custom LLM provider for generic REST APIs"""
    
    async def generate(self, prompt: str, model: str) -> Tuple[str, float, int]:
        start_time = time.time()
        try:
            if not self.config.custom_llm_base_url:
                raise ValueError("Custom LLM base URL not configured")
            
            headers = {}
            if self.config.custom_llm_api_key:
                headers["Authorization"] = f"Bearer {self.config.custom_llm_api_key}"
            
            response = requests.post(
                f"{self.config.custom_llm_base_url}/generate",
                json={"model": model, "prompt": prompt},
                headers=headers,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            data = response.json()
            elapsed = time.time() - start_time
            output = data.get("response", "")
            token_count = len(output.split())
            return output, elapsed, token_count
        except Exception as e:
            logger.error(f"Custom LLM API error: {e}")
            return f"Error: {e}", time.time() - start_time, 0


def get_provider(provider_name: str, config: EvaluationConfig) -> ModelProvider:
    """Factory function to get model provider"""
    providers = {
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "custom": CustomLLMProvider
    }
    
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}")
    
    return providers[provider_name](config)