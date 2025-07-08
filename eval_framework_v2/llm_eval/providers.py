import time
import logging
import asyncio
import requests

logger = logging.getLogger(__name__)

from typing import Tuple
from .config import EvaluationConfig

class ModelProvider:
    """Base class for model providers"""
    def __init__(self, config: EvaluationConfig):
        self.config = config

    async def generate(self, prompt: str, model: str) -> Tuple[str, float, int]:
        raise NotImplementedError

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

import anthropic

class ClaudeProvider(ModelProvider):
    """Anthropic Claude provider"""
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key) if config.anthropic_api_key else None

    async def generate(self, prompt: str, model: str) -> Tuple[str, float, int]:
        if not self.client:
            return "Error: Anthropic API key not provided", 0.0, 0
        start_time = time.time()
        try:
            resp = self.client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            elapsed = time.time() - start_time
            output = resp.content[0].text if resp.content else ""
            tokens = resp.usage.output_tokens if resp.usage else len(output.split())
            return output, elapsed, tokens
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return f"Error: {e}", time.time() - start_time, 0

import openai

class OpenAIProvider(ModelProvider):
    """OpenAI model provider"""
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        if config.openai_api_key:
            openai.api_key = config.openai_api_key
            self.client = openai.OpenAI(api_key=config.openai_api_key)
        else:
            self.client = None

    async def generate(self, prompt: str, model: str) -> Tuple[str, float, int]:
        if not self.client:
            return "Error: OpenAI API key not provided", 0.0, 0
        start_time = time.time()
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            elapsed = time.time() - start_time
            output = resp.choices[0].message.content if resp.choices else ""
            tokens = resp.usage.completion_tokens if resp.usage else len(output.split())
            return output, elapsed, tokens
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"Error: {e}", time.time() - start_time, 0



class CustomLLMProvider(ModelProvider):
    """Custom LLM endpoint provider for french.braidpool.net"""
    
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        self.base_url = config.custom_llm_base_url
        self.api_key = config.custom_llm_api_key
        self.session = requests.Session()
        # Session setup with SSL verification disabled for this endpoint
        self.session.verify = False
        
    async def generate(self, prompt: str, model: str) -> Tuple[str, float, int]:
        start_time = time.time()
        try:
            payload = {
                "prompt": prompt,
                "n_predict": 128
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Make the async request (using requests in a thread pool for async compatibility)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.session.post(
                    f"{self.base_url}/completion",
                    json=payload,
                    headers=headers,
                    timeout=self.config.timeout
                )
            )
            
            response.raise_for_status()
            data = response.json()
            
            elapsed = time.time() - start_time
            
            output = data.get("content", "")
            
            token_count = data.get("tokens_predicted", len(output.split()))
            
            return output, elapsed, token_count
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Custom LLM API request error: {e}")
            return f"Error: {e}", time.time() - start_time, 0
        except Exception as e:
            logger.error(f"Custom LLM API error: {e}")
            return f"Error: {e}", time.time() - start_time, 0