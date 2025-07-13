"""
Configuration classes for the LLM benchmark system.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    ollama_base_url: str = "http://localhost:11434"
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    custom_llm_base_url: Optional[str] = None
    custom_llm_api_key: Optional[str] = None
    timeout: int = 120
    max_workers: int = 4
    cache_dir: Path = Path("./cache")
    results_dir: Path = Path("./results")
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        self.cache_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)