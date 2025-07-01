from dataclasses import dataclass
import os
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

@dataclass
class EvaluationConfig:
    """Configuration for LLM evaluation"""
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY", None)
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY", None)
    evaluator_model: str = os.getenv("EVALUATOR_MODEL", "llama3.2")  # Model used for evaluation
    output_dir: str = os.getenv("OUTPUT_DIR", "evaluation_results") # Directory to save evaluation results
    timeout: int = int(os.getenv("EVALUATION_TIMEOUT", 30))  # Default timeout for LLM calls in seconds