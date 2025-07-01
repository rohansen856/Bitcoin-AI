from dataclasses import dataclass
from typing import Optional

@dataclass
class TestCase:
    """Individual test case structure"""
    input_prompt: str
    expected_output: Optional[str] = None
    context: Optional[str] = None
    category: str = "general"
    difficulty: str = "medium"