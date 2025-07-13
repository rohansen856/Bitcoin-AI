"""
Data models for the LLM benchmark system.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class Exercise:
    """Represents a coding exercise"""
    slug: str
    name: str
    language: str
    instructions: str
    test_code: str
    starter_code: Optional[str] = None
    difficulty: str = "medium"


@dataclass
class BenchmarkResult:
    """Result of a benchmark run"""
    model: str
    exercise: str
    language: str
    success: bool
    execution_time: float
    token_count: int
    error_message: Optional[str] = None
    generated_code: str = ""
    test_output: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "model": self.model,
            "exercise": self.exercise,
            "language": self.language,
            "success": self.success,
            "execution_time": self.execution_time,
            "token_count": self.token_count,
            "error_message": self.error_message,
            "generated_code": self.generated_code,
            "test_output": self.test_output,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'BenchmarkResult':
        """Create from dictionary"""
        data = data.copy()
        if 'timestamp' in data:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)