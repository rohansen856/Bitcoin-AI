import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Results from LLM evaluation"""
    model_name: str
    provider: str
    test_case: object  # Use TestCase instance
    actual_output: str
    relevancy_score: float
    faithfulness_score: float
    response_time: float
    token_count: int
    error: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        # Serialize to dict for JSON export
        data = asdict(self)
        data['test_case'] = asdict(self.test_case)
        return data