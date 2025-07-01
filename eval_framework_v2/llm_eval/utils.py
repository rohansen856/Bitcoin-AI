from typing import List
from .test_case import TestCase

def create_sample_test_cases() -> List[TestCase]:
    return [
        TestCase(input_prompt="Explain quantum computing in simple terms.", category="explanation", difficulty="medium"),
        TestCase(input_prompt="Write a Python function to calculate fibonacci numbers.", category="coding", difficulty="easy"),
        TestCase(input_prompt="Analyze the geopolitical implications of renewable energy adoption.", category="analysis", difficulty="hard"),
        TestCase(input_prompt="What is 15 * 23?", category="math", difficulty="easy"),
        TestCase(input_prompt="Summarize the key points of machine learning in 3 bullet points.", category="summarization", difficulty="easy"),
    ]