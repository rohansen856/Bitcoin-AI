from deepeval.test_case import LLMTestCase
from .models import LLaMA31Model, LLaMA32Evaluator

llama31 = LLaMA31Model()

test_cases = [
    LLMTestCase(
        input="What is the capital of France?",
        expected_output="Paris",
        actual_output=llama31.generate("What is the capital of France?")
    ),
    LLMTestCase(
        input="Who wrote 'Pride and Prejudice'?",
        expected_output="Jane Austen",
        actual_output=llama31.generate("Who wrote 'Pride and Prejudice'?")
    ),
]