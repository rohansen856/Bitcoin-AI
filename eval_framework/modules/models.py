import requests
from deepeval.models import DeepEvalBaseLLM

class LLaMA31Model(DeepEvalBaseLLM):
    def __init__(self):
        self.api_url = "http://localhost:11434/api/generate"
        self.model_name = "llama3.1"

    def load_model(self):
        return self

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(self.api_url, json=payload)
        response.raise_for_status()
        return response.json()["response"].strip()

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

class LLaMA32Evaluator(DeepEvalBaseLLM):
    def __init__(self):
        self.api_url = "http://localhost:11434/api/generate"
        self.model_name = "llama3.2"

    def load_model(self):
        return self

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(self.api_url, json=payload)
        response.raise_for_status()
        return response.json()["response"].strip()

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

from deepeval.test_case import LLMTestCase

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

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

llama32_evaluator = LLaMA32Evaluator()

answer_relevancy_metric = GEval(
    name="Answer Relevancy",
    criteria="Does the answer directly address the question?",
    model=llama32_evaluator,
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
)

from deepeval import evaluate

evaluate(
    test_cases=test_cases,
    metrics=[answer_relevancy_metric]
)
