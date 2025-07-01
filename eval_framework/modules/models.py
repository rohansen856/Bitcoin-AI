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

llama31 = LLaMA31Model()
llama32_evaluator = LLaMA32Evaluator()
