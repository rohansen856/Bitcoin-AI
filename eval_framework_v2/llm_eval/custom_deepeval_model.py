import asyncio
import logging
from deepeval.models.base_model import DeepEvalBaseLLM

logger = logging.getLogger(__name__)

class CustomDeepEvalModel(DeepEvalBaseLLM):
    """Custom DeepEval model wrapper for evaluation"""
    def __init__(self, provider, model_name: str):
        super().__init__(model_name)
        self.provider = provider
        self.model_name = model_name

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            output, _, _ = loop.run_until_complete(
                self.provider.generate(prompt, self.model_name)
            )
            loop.close()
            return output
        except Exception as e:
            logger.error(f"Error in generate: {e}")
            return f"Evaluation error: {e}"

    async def a_generate(self, prompt: str) -> str:
        try:
            output, _, _ = await self.provider.generate(prompt, self.model_name)
            return output
        except Exception as e:
            logger.error(f"Error in a_generate: {e}")
            return f"Evaluation error: {e}"

    def get_model_name(self) -> str:
        return self.model_name