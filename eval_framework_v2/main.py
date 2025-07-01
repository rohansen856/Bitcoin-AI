import asyncio
import logging
from llm_eval.config import EvaluationConfig
from llm_eval.framework import LLMEvaluationFramework
from llm_eval.utils import create_sample_test_cases

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    config = EvaluationConfig()
    framework = LLMEvaluationFramework(config)
    test_cases = create_sample_test_cases()
    framework.add_test_cases(test_cases)
    models_to_eval = {'ollama': ['llama3.2', 'llama3.1']}
    try:
        await framework.run_evaluation(models_to_eval)
        framework.print_summary()
        framework.create_visualizations()
        framework.export_results()
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print("Error: ", e)

if __name__ == "__main__":
    asyncio.run(main())