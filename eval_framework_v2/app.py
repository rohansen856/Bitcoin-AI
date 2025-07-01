"""
LLM Evaluation Framework
Supports Ollama, Claude (Anthropic), and OpenAI models
Provides automated evaluation with metrics and visualizations
"""

import json
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Core dependencies
import requests
import openai
import anthropic
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """Configuration for LLM evaluation"""
    ollama_base_url: str = "http://localhost:11434"
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    evaluator_model: str = "gpt-4"  # Model used for evaluation
    output_dir: str = "evaluation_results"
    timeout: int = 30

@dataclass
class TestCase:
    """Individual test case structure"""
    input_prompt: str
    expected_output: Optional[str] = None
    context: Optional[str] = None
    category: str = "general"
    difficulty: str = "medium"

@dataclass
class EvaluationResult:
    """Results from LLM evaluation"""
    model_name: str
    provider: str
    test_case: TestCase
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

class ModelProvider:
    """Base class for model providers"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    async def generate(self, prompt: str, model: str) -> Tuple[str, float, int]:
        """Generate response from model. Returns (response, time_taken, token_count)"""
        raise NotImplementedError

class OllamaProvider(ModelProvider):
    """Ollama model provider"""
    
    async def generate(self, prompt: str, model: str) -> Tuple[str, float, int]:
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.config.ollama_base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=self.config.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            elapsed_time = time.time() - start_time
            output = data.get("response", "")
            token_count = len(output.split())  # Approximate token count
            
            return output, elapsed_time, token_count
            
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return f"Error: {str(e)}", time.time() - start_time, 0

class ClaudeProvider(ModelProvider):
    """Anthropic Claude provider"""
    
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        if config.anthropic_api_key:
            self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        else:
            self.client = None
    
    async def generate(self, prompt: str, model: str) -> Tuple[str, float, int]:
        if not self.client:
            return "Error: Anthropic API key not provided", 0.0, 0
        
        start_time = time.time()
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            elapsed_time = time.time() - start_time
            output = response.content[0].text if response.content else ""
            token_count = response.usage.output_tokens if response.usage else len(output.split())
            
            return output, elapsed_time, token_count
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return f"Error: {str(e)}", time.time() - start_time, 0

class OpenAIProvider(ModelProvider):
    """OpenAI model provider"""
    
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        if config.openai_api_key:
            openai.api_key = config.openai_api_key
            self.client = openai.OpenAI(api_key=config.openai_api_key)
        else:
            self.client = None
    
    async def generate(self, prompt: str, model: str) -> Tuple[str, float, int]:
        if not self.client:
            return "Error: OpenAI API key not provided", 0.0, 0
        
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            
            elapsed_time = time.time() - start_time
            output = response.choices[0].message.content if response.choices else ""
            token_count = response.usage.completion_tokens if response.usage else len(output.split())
            
            return output, elapsed_time, token_count
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"Error: {str(e)}", time.time() - start_time, 0

class CustomDeepEvalModel(DeepEvalBaseLLM):
    """Custom DeepEval model wrapper for evaluation"""
    
    def __init__(self, provider: ModelProvider, model_name: str):
        super().__init__(model_name)
        self.provider = provider
        self.model_name = model_name
    
    def load_model(self):
        """Load model - required by DeepEval"""
        return self.model_name
    
    def generate(self, prompt: str) -> str:
        """Synchronous generate method required by DeepEval"""
        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, create a new thread for sync operation
                import concurrent.futures
                import threading
                
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(
                            self.provider.generate(prompt, self.model_name)
                        )
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    output, _, _ = future.result(timeout=60)
                    return output
                    
            except RuntimeError:
                # No event loop running, safe to create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    output, _, _ = loop.run_until_complete(
                        self.provider.generate(prompt, self.model_name)
                    )
                    return output
                finally:
                    loop.close()
                    
        except Exception as e:
            logger.error(f"Error in CustomDeepEvalModel.generate: {e}")
            return f"Evaluation error: {str(e)}"
    
    async def a_generate(self, prompt: str) -> str:
        """Async generate method"""
        try:
            output, _, _ = await self.provider.generate(prompt, self.model_name)
            return output
        except Exception as e:
            logger.error(f"Error in CustomDeepEvalModel.a_generate: {e}")
            return f"Evaluation error: {str(e)}"
    
    def get_model_name(self) -> str:
        return self.model_name

class LLMEvaluationFramework:
    """Main evaluation framework"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.providers = {
            'ollama': OllamaProvider(config),
            'claude': ClaudeProvider(config),
            'openai': OpenAIProvider(config)
        }
        self.results: List[EvaluationResult] = []
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def add_test_cases(self, test_cases: List[TestCase]):
        """Add test cases for evaluation"""
        self.test_cases = test_cases
    
    async def evaluate_model(self, provider_name: str, model_name: str) -> List[EvaluationResult]:
        """Evaluate a specific model"""
        provider = self.providers[provider_name]
        results = []
        
        logger.info(f"Evaluating {provider_name}/{model_name}")
        
        # Simple fallback evaluation if DeepEval fails
        def simple_relevancy_score(prompt: str, response: str) -> float:
            """Simple relevancy scoring based on keyword matching and length"""
            if not response or response.startswith("Error:"):
                return 0.0
            
            # Basic scoring: length, keyword overlap, coherence indicators
            prompt_words = set(prompt.lower().split())
            response_words = set(response.lower().split())
            
            # Keyword overlap score
            overlap = len(prompt_words & response_words) / len(prompt_words) if prompt_words else 0
            
            # Length appropriateness (not too short, not excessively long)
            length_score = min(len(response.split()) / 50, 1.0)  # Optimal around 50 words
            
            # Basic coherence (sentences, punctuation)
            coherence_score = 0.8 if any(c in response for c in '.!?') else 0.3
            
            return min((overlap * 0.4 + length_score * 0.3 + coherence_score * 0.3), 1.0)
        
        for i, test_case in enumerate(self.test_cases):
            try:
                logger.info(f"Processing test case {i+1}/{len(self.test_cases)}: {test_case.category}")
                
                # Generate response
                output, response_time, token_count = await provider.generate(
                    test_case.input_prompt, model_name
                )
                
                # Initialize scores
                relevancy_score = 0.0
                faithfulness_score = 0.0
                
                if not output.startswith("Error:"):
                    try:
                        # Try DeepEval evaluation first
                        evaluator_provider = None
                        evaluator_model_name = self.config.evaluator_model
                        
                        # Determine evaluator provider
                        if evaluator_model_name.startswith('gpt') and self.config.openai_api_key:
                            evaluator_provider = self.providers['openai']
                        elif evaluator_model_name.startswith('claude') and self.config.anthropic_api_key:
                            evaluator_provider = self.providers['claude']
                        elif 'ollama' in self.providers:
                            evaluator_provider = self.providers['ollama']
                            evaluator_model_name = 'llama3.2'  # Default Ollama model
                        
                        if evaluator_provider:
                            evaluator_model = CustomDeepEvalModel(evaluator_provider, evaluator_model_name)
                            
                            # Create DeepEval test case
                            eval_test_case = LLMTestCase(
                                input=test_case.input_prompt,
                                actual_output=output,
                                expected_output=test_case.expected_output,
                                context=[test_case.context] if test_case.context else None
                            )
                            
                            # Run relevancy evaluation
                            try:
                                relevancy_metric = AnswerRelevancyMetric(model=evaluator_model, threshold=0.5)
                                relevancy_metric.measure(eval_test_case)
                                relevancy_score = relevancy_metric.score
                                logger.info(f"DeepEval relevancy score: {relevancy_score}")
                            except Exception as relevancy_error:
                                logger.warning(f"DeepEval relevancy failed, using simple scoring: {relevancy_error}")
                                relevancy_score = simple_relevancy_score(test_case.input_prompt, output)
                            
                            # Run faithfulness evaluation if context provided
                            if test_case.context:
                                try:
                                    faithfulness_metric = FaithfulnessMetric(model=evaluator_model, threshold=0.5)
                                    faithfulness_metric.measure(eval_test_case)
                                    faithfulness_score = faithfulness_metric.score
                                    logger.info(f"DeepEval faithfulness score: {faithfulness_score}")
                                except Exception as faithfulness_error:
                                    logger.warning(f"DeepEval faithfulness failed: {faithfulness_error}")
                                    faithfulness_score = 0.5  # Default neutral score
                        else:
                            # Fallback to simple evaluation
                            logger.warning("No evaluator model available, using simple scoring")
                            relevancy_score = simple_relevancy_score(test_case.input_prompt, output)
                            faithfulness_score = 0.5 if test_case.context else 0.0
                            
                    except Exception as eval_error:
                        logger.warning(f"Evaluation failed, using simple scoring: {eval_error}")
                        relevancy_score = simple_relevancy_score(test_case.input_prompt, output)
                        faithfulness_score = 0.5 if test_case.context else 0.0
                
                result = EvaluationResult(
                    model_name=f"{provider_name}/{model_name}",
                    provider=provider_name,
                    test_case=test_case,
                    actual_output=output,
                    relevancy_score=relevancy_score,
                    faithfulness_score=faithfulness_score,
                    response_time=response_time,
                    token_count=token_count,
                    error=None if not output.startswith("Error:") else output
                )
                
                results.append(result)
                self.results.append(result)
                
                logger.info(f"Completed test case {i+1}: relevancy={relevancy_score:.3f}, time={response_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error evaluating test case {i+1}: {e}")
                error_result = EvaluationResult(
                    model_name=f"{provider_name}/{model_name}",
                    provider=provider_name,
                    test_case=test_case,
                    actual_output="",
                    relevancy_score=0.0,
                    faithfulness_score=0.0,
                    response_time=0.0,
                    token_count=0,
                    error=str(e)
                )
                results.append(error_result)
                self.results.append(error_result)
        
        return results
    
    async def run_evaluation(self, models: Dict[str, List[str]]):
        """Run evaluation on multiple models
        Args:
            models: Dict mapping provider names to list of model names
                   e.g., {'ollama': ['llama2', 'mistral'], 'openai': ['gpt-3.5-turbo']}
        """
        logger.info("Starting LLM evaluation framework")
        
        for provider_name, model_list in models.items():
            if provider_name not in self.providers:
                logger.warning(f"Unknown provider: {provider_name}")
                continue
            
            for model_name in model_list:
                await self.evaluate_model(provider_name, model_name)
        
        logger.info("Evaluation completed")
    
    def generate_metrics_summary(self) -> pd.DataFrame:
        """Generate summary metrics DataFrame"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                'Model': result.model_name,
                'Provider': result.provider,
                'Category': result.test_case.category,
                'Difficulty': result.test_case.difficulty,
                'Relevancy Score': result.relevancy_score,
                'Faithfulness Score': result.faithfulness_score,
                'Response Time (s)': result.response_time,
                'Token Count': result.token_count,
                'Has Error': result.error is not None
            })
        
        return pd.DataFrame(data)
    
    def create_visualizations(self):
        """Create stacked graphs and visualizations"""
        if not self.results:
            logger.warning("No results to visualize")
            return
        
        df = self.generate_metrics_summary()
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('LLM Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Average Relevancy Score by Model
        relevancy_avg = df.groupby('Model')['Relevancy Score'].mean().sort_values(ascending=True)
        axes[0, 0].barh(range(len(relevancy_avg)), relevancy_avg.values, 
                       color=plt.cm.viridis(np.linspace(0, 1, len(relevancy_avg))))
        axes[0, 0].set_yticks(range(len(relevancy_avg)))
        axes[0, 0].set_yticklabels(relevancy_avg.index, fontsize=10)
        axes[0, 0].set_xlabel('Average Relevancy Score')
        axes[0, 0].set_title('Model Performance: Relevancy')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # 2. Response Time Distribution
        df_no_errors = df[~df['Has Error']]
        if not df_no_errors.empty:
            for i, model in enumerate(df_no_errors['Model'].unique()):
                model_data = df_no_errors[df_no_errors['Model'] == model]['Response Time (s)']
                axes[0, 1].hist(model_data, alpha=0.6, label=model, bins=10)
            axes[0, 1].set_xlabel('Response Time (seconds)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Response Time Distribution')
            axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Stacked Performance by Category
        category_pivot = df.pivot_table(
            values=['Relevancy Score', 'Faithfulness Score'], 
            index='Model', 
            columns='Category', 
            aggfunc='mean',
            fill_value=0
        )
        
        if not category_pivot.empty:
            category_pivot['Relevancy Score'].plot(kind='bar', stacked=True, ax=axes[1, 0], 
                                                  colormap='Set3', alpha=0.8)
            axes[1, 0].set_title('Relevancy Scores by Category')
            axes[1, 0].set_xlabel('Model')
            axes[1, 0].set_ylabel('Relevancy Score')
            axes[1, 0].legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Error Rate and Token Efficiency
        error_rate = df.groupby('Model')['Has Error'].mean()
        token_avg = df_no_errors.groupby('Model')['Token Count'].mean() if not df_no_errors.empty else pd.Series()
        
        ax_twin = axes[1, 1].twinx()
        
        if not error_rate.empty:
            bars1 = axes[1, 1].bar(range(len(error_rate)), error_rate.values, 
                                  alpha=0.7, color='red', label='Error Rate')
            axes[1, 1].set_ylabel('Error Rate', color='red')
            axes[1, 1].set_ylim(0, max(1, error_rate.max() * 1.1))
        
        if not token_avg.empty:
            bars2 = ax_twin.bar([x + 0.4 for x in range(len(token_avg))], token_avg.values, 
                               alpha=0.7, color='blue', width=0.4, label='Avg Tokens')
            ax_twin.set_ylabel('Average Token Count', color='blue')
        
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_title('Error Rate vs Token Count')
        axes[1, 1].set_xticks(range(len(error_rate)))
        axes[1, 1].set_xticklabels(error_rate.index, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = Path(self.config.output_dir) / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to: {output_path}")
        
        plt.show()
    
    def export_results(self):
        """Export results to various formats"""
        if not self.results:
            logger.warning("No results to export")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export detailed results as JSON
        json_path = Path(self.config.output_dir) / f"detailed_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump([asdict(result) for result in self.results], f, indent=2)
        
        # Export summary as CSV
        df = self.generate_metrics_summary()
        csv_path = Path(self.config.output_dir) / f"summary_metrics_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Results exported to: {json_path} and {csv_path}")
    
    def print_summary(self):
        """Print concise evaluation summary"""
        if not self.results:
            print("No evaluation results available.")
            return
        
        df = self.generate_metrics_summary()
        
        print("\n" + "="*60)
        print("LLM EVALUATION SUMMARY")
        print("="*60)
        
        # Overall metrics
        print(f"Total Test Cases: {len(self.results)}")
        print(f"Models Evaluated: {df['Model'].nunique()}")
        print(f"Categories Tested: {df['Category'].nunique()}")
        
        # Top performers
        print(f"\n🏆 TOP PERFORMERS:")
        top_relevancy = df.groupby('Model')['Relevancy Score'].mean().nlargest(3)
        for i, (model, score) in enumerate(top_relevancy.items(), 1):
            print(f"  {i}. {model}: {score:.3f} relevancy")
        
        # Speed champions
        speed_df = df[~df['Has Error']]
        if not speed_df.empty:
            print(f"\n⚡ FASTEST MODELS:")
            fastest = speed_df.groupby('Model')['Response Time (s)'].mean().nsmallest(3)
            for i, (model, time) in enumerate(fastest.items(), 1):
                print(f"  {i}. {model}: {time:.2f}s avg")
        
        # Error rates
        print(f"\n⚠️  ERROR RATES:")
        error_rates = df.groupby('Model')['Has Error'].mean().sort_values()
        for model, rate in error_rates.items():
            status = "✅" if rate == 0 else "⚠️" if rate < 0.5 else "❌"
            print(f"  {status} {model}: {rate:.1%}")
        
        print("="*60)

# Example usage and test cases
def create_sample_test_cases() -> List[TestCase]:
    """Create sample test cases for evaluation"""
    return [
        TestCase(
            input_prompt="Explain quantum computing in simple terms.",
            category="explanation",
            difficulty="medium"
        ),
        TestCase(
            input_prompt="Write a Python function to calculate fibonacci numbers.",
            category="coding",
            difficulty="easy"
        ),
        TestCase(
            input_prompt="Analyze the geopolitical implications of renewable energy adoption.",
            category="analysis",
            difficulty="hard"
        ),
        TestCase(
            input_prompt="What is 15 * 23?",
            category="math",
            difficulty="easy"
        ),
        TestCase(
            input_prompt="Summarize the key points of machine learning in 3 bullet points.",
            category="summarization",
            difficulty="easy"
        )
    ]

async def main():
    """Main execution function"""
    # Configuration
    config = EvaluationConfig(
        ollama_base_url="http://localhost:11434",
        anthropic_api_key=None,  # Set your API key here: "sk-ant-..."
        openai_api_key=None,     # Set your API key here: "sk-..."
        evaluator_model="llama3.2",  # Use Ollama model as default evaluator
        output_dir="evaluation_results"
    )
    
    # Initialize framework
    framework = LLMEvaluationFramework(config)
    
    # Add test cases
    test_cases = create_sample_test_cases()
    framework.add_test_cases(test_cases)
    
    # Define models to evaluate - start with just Ollama models
    models_to_evaluate = {
        'ollama': ['llama3.2'],  # Add your available Ollama models
        # Uncomment and add API keys to use these:
        # 'openai': ['gpt-3.5-turbo'],      
        # 'claude': ['claude-3-sonnet-20240229']  
    }
    
    try:
        # Run evaluation
        await framework.run_evaluation(models_to_evaluate)
        
        # Generate results
        framework.print_summary()
        framework.create_visualizations()
        framework.export_results()
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"Error: {e}")
        print("Make sure Ollama is running and models are available.")
        print("Check: curl http://localhost:11434/api/tags")

if __name__ == "__main__":
    # Run the evaluation
    asyncio.run(main())