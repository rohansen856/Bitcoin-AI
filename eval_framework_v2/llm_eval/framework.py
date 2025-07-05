from dataclasses import asdict
from datetime import datetime
import json
import logging
import asyncio
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List
from .config import EvaluationConfig
from .test_case import TestCase
from .evaluation_result import EvaluationResult
from .providers import OllamaProvider, ClaudeProvider, OpenAIProvider, CustomLLMProvider
from .custom_deepeval_model import CustomDeepEvalModel
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

logger = logging.getLogger(__name__)

class LLMEvaluationFramework:
    """Main evaluation framework"""
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.providers = {
            'ollama': OllamaProvider(config),
            'claude': ClaudeProvider(config),
            'openai': OpenAIProvider(config),
            'custom': CustomLLMProvider(config)
        }
        self.results: List[EvaluationResult] = []
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def add_test_cases(self, test_cases: List[TestCase]):
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
                            assert self.config.openai_api_key.startswith('sk-'), "Invalid OpenAI key format"
                            evaluator_provider = self.providers['openai']
                            print(f"Using openai LLM provider: {evaluator_model_name}")
                        elif evaluator_model_name.startswith('claude') and self.config.anthropic_api_key:
                            assert self.config.anthropic_api_key.startswith('sk-ant-'), "Invalid Anthropic key format"
                            evaluator_provider = self.providers['claude']
                            print(f"Using claude LLM provider: {evaluator_model_name}")
                        elif evaluator_model_name.startswith('custom') and self.config.custom_llm_base_url:
                            assert self.config.custom_llm_api_key.startswith('sk-'), "Invalid custom API key format"
                            evaluator_provider = self.providers['custom']
                            print(f"Using custom LLM provider: {evaluator_model_name}")
                        elif 'ollama' in self.providers:
                            evaluator_provider = self.providers['ollama']
                            evaluator_model_name = 'llama3.2'  # Default Ollama model
                            print(f"Using ollama LLM provider: {evaluator_model_name}")
                        
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
            e.g., {'ollama': ['llama2', 'mistral'], 'openai': ['gpt-3.5-turbo'], 'custom': ['custom']}
        """
        logger.info("Starting LLM evaluation framework")
        
        for provider_name, model_list in models.items():
            if provider_name not in self.providers:
                logger.warning(f"Unknown provider: {provider_name}")
                continue
            
            for model_name in model_list:
                await self.evaluate_model(provider_name, model_name)
        
        logger.info("Evaluation completed")

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