"""
LLM Code Generation Benchmark System
Inspired by Aider polyglot benchmarks, using Exercism exercises
"""

import json
import time
import asyncio
import logging
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import subprocess
import tempfile
import shutil
import re
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    ollama_base_url: str = "http://localhost:11434"
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    custom_llm_base_url: Optional[str] = None
    custom_llm_api_key: Optional[str] = None
    timeout: int = 120
    max_workers: int = 4
    cache_dir: Path = Path("./cache")
    results_dir: Path = Path("./results")

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

class ModelProvider:
    """Base class for model providers"""
    def __init__(self, config: EvaluationConfig):
        self.config = config

    async def generate(self, prompt: str, model: str) -> Tuple[str, float, int]:
        raise NotImplementedError

class OllamaProvider(ModelProvider):
    """Ollama model provider"""
    async def generate(self, prompt: str, model: str) -> Tuple[str, float, int]:
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.config.ollama_base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=self.config.timeout
            )
            response.raise_for_status()
            data = response.json()
            elapsed = time.time() - start_time
            output = data.get("response", "")
            token_count = len(output.split())
            return output, elapsed, token_count
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return f"Error: {e}", time.time() - start_time, 0

class ExercismFetcher:
    """Fetches exercises from Exercism"""
    
    BASE_URL = "https://exercism.org/api/v2"
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_exercise_list(self, language: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get list of exercises for a language"""
        cache_file = self.cache_dir / f"{language}_exercises.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)[:limit]
        
        try:
            # Note: This is a simplified example. In practice, you'd need to handle
            # Exercism's API authentication and rate limiting
            url = f"{self.BASE_URL}/tracks/{language}/exercises"
            response = requests.get(url)
            response.raise_for_status()
            exercises = response.json()["exercises"][:limit]
            
            with open(cache_file, 'w') as f:
                json.dump(exercises, f)
            
            return exercises
        except Exception as e:
            logger.error(f"Failed to fetch exercises: {e}")
            return []

class CodeExecutor:
    """Executes code and runs tests"""
    
    LANGUAGE_CONFIG = {
        "python": {
            "extension": ".py",
            "run_command": ["python3"],
            "test_framework": "pytest"
        },
        "javascript": {
            "extension": ".js",
            "run_command": ["node"],
            "test_framework": "jest"
        },
        "cpp": {
            "extension": ".cpp",
            "compile_command": ["g++", "-std=c++17", "-o", "{output}", "{source}"],
            "run_command": ["./{output}"],
            "test_framework": "catch2"
        },
        "java": {
            "extension": ".java",
            "compile_command": ["javac", "{source}"],
            "run_command": ["java", "{class_name}"],
            "test_framework": "junit"
        },
        "rust": {
            "extension": ".rs",
            "compile_command": ["rustc", "{source}"],
            "run_command": ["./{output}"],
            "test_framework": "cargo test"
        }
    }
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
    async def execute_code(self, code: str, language: str, test_code: str) -> Tuple[bool, str]:
        """Execute code with tests and return success status and output"""
        lang_config = self.LANGUAGE_CONFIG.get(language)
        if not lang_config:
            return False, f"Unsupported language: {language}"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Write solution file
            solution_file = tmppath / f"solution{lang_config['extension']}"
            with open(solution_file, 'w') as f:
                f.write(code)
            
            # Write test file
            test_file = tmppath / f"test{lang_config['extension']}"
            with open(test_file, 'w') as f:
                f.write(test_code)
            
            try:
                # Compile if needed
                if "compile_command" in lang_config:
                    compile_cmd = [
                        arg.format(source=str(solution_file), output="solution")
                        for arg in lang_config["compile_command"]
                    ]
                    result = subprocess.run(
                        compile_cmd,
                        cwd=tmpdir,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.returncode != 0:
                        return False, f"Compilation failed: {result.stderr}"
                
                # Run tests
                if language == "python":
                    # For Python, we'll run pytest
                    test_cmd = ["python3", "-m", "pytest", str(test_file), "-v"]
                elif language == "javascript":
                    # For JS, we'll run the test file directly with node
                    test_cmd = ["node", str(test_file)]
                else:
                    # For compiled languages, run the executable
                    test_cmd = lang_config["run_command"]
                
                result = subprocess.run(
                    test_cmd,
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                success = result.returncode == 0
                output = result.stdout + result.stderr
                
                return success, output
                
            except subprocess.TimeoutExpired:
                return False, "Execution timed out"
            except Exception as e:
                return False, f"Execution error: {str(e)}"

class PromptGenerator:
    """Generates prompts for different exercises"""
    
    @staticmethod
    def generate_prompt(exercise: Exercise) -> str:
        """Generate a prompt for code generation"""
        prompt = f"""You are an expert {exercise.language} programmer. 
Please solve the following programming exercise.

Exercise: {exercise.name}

Instructions:
{exercise.instructions}

{f"Starter code:{chr(10)}{exercise.starter_code}" if exercise.starter_code else ""}

Requirements:
1. Write a complete, working solution in {exercise.language}
2. The code should pass all test cases
3. Use best practices and clean code principles
4. Include only the solution code, no explanations

Please provide the complete solution code:
"""
        return prompt
    
    @staticmethod
    def extract_code(response: str, language: str) -> str:
        """Extract code from LLM response"""
        # Try to find code blocks
        code_block_pattern = r"```(?:" + language + r")?\n(.*?)```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks, try to extract based on common patterns
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            # Simple heuristic: lines that look like code
            if any([
                line.strip().startswith(('def ', 'class ', 'function ', 'const ', 'let ', 'var ', '#include')),
                line.strip().endswith(('{', '}', ';', ':')),
                'return ' in line,
                '=' in line and not line.strip().startswith('#')
            ]):
                in_code = True
            
            if in_code and line.strip():
                code_lines.append(line)
        
        return '\n'.join(code_lines) if code_lines else response

class LLMBenchmark:
    """Main benchmark orchestrator"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.config.results_dir.mkdir(exist_ok=True)
        self.exercism = ExercismFetcher(config.cache_dir)
        self.executor = CodeExecutor(config)
        self.prompt_gen = PromptGenerator()
        self.providers = {
            "ollama": OllamaProvider(config)
        }
        
    def get_sample_exercises(self) -> List[Exercise]:
        """Get sample exercises for testing"""
        # These are simplified examples - in practice, you'd fetch from Exercism
        return [
            Exercise(
                slug="two-fer",
                name="Two Fer",
                language="python",
                instructions="Create a function that gives out a message: 'One for X, one for me.' where X is the given name. If no name is given, use 'you'.",
                test_code="""
def test_two_fer():
    from solution import two_fer
    assert two_fer() == "One for you, one for me."
    assert two_fer("Alice") == "One for Alice, one for me."
    assert two_fer("Bob") == "One for Bob, one for me."
""",
                starter_code="def two_fer(name=None):\n    pass",
                difficulty="easy"
            ),
            Exercise(
                slug="leap",
                name="Leap Year",
                language="python",
                instructions="Given a year, report if it is a leap year. A leap year occurs on every year that is evenly divisible by 4, except every year that is evenly divisible by 100, unless the year is also evenly divisible by 400.",
                test_code="""
def test_leap_year():
    from solution import is_leap_year
    assert is_leap_year(2000) == True
    assert is_leap_year(2100) == False
    assert is_leap_year(2020) == True
    assert is_leap_year(2021) == False
""",
                difficulty="easy"
            ),
            Exercise(
                slug="fizzbuzz",
                name="FizzBuzz",
                language="javascript",
                instructions="Write a function that returns 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for multiples of both, and the number itself otherwise.",
                test_code="""
const fizzbuzz = require('./solution').fizzbuzz;

console.assert(fizzbuzz(3) === 'Fizz');
console.assert(fizzbuzz(5) === 'Buzz');
console.assert(fizzbuzz(15) === 'FizzBuzz');
console.assert(fizzbuzz(7) === '7');
console.log('All tests passed!');
""",
                difficulty="easy"
            )
        ]
    
    async def run_single_benchmark(self, model: str, exercise: Exercise, provider_name: str = "ollama") -> BenchmarkResult:
        """Run a single benchmark test"""
        provider = self.providers.get(provider_name)
        if not provider:
            return BenchmarkResult(
                model=model,
                exercise=exercise.slug,
                language=exercise.language,
                success=False,
                execution_time=0,
                token_count=0,
                error_message=f"Unknown provider: {provider_name}"
            )
        
        # Generate prompt
        prompt = self.prompt_gen.generate_prompt(exercise)
        
        # Get LLM response
        response, exec_time, tokens = await provider.generate(prompt, model)
        
        if response.startswith("Error:"):
            return BenchmarkResult(
                model=model,
                exercise=exercise.slug,
                language=exercise.language,
                success=False,
                execution_time=exec_time,
                token_count=tokens,
                error_message=response,
                generated_code=""
            )
        
        # Extract code from response
        code = self.prompt_gen.extract_code(response, exercise.language)
        
        # Execute code with tests
        success, output = await self.executor.execute_code(code, exercise.language, exercise.test_code)
        
        return BenchmarkResult(
            model=model,
            exercise=exercise.slug,
            language=exercise.language,
            success=success,
            execution_time=exec_time,
            token_count=tokens,
            error_message=None if success else "Test failed",
            generated_code=code,
            test_output=output
        )
    
    async def run_benchmarks(self, models: List[str], exercises: Optional[List[Exercise]] = None) -> List[BenchmarkResult]:
        """Run benchmarks for multiple models and exercises"""
        if exercises is None:
            exercises = self.get_sample_exercises()
        
        results = []
        total = len(models) * len(exercises)
        completed = 0
        
        for model in models:
            logger.info(f"Testing model: {model}")
            
            for exercise in exercises:
                logger.info(f"  Running exercise: {exercise.name} ({exercise.language})")
                
                result = await self.run_single_benchmark(model, exercise)
                results.append(result)
                
                completed += 1
                success_str = "✓" if result.success else "✗"
                logger.info(f"    Result: {success_str} (Time: {result.execution_time:.2f}s, Tokens: {result.token_count})")
                
                if not result.success and result.error_message:
                    logger.debug(f"    Error: {result.error_message}")
                
                logger.info(f"  Progress: {completed}/{total} ({completed/total*100:.1f}%)")
        
        return results
    
    def save_results(self, results: List[BenchmarkResult], filename: Optional[str] = None):
        """Save benchmark results to file"""
        if filename is None:
            filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.config.results_dir / filename
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "timeout": self.config.timeout,
                "models": list(set(r.model for r in results)),
                "exercises": list(set(r.exercise for r in results))
            },
            "results": [
                {
                    "model": r.model,
                    "exercise": r.exercise,
                    "language": r.language,
                    "success": r.success,
                    "execution_time": r.execution_time,
                    "token_count": r.token_count,
                    "error_message": r.error_message,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to: {filepath}")
    
    def generate_report(self, results: List[BenchmarkResult]) -> str:
        """Generate a markdown report from results"""
        report = []
        report.append("# LLM Code Generation Benchmark Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Summary by model
        report.append("## Summary by Model\n")
        model_stats = {}
        for r in results:
            if r.model not in model_stats:
                model_stats[r.model] = {"total": 0, "passed": 0, "total_time": 0, "total_tokens": 0}
            model_stats[r.model]["total"] += 1
            if r.success:
                model_stats[r.model]["passed"] += 1
            model_stats[r.model]["total_time"] += r.execution_time
            model_stats[r.model]["total_tokens"] += r.token_count
        
        report.append("| Model | Pass Rate | Avg Time (s) | Avg Tokens |")
        report.append("|-------|-----------|--------------|------------|")
        for model, stats in model_stats.items():
            pass_rate = stats["passed"] / stats["total"] * 100
            avg_time = stats["total_time"] / stats["total"]
            avg_tokens = stats["total_tokens"] / stats["total"]
            report.append(f"| {model} | {pass_rate:.1f}% | {avg_time:.2f} | {avg_tokens:.0f} |")
        
        # Summary by language
        report.append("\n## Summary by Language\n")
        lang_stats = {}
        for r in results:
            if r.language not in lang_stats:
                lang_stats[r.language] = {"total": 0, "passed": 0}
            lang_stats[r.language]["total"] += 1
            if r.success:
                lang_stats[r.language]["passed"] += 1
        
        report.append("| Language | Pass Rate | Total Tests |")
        report.append("|----------|-----------|-------------|")
        for lang, stats in lang_stats.items():
            pass_rate = stats["passed"] / stats["total"] * 100
            report.append(f"| {lang} | {pass_rate:.1f}% | {stats['total']} |")
        
        # Detailed results
        report.append("\n## Detailed Results\n")
        for r in sorted(results, key=lambda x: (x.model, x.exercise)):
            status = "✓ PASS" if r.success else "✗ FAIL"
            report.append(f"### {r.model} - {r.exercise} ({r.language})")
            report.append(f"- Status: {status}")
            report.append(f"- Time: {r.execution_time:.2f}s")
            report.append(f"- Tokens: {r.token_count}")
            if r.error_message:
                report.append(f"- Error: {r.error_message}")
            report.append("")
        
        return "\n".join(report)

class BenchmarkVisualizer:
    """Generate visualizations for benchmark results"""
    
    def __init__(self, results: List[BenchmarkResult], output_dir: Path):
        self.results = results
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def create_all_visualizations(self):
        """Create all visualization charts"""
        logger.info("Generating visualizations...")
        
        # Convert results to DataFrame for easier analysis
        self.df = pd.DataFrame([
            {
                'model': r.model,
                'exercise': r.exercise,
                'language': r.language,
                'success': r.success,
                'execution_time': r.execution_time,
                'token_count': r.token_count,
                'timestamp': r.timestamp
            }
            for r in self.results
        ])
        
        # Generate different visualizations
        self.create_success_rate_by_model()
        self.create_success_rate_by_language()
        self.create_execution_time_comparison()
        self.create_token_usage_analysis()
        self.create_heatmap_model_exercise()
        self.create_performance_radar_chart()
        self.create_combined_dashboard()
        
        logger.info(f"Visualizations saved to: {self.output_dir}")
    
    def create_success_rate_by_model(self):
        """Bar chart of success rates by model"""
        plt.figure(figsize=(10, 6))
        
        # Calculate success rates
        success_rates = self.df.groupby('model')['success'].agg(['sum', 'count'])
        success_rates['rate'] = (success_rates['sum'] / success_rates['count']) * 100
        
        # Create bar chart
        bars = plt.bar(success_rates.index, success_rates['rate'])
        
        # Color bars based on performance
        colors = ['#2ecc71' if rate >= 80 else '#f39c12' if rate >= 60 else '#e74c3c' 
                  for rate in success_rates['rate']]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.title('Success Rate by Model', fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Success Rate (%)', fontsize=12)
        plt.ylim(0, 105)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'success_rate_by_model.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_success_rate_by_language(self):
        """Bar chart of success rates by programming language"""
        plt.figure(figsize=(10, 6))
        
        # Calculate success rates by language
        lang_success = self.df.groupby('language')['success'].agg(['sum', 'count'])
        lang_success['rate'] = (lang_success['sum'] / lang_success['count']) * 100
        
        # Create horizontal bar chart
        bars = plt.barh(lang_success.index, lang_success['rate'])
        
        # Apply gradient colors
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.title('Success Rate by Programming Language', fontsize=16, fontweight='bold')
        plt.xlabel('Success Rate (%)', fontsize=12)
        plt.ylabel('Language', fontsize=12)
        plt.xlim(0, 105)
        
        # Add value labels
        for i, (idx, row) in enumerate(lang_success.iterrows()):
            plt.text(row['rate'] + 1, i, f"{row['rate']:.1f}%", 
                    va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'success_rate_by_language.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_execution_time_comparison(self):
        """Box plot of execution times by model"""
        plt.figure(figsize=(12, 6))
        
        # Create box plot
        models = self.df['model'].unique()
        data_to_plot = [self.df[self.df['model'] == model]['execution_time'].values 
                        for model in models]
        
        bp = plt.boxplot(data_to_plot, labels=models, patch_artist=True)
        
        # Customize box plot colors
        colors = sns.color_palette("husl", len(models))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.title('Execution Time Distribution by Model', fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Execution Time (seconds)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add median values
        medians = [np.median(data) for data in data_to_plot]
        for i, median in enumerate(medians):
            plt.text(i + 1, median + 0.5, f'{median:.2f}s', 
                    ha='center', va='bottom', fontsize=10)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'execution_time_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_token_usage_analysis(self):
        """Scatter plot of tokens vs execution time"""
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot for each model
        models = self.df['model'].unique()
        colors = sns.color_palette("husl", len(models))
        
        for model, color in zip(models, colors):
            model_data = self.df[self.df['model'] == model]
            success_data = model_data[model_data['success']]
            fail_data = model_data[~model_data['success']]
            
            # Plot successful runs
            plt.scatter(success_data['token_count'], success_data['execution_time'], 
                       label=f'{model} (success)', color=color, s=100, alpha=0.7, marker='o')
            
            # Plot failed runs
            plt.scatter(fail_data['token_count'], fail_data['execution_time'], 
                       label=f'{model} (failed)', color=color, s=100, alpha=0.7, marker='x')
        
        plt.title('Token Usage vs Execution Time', fontsize=16, fontweight='bold')
        plt.xlabel('Token Count', fontsize=12)
        plt.ylabel('Execution Time (seconds)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add trend line
        x = self.df['token_count']
        y = self.df['execution_time']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(sorted(x), p(sorted(x)), "r--", alpha=0.5, label='Trend')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'token_usage_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_heatmap_model_exercise(self):
        """Heatmap showing success/failure for each model-exercise combination"""
        plt.figure(figsize=(12, 8))
        
        # Pivot the data for heatmap
        pivot_data = self.df.pivot_table(
            values='success', 
            index='exercise', 
            columns='model', 
            aggfunc='mean'
        ) * 100  # Convert to percentage
        
        # Create heatmap
        sns.heatmap(pivot_data, 
                   annot=True, 
                   fmt='.0f', 
                   cmap='RdYlGn', 
                   center=50,
                   cbar_kws={'label': 'Success Rate (%)'},
                   linewidths=0.5,
                   square=True)
        
        plt.title('Model Performance Heatmap by Exercise', fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Exercise', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'heatmap_model_exercise.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_performance_radar_chart(self):
        """Radar chart comparing models across multiple metrics"""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Calculate metrics for each model
        models = self.df['model'].unique()
        metrics = ['Success Rate', 'Avg Speed', 'Token Efficiency', 'Consistency']
        
        # Prepare data
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, model in enumerate(models):
            model_data = self.df[self.df['model'] == model]
            
            # Calculate metrics (normalized to 0-100)
            success_rate = (model_data['success'].sum() / len(model_data)) * 100
            avg_speed = 100 - min((model_data['execution_time'].mean() / 10) * 100, 100)  # Inverted, capped at 100
            token_efficiency = 100 - min((model_data['token_count'].mean() / 1000) * 100, 100)  # Inverted, capped at 100
            consistency = 100 - min((model_data['execution_time'].std() / model_data['execution_time'].mean()) * 100, 100)
            
            values = [success_rate, avg_speed, token_efficiency, consistency]
            values += values[:1]  # Complete the circle
            
            # Plot
            ax.plot(angles, values, 'o-', linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.25)
        
        # Customize chart
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 100)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_yticklabels(['0', '20', '40', '60', '80', '100'])
        ax.grid(True)
        
        plt.title('Model Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_combined_dashboard(self):
        """Create a comprehensive dashboard with multiple visualizations"""
        fig = plt.figure(figsize=(20, 12))
        
        # Define grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Overall Summary (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._create_summary_stats(ax1)
        
        # 2. Success rate by model (top-middle and top-right)
        ax2 = fig.add_subplot(gs[0, 1:])
        self._create_model_comparison_bars(ax2)
        
        # 3. Language performance (middle-left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._create_language_pie_chart(ax3)
        
        # 4. Time distribution (middle-center)
        ax4 = fig.add_subplot(gs[1, 1])
        self._create_time_distribution(ax4)
        
        # 5. Token distribution (middle-right)
        ax5 = fig.add_subplot(gs[1, 2])
        self._create_token_distribution(ax5)
        
        # 6. Performance matrix (bottom row)
        ax6 = fig.add_subplot(gs[2, :])
        self._create_performance_matrix(ax6)
        
        plt.suptitle('LLM Code Generation Benchmark Dashboard', fontsize=20, fontweight='bold')
        plt.savefig(self.output_dir / 'benchmark_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_summary_stats(self, ax):
        """Create summary statistics panel"""
        ax.axis('off')
        
        # Calculate overall stats
        total_tests = len(self.df)
        total_passed = self.df['success'].sum()
        overall_success_rate = (total_passed / total_tests) * 100
        avg_time = self.df['execution_time'].mean()
        avg_tokens = self.df['token_count'].mean()
        
        # Create text summary
        summary_text = f"""
        OVERALL STATISTICS
        
        Total Tests: {total_tests}
        Passed: {total_passed}
        Failed: {total_tests - total_passed}
        
        Success Rate: {overall_success_rate:.1f}%
        Avg Time: {avg_time:.2f}s
        Avg Tokens: {avg_tokens:.0f}
        """
        
        ax.text(0.1, 0.5, summary_text, fontsize=14, 
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5))
        
        ax.set_title('Summary Statistics', fontsize=14, fontweight='bold')
    
    def _create_model_comparison_bars(self, ax):
        """Create grouped bar chart for model comparison"""
        model_stats = self.df.groupby('model').agg({
            'success': ['sum', 'count'],
            'execution_time': 'mean',
            'token_count': 'mean'
        })
        
        model_stats['success_rate'] = (model_stats['success']['sum'] / model_stats['success']['count']) * 100
        
        x = np.arange(len(model_stats.index))
        width = 0.35
        
        # Success rate bars
        bars1 = ax.bar(x - width/2, model_stats['success_rate'], width, label='Success Rate (%)')
        
        # Normalized execution time (inverted for better visualization)
        max_time = model_stats['execution_time']['mean'].max()
        normalized_time = (1 - model_stats['execution_time']['mean'] / max_time) * 100
        bars2 = ax.bar(x + width/2, normalized_time, width, label='Speed Score')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_stats.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_language_pie_chart(self, ax):
        """Create pie chart for language distribution"""
        lang_counts = self.df.groupby('language')['success'].agg(['sum', 'count'])
        lang_counts['success_rate'] = lang_counts['sum'] / lang_counts['count']
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(lang_counts)))
        wedges, texts, autotexts = ax.pie(lang_counts['count'], 
                                          labels=lang_counts.index,
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          startangle=90)
        
        ax.set_title('Test Distribution by Language', fontsize=14, fontweight='bold')
    
    def _create_time_distribution(self, ax):
        """Create histogram of execution times"""
        ax.hist(self.df['execution_time'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(self.df['execution_time'].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {self.df["execution_time"].mean():.2f}s')
        ax.set_xlabel('Execution Time (s)')
        ax.set_ylabel('Frequency')
        ax.set_title('Execution Time Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_token_distribution(self, ax):
        """Create histogram of token counts"""
        ax.hist(self.df['token_count'], bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
        ax.axvline(self.df['token_count'].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {self.df["token_count"].mean():.0f}')
        ax.set_xlabel('Token Count')
        ax.set_ylabel('Frequency')
        ax.set_title('Token Usage Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_performance_matrix(self, ax):
        """Create a performance matrix visualization"""
        # Create a matrix of model vs exercise with color coding
        models = sorted(self.df['model'].unique())
        exercises = sorted(self.df['exercise'].unique())
        
        matrix = np.zeros((len(models), len(exercises)))
        
        for i, model in enumerate(models):
            for j, exercise in enumerate(exercises):
                result = self.df[(self.df['model'] == model) & (self.df['exercise'] == exercise)]
                if not result.empty:
                    matrix[i, j] = 1 if result.iloc[0]['success'] else -1
        
        # Custom colormap: red for fail, green for pass, white for no data
        colors = ['red', 'white', 'green']
        n_bins = 3
        cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks
        ax.set_xticks(np.arange(len(exercises)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(exercises, rotation=45, ha='right')
        ax.set_yticklabels(models)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1])
        cbar.ax.set_yticklabels(['Failed', 'No Data', 'Passed'])
        
        ax.set_title('Model-Exercise Performance Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Exercise')
        ax.set_ylabel('Model')
        
        # Add grid
        ax.set_xticks(np.arange(len(exercises)+1)-.5, minor=True)
        ax.set_yticks(np.arange(len(models)+1)-.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)

async def main():
    """Main entry point"""
    # Configuration
    config = EvaluationConfig()
    
    # Create benchmark instance
    benchmark = LLMBenchmark(config)
    
    # Models to test (adjust based on what's available in your Ollama)
    models = [
        "llama3.2:latest",
        "llama3.1:latest",
        "deepseek-coder:6.7b-instruct",
        # Add more models as needed
    ]
    
    # Run benchmarks
    logger.info("Starting LLM Code Generation Benchmarks")
    results = await benchmark.run_benchmarks(models)
    
    # Save results
    benchmark.save_results(results)
    
    # Generate and save report
    report = benchmark.generate_report(results)
    report_path = config.results_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to: {report_path}")
    
    # Generate visualizations
    visualizer = BenchmarkVisualizer(results, config.results_dir / "visualizations")
    visualizer.create_all_visualizations()
    
    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK COMPLETE")
    print("="*50)
    for model in models:
        model_results = [r for r in results if r.model == model]
        passed = sum(1 for r in model_results if r.success)
        total = len(model_results)
        print(f"{model}: {passed}/{total} passed ({passed/total*100:.1f}%)")
    
    print(f"\nResults saved to: {config.results_dir}")
    print(f"Visualizations saved to: {config.results_dir / 'visualizations'}")

if __name__ == "__main__":
    asyncio.run(main())