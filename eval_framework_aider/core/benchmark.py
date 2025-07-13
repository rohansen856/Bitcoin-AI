"""
Main benchmark orchestrator for running LLM code generation tests.
"""

import json
import asyncio
from typing import List, Optional
from datetime import datetime
from pathlib import Path

from models.data_models import Exercise, BenchmarkResult
from models.providers import get_provider
from core.config import EvaluationConfig
from core.executor import CodeExecutor
from core.prompt_generator import PromptGenerator
from data.exercism_fetcher import SampleExerciseProvider
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMBenchmark:
    """Main benchmark orchestrator"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.executor = CodeExecutor(config)
        self.prompt_gen = PromptGenerator()
        self.sample_provider = SampleExerciseProvider()
        
    def get_exercises(self, language: Optional[str] = None, 
                     difficulty: Optional[str] = None) -> List[Exercise]:
        """Get exercises based on filters"""
        exercises = self.sample_provider.get_sample_exercises()
        
        if language:
            exercises = [ex for ex in exercises if ex.language == language]
        
        if difficulty:
            exercises = [ex for ex in exercises if ex.difficulty == difficulty]
            
        return exercises
    
    async def run_single_benchmark(self, model: str, exercise: Exercise, 
                                 provider_name: str = "ollama") -> BenchmarkResult:
        """Run a single benchmark test"""
        try:
            provider = get_provider(provider_name, self.config)
        except ValueError as e:
            return BenchmarkResult(
                model=model,
                exercise=exercise.slug,
                language=exercise.language,
                success=False,
                execution_time=0,
                token_count=0,
                error_message=str(e)
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
        
        # Extract and clean code from response
        code = self.prompt_gen.extract_code(response, exercise.language)
        code = self.prompt_gen.clean_code(code)
        
        # Validate code structure
        if not self.prompt_gen.validate_code_structure(code, exercise.language):
            return BenchmarkResult(
                model=model,
                exercise=exercise.slug,
                language=exercise.language,
                success=False,
                execution_time=exec_time,
                token_count=tokens,
                error_message="Invalid code structure",
                generated_code=code
            )
        
        # Execute code with tests
        success, output = await self.executor.execute_code(
            code, exercise.language, exercise.test_code
        )
        
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
    
    async def run_benchmarks(self, models: List[str], 
                           exercises: Optional[List[Exercise]] = None,
                           provider_name: str = "ollama") -> List[BenchmarkResult]:
        """Run benchmarks for multiple models and exercises"""
        if exercises is None:
            exercises = self.get_exercises()
        
        results = []
        total = len(models) * len(exercises)
        completed = 0
        
        # Use semaphore to limit concurrent executions
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def run_with_semaphore(model: str, exercise: Exercise) -> BenchmarkResult:
            async with semaphore:
                return await self.run_single_benchmark(model, exercise, provider_name)
        
        # Create all tasks
        tasks = []
        for model in models:
            for exercise in exercises:
                task = asyncio.create_task(run_with_semaphore(model, exercise))
                tasks.append((model, exercise.name, task))
        
        # Process results as they complete
        for model, exercise_name, task in tasks:
            logger.info(f"Running: {model} - {exercise_name}")
            
            result = await task
            results.append(result)
            
            completed += 1
            success_str = "✓" if result.success else "✗"
            logger.info(f"  Result: {success_str} (Time: {result.execution_time:.2f}s, "
                       f"Tokens: {result.token_count})")
            
            if not result.success and result.error_message:
                logger.debug(f"  Error: {result.error_message}")
            
            logger.info(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")
        
        return results
    
    async def run_parallel_benchmarks(self, models: List[str], 
                                    exercises: Optional[List[Exercise]] = None,
                                    provider_name: str = "ollama") -> List[BenchmarkResult]:
        """Run benchmarks in parallel for better performance"""
        if exercises is None:
            exercises = self.get_exercises()
        
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def run_benchmark_with_progress(model: str, exercise: Exercise) -> BenchmarkResult:
            async with semaphore:
                logger.info(f"Starting: {model} - {exercise.name}")
                result = await self.run_single_benchmark(model, exercise, provider_name)
                status = "✓" if result.success else "✗"
                logger.info(f"Completed: {model} - {exercise.name} {status}")
                return result
        
        # Create all tasks
        tasks = []
        for model in models:
            for exercise in exercises:
                task = asyncio.create_task(run_benchmark_with_progress(model, exercise))
                tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    def save_results(self, results: List[BenchmarkResult], 
                    filename: Optional[str] = None) -> Path:
        """Save benchmark results to file"""
        if filename is None:
            filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.config.results_dir / filename
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "timeout": self.config.timeout,
                "max_workers": self.config.max_workers,
                "models": list(set(r.model for r in results)),
                "exercises": list(set(r.exercise for r in results)),
                "languages": list(set(r.language for r in results))
            },
            "results": [result.to_dict() for result in results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to: {filepath}")
        return filepath
    
    def load_results(self, filepath: Path) -> List[BenchmarkResult]:
        """Load benchmark results from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            results = []
            for result_data in data["results"]:
                result = BenchmarkResult.from_dict(result_data)
                results.append(result)
            
            logger.info(f"Loaded {len(results)} results from {filepath}")
            return results
        
        except Exception as e:
            logger.error(f"Failed to load results from {filepath}: {e}")
            return []
    
    def generate_report(self, results: List[BenchmarkResult]) -> str:
        """Generate a markdown report from results"""
        report = ReportGenerator.generate_markdown_report(results)
        return report
    
    def get_summary_stats(self, results: List[BenchmarkResult]) -> dict:
        """Get summary statistics from results"""
        if not results:
            return {}
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.success)
        
        # Group by model
        model_stats = {}
        for result in results:
            if result.model not in model_stats:
                model_stats[result.model] = {
                    "total": 0, "passed": 0, "times": [], "tokens": []
                }
            
            stats = model_stats[result.model]
            stats["total"] += 1
            if result.success:
                stats["passed"] += 1
            stats["times"].append(result.execution_time)
            stats["tokens"].append(result.token_count)
        
        # Calculate averages
        for model, stats in model_stats.items():
            stats["pass_rate"] = (stats["passed"] / stats["total"]) * 100
            stats["avg_time"] = sum(stats["times"]) / len(stats["times"])
            stats["avg_tokens"] = sum(stats["tokens"]) / len(stats["tokens"])
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "overall_pass_rate": (passed_tests / total_tests) * 100,
            "model_stats": model_stats
        }


class ReportGenerator:
    """Generate reports from benchmark results"""
    
    @staticmethod
    def generate_markdown_report(results: List[BenchmarkResult]) -> str:
        """Generate a markdown report from results"""
        if not results:
            return "# No Results Available\n"
        
        report = []
        report.append("# LLM Code Generation Benchmark Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Summary statistics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.success)
        overall_pass_rate = (passed_tests / total_tests) * 100
        
        report.append("## Overall Summary")
        report.append(f"- Total tests: {total_tests}")
        report.append(f"- Passed: {passed_tests}")
        report.append(f"- Failed: {total_tests - passed_tests}")
        report.append(f"- Overall pass rate: {overall_pass_rate:.1f}%\n")
        
        # Summary by model
        report.append("## Summary by Model\n")
        model_stats = {}
        for r in results:
            if r.model not in model_stats:
                model_stats[r.model] = {
                    "total": 0, "passed": 0, "total_time": 0, "total_tokens": 0
                }
            model_stats[r.model]["total"] += 1
            if r.success:
                model_stats[r.model]["passed"] += 1
            model_stats[r.model]["total_time"] += r.execution_time
            model_stats[r.model]["total_tokens"] += r.token_count
        
        report.append("| Model | Pass Rate | Avg Time (s) | Avg Tokens |")
        report.append("|-------|-----------|--------------|------------|")
        for model, stats in sorted(model_stats.items()):
            pass_rate = (stats["passed"] / stats["total"]) * 100
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
        for lang, stats in sorted(lang_stats.items()):
            pass_rate = (stats["passed"] / stats["total"]) * 100
            report.append(f"| {lang} | {pass_rate:.1f}% | {stats['total']} |")
        
        # Summary by exercise
        report.append("\n## Summary by Exercise\n")
        exercise_stats = {}
        for r in results:
            if r.exercise not in exercise_stats:
                exercise_stats[r.exercise] = {"total": 0, "passed": 0}
            exercise_stats[r.exercise]["total"] += 1
            if r.success:
                exercise_stats[r.exercise]["passed"] += 1
        
        report.append("| Exercise | Pass Rate | Total Tests |")
        report.append("|----------|-----------|-------------|")
        for exercise, stats in sorted(exercise_stats.items()):
            pass_rate = (stats["passed"] / stats["total"]) * 100
            report.append(f"| {exercise} | {pass_rate:.1f}% | {stats['total']} |")
        
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
            if r.generated_code:
                report.append(f"- Code length: {len(r.generated_code)} characters")
            report.append("")
        
        return "\n".join(report)
    
    @staticmethod
    def generate_csv_report(results: List[BenchmarkResult]) -> str:
        """Generate CSV report from results"""
        if not results:
            return "model,exercise,language,success,execution_time,token_count,error_message\n"
        
        lines = ["model,exercise,language,success,execution_time,token_count,error_message"]
        
        for r in results:
            error_msg = (r.error_message or "").replace(",", ";").replace("\n", " ")
            line = f"{r.model},{r.exercise},{r.language},{r.success},{r.execution_time:.2f},{r.token_count},{error_msg}"
            lines.append(line)
        
        return "\n".join(lines)
    
    @staticmethod
    def generate_json_summary(results: List[BenchmarkResult]) -> str:
        """Generate JSON summary from results"""
        benchmark = LLMBenchmark(EvaluationConfig())
        summary = benchmark.get_summary_stats(results)
        return json.dumps(summary, indent=2)