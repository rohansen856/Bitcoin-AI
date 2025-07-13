"""
Main entry point for the LLM Code Generation Benchmark System.
"""

import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from core.config import EvaluationConfig
from core.benchmark import LLMBenchmark, ReportGenerator
from utils.logger import configure_benchmark_logging, get_logger
from visualization.visualizer import BenchmarkVisualizer


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="LLM Code Generation Benchmark System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --models llama3.2:latest deepseek-coder:6.7b-instruct
  python main.py --models gpt-4 --provider openai --language python
  python main.py --config custom_config.json --visualize-only
        """
    )
    
    # Model configuration
    parser.add_argument(
        "--models", 
        nargs="+", 
        default=["llama3.1:latest", "llama3.2:latest", "deepseek-coder:6.7b-instruct"],
        help="List of models to benchmark"
    )
    
    parser.add_argument(
        "--provider", 
        default="ollama",
        choices=["ollama", "openai", "anthropic", "custom"],
        help="Model provider to use"
    )
    
    # Exercise filtering
    parser.add_argument(
        "--language", 
        help="Filter exercises by programming language"
    )
    
    parser.add_argument(
        "--difficulty", 
        choices=["easy", "medium", "hard"],
        help="Filter exercises by difficulty"
    )
    
    parser.add_argument(
        "--max-exercises", 
        type=int, 
        default=None,
        help="Maximum number of exercises to run"
    )
    
    # Configuration
    parser.add_argument(
        "--config", 
        type=Path,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=120,
        help="Timeout for each benchmark in seconds"
    )
    
    parser.add_argument(
        "--max-workers", 
        type=int, 
        default=4,
        help="Maximum number of concurrent workers"
    )
    
    # Output configuration
    parser.add_argument(
        "--results-dir", 
        type=Path, 
        default=Path("./results"),
        help="Directory to store results"
    )
    
    parser.add_argument(
        "--cache-dir", 
        type=Path, 
        default=Path("./cache"),
        help="Directory for caching data"
    )
    
    parser.add_argument(
        "--log-level", 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    # Execution modes
    parser.add_argument(
        "--visualize-only", 
        action="store_true",
        help="Only generate visualizations from existing results"
    )
    
    parser.add_argument(
        "--parallel", 
        action="store_true",
        help="Run benchmarks in parallel for better performance"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be run without actually running benchmarks"
    )
    
    # Visualization options
    parser.add_argument(
        "--skip-visualization", 
        action="store_true",
        help="Skip generating visualizations"
    )
    
    parser.add_argument(
        "--output-format", 
        choices=["json", "csv", "markdown", "all"],
        default="all",
        help="Output format for results"
    )
    
    return parser.parse_args()


def load_config(config_path: Optional[Path], args) -> EvaluationConfig:
    """Load configuration from file or command line args"""
    if config_path and config_path.exists():
        import json
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Override with command line args
        config_data.update({
            "timeout": args.timeout,
            "max_workers": args.max_workers,
            "results_dir": args.results_dir,
            "cache_dir": args.cache_dir
        })
        
        return EvaluationConfig(**config_data)
    else:
        return EvaluationConfig(
            timeout=args.timeout,
            max_workers=args.max_workers,
            results_dir=args.results_dir,
            cache_dir=args.cache_dir
        )


async def run_benchmarks(args, config: EvaluationConfig):
    """Run the benchmarks"""
    logger = get_logger(__name__)
    
    # Create benchmark instance
    benchmark = LLMBenchmark(config)
    
    # Get exercises
    exercises = benchmark.get_exercises(
        language=args.language,
        difficulty=args.difficulty
    )
    
    if args.max_exercises and len(exercises) > args.max_exercises:
        exercises = exercises[:args.max_exercises]
    
    logger.info(f"Running benchmarks for {len(args.models)} models "
                f"on {len(exercises)} exercises")
    
    if args.dry_run:
        logger.info("DRY RUN - Would run the following:")
        for model in args.models:
            logger.info(f"  Model: {model}")
        for exercise in exercises:
            logger.info(f"  Exercise: {exercise.name} ({exercise.language})")
        return []
    
    # Run benchmarks
    if args.parallel:
        logger.info("Running benchmarks in parallel mode")
        results = await benchmark.run_parallel_benchmarks(
            args.models, exercises, args.provider
        )
    else:
        logger.info("Running benchmarks sequentially")
        results = await benchmark.run_benchmarks(
            args.models, exercises, args.provider
        )
    
    # Save results
    results_file = benchmark.save_results(results)
    
    # Generate reports
    if args.output_format in ["markdown", "all"]:
        report = benchmark.generate_report(results)
        report_path = config.results_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Markdown report saved to: {report_path}")
    
    if args.output_format in ["csv", "all"]:
        csv_report = ReportGenerator.generate_csv_report(results)
        csv_path = config.results_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(csv_path, 'w') as f:
            f.write(csv_report)
        logger.info(f"CSV report saved to: {csv_path}")
    
    # Print summary
    summary = benchmark.get_summary_stats(results)
    logger.info(f"Benchmark complete! Overall pass rate: {summary['overall_pass_rate']:.1f}%")
    
    return results


def generate_visualizations(results, config: EvaluationConfig, logger):
    """Generate visualizations from results"""
    if not results:
        logger.warning("No results to visualize")
        return
    
    viz_dir = config.results_dir / "visualizations"
    visualizer = BenchmarkVisualizer(results, viz_dir)
    
    logger.info("Generating visualizations...")
    created_files = visualizer.create_all_visualizations()
    
    logger.info(f"Generated {len(created_files)} visualization files:")
    for file_path in created_files:
        logger.info(f"  {file_path}")


async def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config, args)
    
    # Configure logging
    configure_benchmark_logging(
        config.results_dir,
        args.log_level,
        enable_file_logging=not args.dry_run
    )
    
    logger = get_logger(__name__)
    logger.info("Starting LLM Code Generation Benchmark System")
    logger.info(f"Configuration: {config}")
    
    try:
        if args.visualize_only:
            # Load existing results and generate visualizations
            logger.info("Visualize-only mode: looking for existing results")
            results_files = list(config.results_dir.glob("benchmark_*.json"))
            
            if not results_files:
                logger.error("No existing results found for visualization")
                return
            
            # Load the most recent results
            latest_results_file = max(results_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Loading results from: {latest_results_file}")
            
            benchmark = LLMBenchmark(config)
            results = benchmark.load_results(latest_results_file)
            
            if not results:
                logger.error("Failed to load results")
                return
                
        else:
            # Run benchmarks
            results = await run_benchmarks(args, config)
        
        # Generate visualizations unless skipped
        if not args.skip_visualization:
            generate_visualizations(results, config, logger)
        
        # Final summary
        if results:
            logger.info("="*50)
            logger.info("BENCHMARK COMPLETE")
            logger.info("="*50)
            
            # Group results by model
            model_results = {}
            for result in results:
                if result.model not in model_results:
                    model_results[result.model] = []
                model_results[result.model].append(result)
            
            # Print summary for each model
            for model, model_result_list in model_results.items():
                passed = sum(1 for r in model_result_list if r.success)
                total = len(model_result_list)
                logger.info(f"{model}: {passed}/{total} passed ({passed/total*100:.1f}%)")
            
            logger.info(f"Results saved to: {config.results_dir}")
            if not args.skip_visualization:
                logger.info(f"Visualizations saved to: {config.results_dir / 'visualizations'}")
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


def run_quick_benchmark():
    """Quick benchmark function for testing"""
    import asyncio
    
    # Simple configuration for quick testing
    config = EvaluationConfig(
        timeout=60,
        max_workers=2,
        results_dir=Path("./quick_results")
    )
    
    configure_benchmark_logging(config.results_dir, "INFO")
    
    async def quick_run():
        benchmark = LLMBenchmark(config)
        
        # Get a few sample exercises
        exercises = benchmark.get_exercises(language="python")[:2]
        models = ["llama3.2:latest"]
        
        logger = get_logger(__name__)
        logger.info("Running quick benchmark...")
        
        results = await benchmark.run_benchmarks(models, exercises)
        
        # Save and summarize
        benchmark.save_results(results)
        summary = benchmark.get_summary_stats(results)
        logger.info(f"Quick benchmark complete: {summary['overall_pass_rate']:.1f}% pass rate")
        
        return results
    
    return asyncio.run(quick_run())


if __name__ == "__main__":
    asyncio.run(main())