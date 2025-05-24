#!/usr/bin/env python3
"""
Comparative evaluation script for different sampling methods.

This script runs greedy, top-p, DCBS, and random sampling on the same dataset,
computes accuracy for each method, and generates comparative visualizations.
"""

import argparse
import json
import os
import sys
import datetime
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from src import (
        EvaluationConfig, 
        EvaluationRunner, 
        load_benchmark_data,
        generate_all_visualizations,
        setup_logging, 
        eval_logger as logger
    )
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Make sure you're running from the project root directory")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for src at: {src_path}")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare different sampling methods on multiple-choice tasks"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="meta-llama/Llama-2-7b-chat-hf",
        help="HuggingFace model name or path"
    )
    
    parser.add_argument(
        "--benchmark", 
        type=str, 
        default="data/bench_wino.json",
        help="Path to benchmark JSON file"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results",
        help="Output directory for results and visualizations"
    )
    
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit number of examples for testing (default: use all)"
    )
    
    parser.add_argument(
        "--top-p", 
        type=float, 
        default=0.9,
        help="Top-p value for nucleus sampling"
    )
    
    parser.add_argument(
        "--k", 
        type=int, 
        default=8,
        help="Number of clusters for DCBS"
    )
    
    parser.add_argument(
        "--top-n", 
        type=int, 
        default=50,
        help="Top-n tokens to consider for DCBS clustering"
    )
    
    parser.add_argument(
        "--no-cot", 
        action="store_true",
        help="Disable chain-of-thought reasoning"
    )
    
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--save-details", 
        action="store_true",
        help="Save detailed per-example results"
    )
    
    return parser.parse_args()


def save_results(results: dict, output_dir: str, save_details: bool = False):
    """Save evaluation results to JSON files."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary statistics
    summary_path = os.path.join(output_dir, "summary_results.json")
    summary_data = {
        "statistics": results["statistics"],
        "config": results["config"],
        "timestamp": str(datetime.datetime.now())
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    
    logger.info(f"Summary results saved to: {summary_path}")
    
    # Save detailed results if requested
    if save_details:
        details_path = os.path.join(output_dir, "detailed_results.json")
        with open(details_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Detailed results saved to: {details_path}")


def print_summary(results: dict):
    """Print a summary of the evaluation results."""
    print("\n" + "="*60)
    print("COMPARATIVE EVALUATION RESULTS")
    print("="*60)
    
    statistics = results["statistics"]
    config = results["config"]
    
    print(f"Model: {config['model']}")
    print(f"Total Examples: {config['total_examples']}")
    print(f"Methods: {', '.join(config['methods'])}")
    print("-" * 60)
    
    # Sort methods by accuracy for display
    method_stats = [(method, stats) for method, stats in statistics.items()]
    method_stats.sort(key=lambda x: x[1]["accuracy"], reverse=True)
    
    print(f"{'Method':<12} {'Accuracy':<12} {'95% CI':<20} {'N':<8} {'Time (ms)':<12}")
    print("-" * 60)
    
    for method, stats in method_stats:
        ci = stats.get("confidence_interval", (0, 0))
        ci_str = f"({ci[0]:.1f}, {ci[1]:.1f})"
        
        print(f"{method.title():<12} "
              f"{stats['accuracy']:.2f}%{'':<6} "
              f"{ci_str:<20} "
              f"{stats['total']:<8} "
              f"{stats.get('avg_time_ms', 0):.2f}")
    
    print("-" * 60)
    print("✓ Random baseline: 50.0%")
    print("✓ Charts saved to results/")
    print("="*60)


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Create evaluation configuration
    config = EvaluationConfig(
        model_name=args.model,
        benchmark_path=args.benchmark,
        output_dir=args.output_dir,
        limit=args.limit,
        top_p=args.top_p,
        k=args.k,
        top_n=args.top_n,
        include_cot=not args.no_cot,
        log_level=args.log_level
    )
    
    logger.info("Starting comparative evaluation")
    logger.info(f"Configuration: {config}")
    
    try:
        # Load benchmark data
        benchmark_data = load_benchmark_data(config.benchmark_path)
        
        # Create evaluation runner
        runner = EvaluationRunner(config)
        
        # Run evaluation
        logger.info("Running evaluation across all sampling methods...")
        results = runner.run_evaluation(benchmark_data)
        
        # Save results
        save_results(results, config.output_dir, args.save_details)
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        generate_all_visualizations(results, config.output_dir)
        
        # Print summary
        print_summary(results)
        
        logger.info("Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main() 