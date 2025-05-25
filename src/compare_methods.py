#!/usr/bin/env python3
"""
Comparative evaluation script for different sampling methods.
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import yaml

# Clean imports without sys.path manipulation
try:
    from src.errors import eval_logger as logger
    from src.errors import setup_logging
    from src.evaluation_core import (
        EvaluationConfig,
        EvaluationRunner,
        load_benchmark_data,
    )
    from src.visualization import generate_all_visualizations
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Make sure you're running from the project root directory")
    print(f"Current working directory: {os.getcwd()}")
    print("Try: pip install -e .")
    sys.exit(1)


def load_config_yaml(config_path: str = "configs/study_config.yaml") -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
    except FileNotFoundError:
        logger.warning(f"Configuration file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in configuration file: {config_path}: {e}")
        return {}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare different sampling methods on multiple-choice tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/study_config.yaml",
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="HuggingFace model name or path (overrides config)",
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        help="Path to benchmark JSON file (overrides config)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results and visualizations (overrides config)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of examples for testing (overrides config)",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        help="Top-p value for nucleus sampling (overrides config)",
    )

    parser.add_argument(
        "--k", type=int, help="Number of clusters for DCBS (overrides config)"
    )

    parser.add_argument(
        "--top-n",
        type=int,
        help="Top-n tokens to consider for DCBS clustering (overrides config)",
    )

    parser.add_argument(
        "--no-cot",
        action="store_true",
        help="Disable chain-of-thought reasoning (overrides config)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (overrides config)",
    )

    parser.add_argument(
        "--save-details", action="store_true", help="Save detailed per-example results"
    )

    parser.add_argument(
        "--load-in-4bit", action="store_true", help="Load model with 4-bit quantization"
    )

    parser.add_argument(
        "--samplers",
        nargs="+",
        choices=["greedy", "top-p", "dcbs", "random"],
        help="Specify which samplers to evaluate",
    )

    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable DCBS caching for performance comparison",
    )

    return parser.parse_args()


def create_evaluation_config(
    yaml_config: dict, args: argparse.Namespace
) -> EvaluationConfig:
    """Create EvaluationConfig from YAML config and CLI arguments.

    CLI arguments override YAML config values.

    Args:
        yaml_config: Configuration loaded from YAML file
        args: Parsed command line arguments

    Returns:
        EvaluationConfig instance
    """
    # Start with YAML config values, use reasonable defaults if missing
    model_name = yaml_config.get("model_path", "meta-llama/Llama-3.2-1B")
    benchmark_path = yaml_config.get("benchmark", "data/arc_easy_full.json")
    output_dir = "results"  # Always use results as default
    limit = None

    # DCBS parameters
    dcbs_params = yaml_config.get("dcbs_params", {})
    top_p = (
        yaml_config.get("p_values", [0.9])[0] if yaml_config.get("p_values") else 0.9
    )
    k = yaml_config.get(
        "clusters",
        (
            yaml_config.get("k", [8])[0]
            if isinstance(yaml_config.get("k"), list)
            else yaml_config.get("k", 8)
        ),
    )
    top_n = yaml_config.get("top_n", 50)

    # Other settings
    include_cot = True  # Default to True for CoT
    log_level = yaml_config.get("log_level", "INFO")
    load_in_4bit = False  # Default to False
    enable_caching = True  # Default to True for caching

    # Apply CLI overrides (only if explicitly provided)
    if args.model is not None:
        model_name = args.model
    if args.benchmark is not None:
        benchmark_path = args.benchmark
    if args.output_dir is not None:
        output_dir = args.output_dir
    if args.limit is not None:
        limit = args.limit
    if args.top_p is not None:
        top_p = args.top_p
    if args.k is not None:
        k = args.k
    if args.top_n is not None:
        top_n = args.top_n
    if args.no_cot:
        include_cot = False
    if args.log_level is not None:
        log_level = args.log_level
    if args.load_in_4bit:
        load_in_4bit = True
    if args.disable_cache:
        enable_caching = False

    # Create and return the config
    config = EvaluationConfig(
        model_name=model_name,
        benchmark_path=benchmark_path,
        output_dir=output_dir,
        limit=limit,
        top_p=top_p,
        k=k,
        top_n=top_n,
        include_cot=include_cot,
        log_level=log_level,
        load_in_4bit=load_in_4bit,
        enable_caching=enable_caching,
    )

    logger.info(
        f"Final configuration: model={model_name}, benchmark={benchmark_path}, "
        f"k={k}, top_n={top_n}, top_p={top_p}, include_cot={include_cot}"
    )

    return config


def save_results(results: dict, output_dir: str, save_details: bool = False):
    """Save evaluation results to JSON files."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save summary statistics
    summary_path = os.path.join(output_dir, "summary_results.json")
    summary_data = {
        "statistics": results["statistics"],
        "config": results["config"],
        "timestamp": str(datetime.datetime.now()),
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
    print("\n" + "=" * 60)
    print("COMPARATIVE EVALUATION RESULTS")
    print("=" * 60)

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

        print(
            f"{method.title():<12} "
            f"{stats['accuracy']:.2f}%{'':<6} "
            f"{ci_str:<20} "
            f"{stats['total']:<8} "
            f"{stats.get('avg_time_ms', 0):.2f}"
        )

    print("-" * 60)
    print("✓ Random baseline for 4-option: 25.0%")
    print("✓ Charts saved to results/")
    print("=" * 60)


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    # Load YAML configuration first
    yaml_config = load_config_yaml(args.config)

    # Setup logging (use CLI override if provided, otherwise YAML config, otherwise default)
    log_level = (
        args.log_level if args.log_level else yaml_config.get("log_level", "INFO")
    )
    setup_logging(log_level=log_level)

    # Create evaluation configuration
    config = create_evaluation_config(yaml_config, args)

    logger.info("Starting comparative evaluation")
    logger.info(f"Configuration: {config}")

    try:
        # Load benchmark data
        benchmark_data = load_benchmark_data(config.benchmark_path)

        # Create evaluation runner
        runner = EvaluationRunner(config)

        # Filter samplers if specified
        if args.samplers:
            # Filter the samplers to only include requested ones
            available_samplers = {
                k: v for k, v in runner.samplers.items() if k in args.samplers
            }
            runner.samplers = available_samplers
            logger.info(f"Using samplers: {list(available_samplers.keys())}")

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
