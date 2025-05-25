#!/usr/bin/env python3
"""
Unified DCBS Evaluation Framework
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

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
    parser = argparse.ArgumentParser(
        description="Unified DCBS evaluation framework with advanced features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core configuration
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

    # Evaluation parameters
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

    # Advanced features
    parser.add_argument(
        "--no-cot",
        action="store_true",
        help="Disable chain-of-thought reasoning (overrides config)",
    )

    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable DCBS caching for performance comparison",
    )

    parser.add_argument(
        "--sweep-k",
        nargs="+",
        type=int,
        help="Sweep over multiple k values for DCBS",
    )

    parser.add_argument(
        "--sweep-top-n",
        nargs="+",
        type=int,
        help="Sweep over multiple top-n values for DCBS",
    )

    parser.add_argument(
        "--sweep-top-p",
        nargs="+",
        type=float,
        help="Sweep over multiple top-p values",
    )

    # Output and logging
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (overrides config)",
    )

    parser.add_argument(
        "--save-details", 
        action="store_true", 
        help="Save detailed per-example results"
    )

    parser.add_argument(
        "--output-format",
        choices=["json", "csv", "both"],
        default="json",
        help="Output format for results",
    )

    # Performance options
    parser.add_argument(
        "--load-in-4bit", 
        action="store_true", 
        help="Load model with 4-bit quantization"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Batch size for processing examples",
    )

    parser.add_argument(
        "--memory-profiling",
        action="store_true",
        help="Enable detailed memory profiling",
    )

    # Sampler selection
    parser.add_argument(
        "--samplers",
        nargs="+",
        choices=["greedy", "top-p", "dcbs", "random"],
        help="Specify which samplers to evaluate",
    )

    return parser.parse_args()


def create_evaluation_config(
    yaml_config: dict, args: argparse.Namespace
) -> EvaluationConfig:
    # Start with YAML config values, use reasonable defaults if missing
    model_name = yaml_config.get("model_path", "meta-llama/Llama-3.2-1B")
    benchmark_path = yaml_config.get("benchmark", "data/arc_easy_full.json")
    output_dir = "results"
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
    include_cot = True
    log_level = yaml_config.get("log_level", "INFO")
    load_in_4bit = False
    enable_caching = True

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
        f"k={k}, top_n={top_n}, top_p={top_p}, include_cot={include_cot}, "
        f"enable_caching={enable_caching}"
    )

    return config


def run_parameter_sweep(
    base_config: EvaluationConfig,
    benchmark_data: List[dict],
    args: argparse.Namespace,
) -> Dict:
    # Determine parameter ranges
    k_values = args.sweep_k if args.sweep_k else [base_config.k]
    top_n_values = args.sweep_top_n if args.sweep_top_n else [base_config.top_n]
    top_p_values = args.sweep_top_p if args.sweep_top_p else [base_config.top_p]
    
    all_results = []
    
    logger.info(f"Running parameter sweep: k={k_values}, top_n={top_n_values}, top_p={top_p_values}")
    
    for k in k_values:
        for top_n in top_n_values:
            for top_p in top_p_values:
                logger.info(f"Evaluating with k={k}, top_n={top_n}, top_p={top_p}")
                
                sweep_config = EvaluationConfig(
                    model_name=base_config.model_name,
                    benchmark_path=base_config.benchmark_path,
                    output_dir=base_config.output_dir,
                    limit=base_config.limit,
                    top_p=top_p,
                    k=k,
                    top_n=top_n,
                    include_cot=base_config.include_cot,
                    log_level=base_config.log_level,
                    load_in_4bit=base_config.load_in_4bit,
                    enable_caching=base_config.enable_caching,
                )
                
                runner = EvaluationRunner(sweep_config)
                
                if args.samplers:
                    available_samplers = {
                        k: v for k, v in runner.samplers.items() if k in args.samplers
                    }
                    runner.samplers = available_samplers
                
                results = runner.run_evaluation(benchmark_data)
                
                results["sweep_params"] = {"k": k, "top_n": top_n, "top_p": top_p}
                all_results.append(results)
    
    return {"sweep_results": all_results, "base_config": base_config.__dict__}


def save_results(results: Dict, output_dir: str, args: argparse.Namespace):
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.output_format in ["json", "both"]:
        json_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"JSON results saved to: {json_path}")

    if args.output_format in ["csv", "both"]:
        csv_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.csv")
        # TODO: Implement CSV export functionality
        logger.info(f"CSV export not yet implemented: {csv_path}")

    if args.save_details:
        details_path = os.path.join(output_dir, f"detailed_results_{timestamp}.json")
        with open(details_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Detailed results saved to: {details_path}")


def print_summary(results: Dict, is_sweep: bool = False):
    print("\n" + "=" * 70)
    print("UNIFIED DCBS EVALUATION RESULTS")
    print("=" * 70)

    if is_sweep:
        print("PARAMETER SWEEP RESULTS")
        print("-" * 70)
        
        for i, sweep_result in enumerate(results["sweep_results"]):
            params = sweep_result["sweep_params"]
            statistics = sweep_result["statistics"]
            
            print(f"\nConfiguration {i+1}: k={params['k']}, top_n={params['top_n']}, top_p={params['top_p']}")
            print(f"{'Method':<12} {'Accuracy':<12} {'Correct/Total':<15} {'Time (ms)':<12}")
            print("-" * 60)
            
            method_stats = [(method, stats) for method, stats in statistics.items()]
            method_stats.sort(key=lambda x: x[1]["accuracy"], reverse=True)
            
            for method, stats in method_stats:
                print(
                    f"{method.title():<12} "
                    f"{stats['accuracy']:.2f}%{'':<6} "
                    f"{stats['correct']}/{stats['total']:<10} "
                    f"{stats.get('avg_time_ms', 0):.2f}"
                )
    else:
        statistics = results["statistics"]
        config = results["config"]

        print(f"Model: {config['model']}")
        print(f"Total Examples: {config['total_examples']}")
        print(f"Methods: {', '.join(config['methods'])}")
        print(f"Caching: {'Enabled' if config.get('enable_caching', True) else 'Disabled'}")
        print("-" * 70)

        method_stats = [(method, stats) for method, stats in statistics.items()]
        method_stats.sort(key=lambda x: x[1]["accuracy"], reverse=True)

        print(f"{'Method':<12} {'Accuracy':<12} {'95% CI':<20} {'N':<8} {'Time (ms)':<12}")
        print("-" * 70)

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

    print("-" * 70)
    print("✓ Random baseline for 4-option: 25.0%")
    print("✓ Charts saved to results/")
    print("=" * 70)


def main():
    args = parse_arguments()

    yaml_config = load_config_yaml(args.config)

    log_level = (
        args.log_level if args.log_level else yaml_config.get("log_level", "INFO")
    )
    setup_logging(log_level=log_level)

    config = create_evaluation_config(yaml_config, args)

    logger.info("Starting unified DCBS evaluation framework")
    logger.info(f"Configuration: {config}")

    try:
        benchmark_data = load_benchmark_data(config.benchmark_path)

        is_sweep = any([args.sweep_k, args.sweep_top_n, args.sweep_top_p])
        
        if is_sweep:
            logger.info("Running parameter sweep evaluation...")
            results = run_parameter_sweep(config, benchmark_data, args)
        else:
            logger.info("Running single configuration evaluation...")
            runner = EvaluationRunner(config)

            if args.samplers:
                available_samplers = {
                    k: v for k, v in runner.samplers.items() if k in args.samplers
                }
                runner.samplers = available_samplers
                logger.info(f"Using samplers: {list(available_samplers.keys())}")

            results = runner.run_evaluation(benchmark_data)

        save_results(results, config.output_dir, args)

        if not is_sweep:
            logger.info("Generating visualizations...")
            generate_all_visualizations(results, config.output_dir)

        print_summary(results, is_sweep)

        logger.info("Unified evaluation completed successfully!")

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main() 