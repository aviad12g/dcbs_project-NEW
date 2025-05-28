#!/usr/bin/env python3
"""
Run chat-based evaluation for language models.

This script evaluates language models using the unified DCBS evaluation framework.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from compare_methods import EvaluationFramework
from src.evaluation_core.config import EvaluationConfig


def main():
    """Parse arguments and run chat evaluation using unified framework."""
    parser = argparse.ArgumentParser(
        description="Chat-based evaluation with chain-of-thought using unified framework"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument(
        "--benchmark", type=str, required=True, help="Benchmark JSON file"
    )
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p value for nucleus sampling"
    )
    parser.add_argument("--k", type=int, default=8, help="Number of clusters for DCBS")
    parser.add_argument("--top-n", type=int, default=50, help="Top-n tokens for DCBS")
    parser.add_argument("--limit", type=int, help="Limit number of examples")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model with 4-bit quantization")
    parser.add_argument("--output-format", choices=["json", "csv", "both"], default="both", help="Output format")

    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Create evaluation configuration
    config = EvaluationConfig(
        model_name=args.model,
        benchmark_path=args.benchmark,
        output_dir=args.output,
        limit=args.limit,
        top_p=args.top_p,
        k=args.k,
        top_n=args.top_n,
        log_level=args.log_level,
        load_in_4bit=args.load_in_4bit,
        include_cot=True,  # Chat evaluation typically uses CoT
        enable_caching=True
    )
    
    # Create a mock args object for the framework
    class MockArgs:
        def __init__(self, config, output_format):
            self.config = None  # Use direct config
            self.log_level = config.log_level
            self.output_format = output_format
            self.save_details = True
            self.samplers = None  # Use all samplers
            self.sweep_k = None
            self.sweep_top_n = None
            self.sweep_top_p = None
    
    # Create framework with mock args
    framework = EvaluationFramework()
    framework.config = config
    framework.args = MockArgs(config, args.output_format)
    
    try:
        # Run the evaluation
        framework.run()
        print(f"Evaluation completed successfully! Results saved to {args.output}")
        return 0
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 