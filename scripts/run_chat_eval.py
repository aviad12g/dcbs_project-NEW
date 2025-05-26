#!/usr/bin/env python3
"""
Run chat-based evaluation for language models.

This script evaluates language models using chat templates and conversational prompts.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chat_eval import main as run_chat_eval


def main():
    """Parse arguments and run chat evaluation."""
    parser = argparse.ArgumentParser(
        description="Chat-based evaluation with chain-of-thought"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument(
        "--benchmark", type=str, required=True, help="Benchmark JSON file"
    )
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p value for nucleus sampling"
    )
    parser.add_argument("--k", type=int, default=8, help="Number of clusters for DCBS")
    parser.add_argument("--top-n", type=int, default=50, help="Top-n tokens for DCBS")
    parser.add_argument("--limit", type=int, help="Limit number of examples")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Run the evaluation
    exit_code = run_chat_eval(args)
    return exit_code


if __name__ == "__main__":
    sys.exit(main()) 