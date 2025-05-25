#!/usr/bin/env python3
"""
DCBS Evaluation main entry script.

This script runs the evaluation of Disjunctive Category Beam Search (DCBS) against
other sampling methods (greedy, top-p, random) on Winogrande examples.
All methods are filtered to only allowed answer tokens.
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Set


def main():
    """Main entry point for the DCBS evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate DCBS against other sampling methods"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/study_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--model_name", type=str, help="Model name/path (overrides config)"
    )
    parser.add_argument(
        "--benchmark", type=str, help="Path to benchmark file (overrides config)"
    )
    parser.add_argument(
        "--out_csv", type=str, help="Path to output CSV file (overrides config)"
    )
    parser.add_argument(
        "--inject_reasoning",
        action="store_true",
        help="Inject chain-of-thought prompting (enabled by default)",
    )
    parser.add_argument(
        "--limit", type=int, help="Limit evaluation to this many examples"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        help="Path to log file (if not specified, logs to console only)",
    )

    args = parser.parse_args()

    # Import the run_dcbs_eval module from src
    try:
        # Import and use the main function from src.run_dcbs_eval
        from src.run_dcbs_eval import main as run_eval

        # Pass the args object directly
        exit_code = run_eval(args)
        sys.exit(exit_code)
    except ModuleNotFoundError:
        print("Error: Could not import the evaluation module.")
        print("Make sure you're running this script from the project root directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running evaluation: {e}")
        # Print more detailed error information
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
