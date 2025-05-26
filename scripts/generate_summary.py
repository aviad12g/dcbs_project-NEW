#!/usr/bin/env python3
"""
Generate a final summary of DCBS evaluation results in Markdown format.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.final_summary import generate_summary


def main():
    """Parse arguments and generate summary."""
    parser = argparse.ArgumentParser(
        description="Generate summary of DCBS evaluation results"
    )
    parser.add_argument(
        "--input_csv", type=str, required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--out_file", type=str, required=True, help="Path to output markdown file"
    )

    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    
    # Generate the summary
    generate_summary(args.input_csv, args.out_file)
    
    print(f"Summary report generated at {args.out_file}")
    return 0


if __name__ == "__main__":
    main() 