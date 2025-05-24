#!/usr/bin/env python3
"""
Convert Winogrande 1.1 JSONL format to the JSON format expected by the DCBS evaluation framework.
Usage: python -m src.convert_winogrande input.jsonl output.json
"""

import json
import argparse
import os
from typing import List, Dict, Any


def convert_winogrande_format(input_jsonl: str, output_json: str) -> None:
    """Convert Winogrande 1.1 JSONL format to DCBS evaluation JSON format.

    Args:
        input_jsonl: Path to input JSONL file (Winogrande dev/train/test)
        output_json: Path to output JSON file for DCBS evaluation
    """
    # Read JSONL file (one JSON object per line)
    examples = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line.strip()))

    # Create formatted output
    output_examples = []
    for i, example in enumerate(examples):
        # Extract the sentence parts and options from Winogrande format
        sentence = example.get("sentence", "")
        option1 = example.get("option1", "")
        option2 = example.get("option2", "")

        # Get the correct answer (if available - test set won't have it)
        correct_option = example.get("answer", "1")

        # Format for DCBS evaluation
        formatted_example = {
            "id": example.get("qID", f"winogrande_{i}"),
            "sentence": sentence,
            "option1": option1,
            "option2": option2,
            "correct_option": correct_option,
        }

        output_examples.append(formatted_example)

    # Write to JSON file
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_examples, f, indent=2)

    print(
        f"Converted {len(output_examples)} examples from {input_jsonl} to {output_json}"
    )


def main():
    """Parse arguments and convert file."""
    parser = argparse.ArgumentParser(
        description="Convert Winogrande JSONL to DCBS evaluation JSON"
    )
    parser.add_argument("input_jsonl", type=str, help="Path to input JSONL file")
    parser.add_argument("output_json", type=str, help="Path to output JSON file")

    args = parser.parse_args()
    convert_winogrande_format(args.input_jsonl, args.output_json)


if __name__ == "__main__":
    main()
