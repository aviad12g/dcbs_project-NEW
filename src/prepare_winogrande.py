#!/usr/bin/env python3
"""
Prepare the Winogrande benchmark for DCBS evaluation.
Loads examples from the Winogrande validation set.
"""

import os
import json
import argparse
from datasets import load_dataset


def process_example(example, idx):
    """Process a single Winogrande example for evaluation."""
    sentence_with_blank = example["sentence"]
    option1 = example["option1"]
    option2 = example["option2"]

    correct_option = "1" if example["answer"] == "1" else "2"

    example_id = example.get("id", f"wino_{idx}")

    return {
        "id": example_id,
        "sentence": sentence_with_blank,
        "option1": option1,
        "option2": option2,
        "correct_option": correct_option,
    }


def prepare(num_examples=None, seed=42, output_file="data/bench_wino.json"):
    """Prepare Winogrande benchmark data from the official validation split."""
    print(f"Preparing Winogrande benchmark data...")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print("Loading Winogrande XL validation dataset...")
    dataset = load_dataset("winogrande", "winogrande_xl", split="validation")
    print(f"Loaded {len(dataset)} examples from Winogrande XL validation set")

    dataset = dataset.shuffle(seed=seed)

    subset = dataset.select(range(num_examples or len(dataset)))

    processed_examples = [process_example(ex, idx) for idx, ex in enumerate(subset)]

    with open(output_file, "w") as f:
        json.dump(processed_examples, f, indent=2)

    print(f"Processed examples saved to {output_file}")
    print(f"Number of examples: {len(processed_examples)}")

    if processed_examples:
        print("\nSample example:")
        print(json.dumps(processed_examples[0], indent=2))

    return processed_examples


def main():
    """Main function to prepare the Winogrande benchmark data."""
    parser = argparse.ArgumentParser(description="Prepare Winogrande benchmark data")
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="Number of examples to include in the benchmark (default: use all 1,267 examples)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling (default: 42)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/bench_wino.json",
        help="Path to save the processed examples (default: data/bench_wino.json)",
    )

    args = parser.parse_args()

    prepare(
        num_examples=args.num_examples, seed=args.seed, output_file=args.output_file
    )


if __name__ == "__main__":
    main()
