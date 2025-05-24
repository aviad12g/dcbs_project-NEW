#!/usr/bin/env python3
"""
Download Winogrande 1.1 dataset from Hugging Face datasets and save as JSONL files.
"""

import argparse
import json
import os

from datasets import load_dataset


def download_and_save_winogrande(output_dir: str, splits: list = None) -> None:
    """Download Winogrande dataset and save each split as JSONL file.

    Args:
        output_dir: Directory to save the JSONL files
        splits: Which splits to download (default: all)
    """
    # Default to all splits if none specified
    if splits is None:
        splits = ["train", "dev", "test"]

    print(f"Loading Winogrande dataset from Hugging Face...")
    dataset = load_dataset("winogrande", "winogrande_xl")

    # Print available splits
    print(f"Available splits in dataset: {list(dataset.keys())}")

    os.makedirs(output_dir, exist_ok=True)

    # Save each requested split
    for split_name in splits:
        if split_name in dataset:
            split_data = dataset[split_name]
            output_file = os.path.join(output_dir, f"{split_name}.jsonl")

            print(f"Saving {len(split_data)} examples to {output_file}")
            with open(output_file, "w", encoding="utf-8") as f:
                for item in split_data:
                    # Convert to JSON and write one object per line
                    f.write(json.dumps(item) + "\n")

            print(f"✓ Saved {split_name} split")
        else:
            print(f"! Split {split_name} not found in dataset")

    print(f"Download complete. Files saved to {output_dir}")


def main():
    """Parse arguments and download dataset."""
    parser = argparse.ArgumentParser(
        description="Download Winogrande dataset from Hugging Face"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/winogrande",
        help="Directory to save the JSONL files",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        help="Which splits to download (default: all)",
    )

    args = parser.parse_args()
    download_and_save_winogrande(args.output_dir, args.splits)


if __name__ == "__main__":
    main()
