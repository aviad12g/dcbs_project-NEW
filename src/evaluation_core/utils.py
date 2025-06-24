"""
Utility functions for evaluation.

This module provides utility functions for loading and validating
benchmark data and other common operations.
"""

import json
import os
from typing import Dict, List

from src.errors import DataError, eval_logger as logger


def load_benchmark_data(benchmark_path: str) -> List[Dict]:
    """Load benchmark data with validation. Supports both file paths and dataset names."""
    logger.info(f"Loading benchmark: {benchmark_path}")

    # Check if this is a dataset name (like 'arc_challenge') rather than a file path
    if not os.path.exists(benchmark_path) and not benchmark_path.endswith('.json'):
        # Try to load as a dataset name
        try:
            from data_loaders import load_dataset
            logger.info(f"Loading dataset: {benchmark_path}")
            dataset_data = load_dataset(benchmark_path)
            
            # Convert to the expected format
            converted_data = []
            for item in dataset_data:
                converted_item = {
                    "id": item["id"],
                    "question": item["question"],
                    "options": item["choices"],
                    "correct_option": item["correct_option"],
                    "correct_answer": item["correct_answer"]
                }
                converted_data.append(converted_item)
            
            logger.info(f"Loaded {len(converted_data)} examples from dataset {benchmark_path}")
            return converted_data
            
        except ImportError:
            logger.error("data_loaders module not available")
            raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")
        except Exception as e:
            logger.error(f"Failed to load dataset {benchmark_path}: {e}")
            raise DataError(f"Error loading dataset {benchmark_path}: {e}")

    # Original file-based loading
    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    try:
        with open(benchmark_path, "r") as f:
            data = json.load(f)

        # Validate data structure
        if not isinstance(data, list):
            raise DataError("Benchmark data must be a list of examples")

        if len(data) == 0:
            raise DataError("Benchmark data is empty")

        # Validate first example has required fields (support both formats)
        first_example = data[0]

        # Check for ARC Easy format
        arc_fields = ["question", "options"]
        has_arc = all(field in first_example for field in arc_fields)

        if not has_arc:
            raise DataError(
                f"Benchmark examples must contain ARC Easy fields {arc_fields}"
            )

        # Log which format was detected
        logger.info(f"Detected ARC Easy format dataset")

        logger.info(f"Loaded {len(data)} examples")
        return data

    except json.JSONDecodeError as e:
        raise DataError(f"Invalid JSON in benchmark file: {e}")
    except Exception as e:
        raise DataError(f"Error loading benchmark data: {e}") 