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
    """Load benchmark data with validation."""
    logger.info(f"Loading benchmark: {benchmark_path}")

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