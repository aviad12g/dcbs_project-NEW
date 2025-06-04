"""Utility metrics for evaluation results."""

from typing import Tuple

import numpy as np


def calculate_confidence_interval(correct: int, total: int) -> Tuple[float, float]:
    """Calculate Wilson score interval for accuracy percentage."""
    if total == 0:
        return (0.0, 0.0)
    p = correct / total
    z = 1.96  # 95% confidence
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    return (max(0, center - margin) * 100, min(100, center + margin) * 100)
