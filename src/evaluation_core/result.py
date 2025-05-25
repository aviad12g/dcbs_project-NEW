"""
Result classes for evaluation data.

This module defines the result dataclass used to store
evaluation outcomes for individual examples.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class EvaluationResult:
    """Results from evaluating a single example with a sampler."""

    example_id: str
    method: str
    correct: bool
    elapsed_ms: float
    pred_id: int
    predicted_answer: str
    cot_reasoning: Optional[str] = None
    answer_probs: Optional[Dict[str, float]] = None 