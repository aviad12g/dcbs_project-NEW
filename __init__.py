"""
DCBS Evaluation - Deterministic Category Based Sampling evaluation harness.

This package provides a comprehensive framework for evaluating different
sampling methods on multiple-choice reasoning tasks.
"""

__version__ = "0.1.0"

# Re-export main classes and functions
try:
    from src.evaluation_core import EvaluationConfig, EvaluationRunner
    from src.visualization import generate_all_visualizations

    __all__ = ["EvaluationRunner", "EvaluationConfig", "generate_all_visualizations"]

except ImportError:
    # Graceful handling if modules aren't available
    __all__ = []
