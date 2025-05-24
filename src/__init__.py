"""
DCBS evaluation framework.

This package contains the evaluation framework for comparing
Disjunctive Category Beam Search (DCBS) against other sampling methods
on language model tasks.
"""

__version__ = "1.0.0"

# Expose main classes and functions for easy importing
from .evaluation_core import EvaluationConfig, EvaluationRunner, load_benchmark_data
from .visualization import generate_all_visualizations
from .errors import setup_logging, eval_logger
