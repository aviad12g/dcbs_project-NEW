"""
Core evaluation logic shared across different evaluation scripts.

This package provides common functionality for model loading, evaluation,
and result processing to eliminate code duplication.
"""

from .config import EvaluationConfig
from .result import EvaluationResult
from .sampler_factory import SamplerFactory
from .example_processor import ExampleProcessor
from .runner import EvaluationRunner
from .utils import load_benchmark_data

# Import improved components
try:
    from .improved_model_manager import ImprovedModelManager
    from .improved_example_processor import ImprovedExampleProcessor  
    from .improved_runner import ImprovedEvaluationRunner
except ImportError:
    # Gracefully handle if improved components aren't available
    pass

# Keep ModelManager import conditional to avoid ChatTemplateManager dependency
try:
    from .model_manager import ModelManager
except ImportError:
    # Use improved version if original fails
    try:
        from .improved_model_manager import ImprovedModelManager as ModelManager
    except ImportError:
        pass

__all__ = [
    "EvaluationConfig",
    "EvaluationResult", 
    "ModelManager",
    "SamplerFactory",
    "ExampleProcessor",
    "EvaluationRunner",
    "load_benchmark_data",
] 