"""
Core evaluation logic shared across different evaluation scripts.

This package provides common functionality for model loading, evaluation,
and result processing to eliminate code duplication.
"""

from .config import EvaluationConfig
from .result import EvaluationResult
from .model_manager import ModelManager
from .template_manager import ChatTemplateManager
from .sampler_factory import SamplerFactory
from .example_processor import ExampleProcessor
from .runner import EvaluationRunner
from .utils import load_benchmark_data

__all__ = [
    "EvaluationConfig",
    "EvaluationResult", 
    "ModelManager",
    "ChatTemplateManager",
    "SamplerFactory",
    "ExampleProcessor",
    "EvaluationRunner",
    "load_benchmark_data",
] 