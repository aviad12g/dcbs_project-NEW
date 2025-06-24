"""
Core evaluation logic for DCBS evaluation framework.

This package provides the main functionality for model loading, evaluation,
and result processing with proper conversation flow and fixed parameter handling.
"""

from .config import EvaluationConfig
from .result import EvaluationResult
from .sampler_factory import SamplerFactory
from .utils import load_benchmark_data
from .model_manager import ModelManager
from .example_processor import ExampleProcessor
from .runner import EvaluationRunner
from .message_templates import MessageTemplateGenerator
from .token_generator import TokenGenerator
from .question_answerer import QuestionAnswerer
from .disagreement_tracker import DisagreementTracker, DisagreementAwareQuestionAnswerer
from .multi_dataset_runner import MultiDatasetEvaluationRunner

__all__ = [
    "EvaluationConfig",
    "EvaluationResult",
    "ModelManager",
    "SamplerFactory",
    "ExampleProcessor",
    "EvaluationRunner",
    "MultiDatasetEvaluationRunner",
    "load_benchmark_data",
    "MessageTemplateGenerator",
    "TokenGenerator", 
    "QuestionAnswerer",
    "DisagreementTracker",
    "DisagreementAwareQuestionAnswerer",
] 