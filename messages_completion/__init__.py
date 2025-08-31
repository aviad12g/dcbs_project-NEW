"""
messages_completion: given a list of conversations, return completions 
(deterministic, batch-invariant), with optional token IDs/logprobs.
"""

from .completer import MessageCompleter
from .message_processor import MessageProcessor, MessageBatch
from .output_types import TokenInfo, CompletionResult, BatchCompletionResult
from .model_interface import ModelInterface, HuggingFaceModelInterface
from .completion_engine import CompletionEngine, CompletionConfig
from .sampling_interface import SamplingInterface, DCBSSamplingInterface

__version__ = "1.0.0"

__all__ = [
    "MessageCompleter",
    "MessageProcessor", "MessageBatch",
    "TokenInfo", "CompletionResult", "BatchCompletionResult",
    "ModelInterface", "HuggingFaceModelInterface",
    "CompletionEngine", "CompletionConfig",
    "SamplingInterface", "DCBSSamplingInterface",
]