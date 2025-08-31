"""
Messages Completion Module

A clean interface for batched message completion with configurable sampling methods.
Exports only two main classes: CompletionConfig and MessageCompleter.
"""

from .config import CompletionConfig
from .completer import MessageCompleter

__version__ = "1.0.0"

__all__ = [
    "CompletionConfig",
    "MessageCompleter",
]