"""Factory function for creating samplers.

This module now only provides an explicit sampler for DCBS. Standard
methods like greedy and top-p are handled directly by model.generate
in higher-level orchestration (e.g., MessageCompleter), so creating
dedicated sampler objects for them is unnecessary here.
"""

from typing import Dict, Any, Optional
from enum import Enum

from .base import Sampler
from .dcbs import DCBSSampler


class SamplingMethod(Enum):
    """Supported sampling methods for explicit sampler creation."""
    DCBS = "dcbs"


def create_sampler(method: str, parameters: Optional[Dict[str, Any]] = None) -> Optional[Sampler]:
    """Create an explicit sampler instance when needed.

    - For "dcbs": returns a DCBSSampler instance.
    - For standard methods ("greedy", "top_p"/"nucleus"): returns None,
      indicating that callers should use model.generate directly.
    """
    if parameters is None:
        parameters = {}

    method = method.lower()

    if method == "dcbs":
        return DCBSSampler(**parameters)
    if method in ("greedy", "top_p", "nucleus"):
        return None
    raise ValueError(f"Unknown sampling method: {method}")


def get_available_methods() -> list[str]:
    """Get list of methods with explicit sampler implementations."""
    return ["dcbs"]
