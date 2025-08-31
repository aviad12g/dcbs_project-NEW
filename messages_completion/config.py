"""
Configuration class for message completion.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
from enum import Enum


class SamplingMethod(Enum):
    """Supported sampling methods."""
    GREEDY = "greedy"
    TOP_P = "top_p"
    DCBS = "dcbs"


@dataclass
class CompletionConfig:
    """Configuration for message completion.
    
    This is the main configuration class that controls all aspects of completion:
    model selection, sampling method, generation parameters, and output options.
    """
    
    # Model configuration
    model_name: str
    device: Optional[str] = None  # Auto-detect if None
    load_in_4bit: bool = False
    
    # Generation parameters
    max_new_tokens: int = 50
    sampling_method: Union[SamplingMethod, str] = SamplingMethod.GREEDY
    

    
    # Sampling-specific parameters
    sampling_params: Optional[Dict[str, Any]] = None
    
    # Output configuration
    return_logprobs: bool = False
    return_token_ids: bool = True
    
    # Performance settings
    batch_size: Optional[int] = None  # Auto-determine if None
    deterministic: bool = True  # Force deterministic generation
    
    def __post_init__(self):
        """Validate and normalize configuration."""
        # Convert string to enum if needed
        if isinstance(self.sampling_method, str):
            try:
                self.sampling_method = SamplingMethod(self.sampling_method.lower())
            except ValueError:
                raise ValueError(f"Unsupported sampling method: {self.sampling_method}")
        
        # Set default sampling parameters
        if self.sampling_params is None:
            self.sampling_params = {}
        
        # Validate sampling parameters based on method
        self._validate_sampling_params()
    
    def _validate_sampling_params(self):
        """Validate sampling parameters for the chosen method."""
        method = self.sampling_method
        params = self.sampling_params
        
        if method == SamplingMethod.GREEDY:
            # Greedy sampling needs no parameters
            pass
            
        elif method == SamplingMethod.TOP_P:
            # Validate top-p parameters
            if "p" not in params:
                params["p"] = 0.9
            if "temperature" not in params:
                params["temperature"] = 1.0
                
            if not 0.0 < params["p"] <= 1.0:
                raise ValueError(f"top_p must be in (0, 1], got {params['p']}")
            if params["temperature"] <= 0:
                raise ValueError(f"temperature must be > 0, got {params['temperature']}")
                
        elif method == SamplingMethod.DCBS:
            # Validate and set DCBS parameters with explicit defaults
            self._validate_dcbs_params(params)
    
    def _validate_dcbs_params(self, params: Dict[str, Any]):
        """Validate DCBS-specific parameters."""
        # Core DCBS parameters
        if "k" not in params:
            params["k"] = 8
        if "top_n" not in params:
            params["top_n"] = 50
        if "clustering_method" not in params:
            params["clustering_method"] = "dbscan"
        
        # Deterministic DCBS parameters
        if "weighted" not in params:
            params["weighted"] = False  # False = deterministic
        if "levels" not in params:
            params["levels"] = 1
        if "tie_break" not in params:
            params["tie_break"] = "min_id"  # or "max_prob"
        if "seed" not in params:
            params["seed"] = None
        
        # Optional frozen clusters path
        if "assignments_path" not in params:
            params["assignments_path"] = None
        
        # Validate values
        if params["k"] <= 0:
            raise ValueError(f"DCBS k must be > 0, got {params['k']}")
        if params["top_n"] <= 0:
            raise ValueError(f"DCBS top_n must be > 0, got {params['top_n']}")
        if params["levels"] <= 0:
            raise ValueError(f"DCBS levels must be > 0, got {params['levels']}")
        if params["tie_break"] not in ["min_id", "max_prob"]:
            raise ValueError(f"DCBS tie_break must be 'min_id' or 'max_prob', got {params['tie_break']}")
        if not isinstance(params["weighted"], bool):
            raise ValueError(f"DCBS weighted must be boolean, got {params['weighted']}")
        
        # Validate clustering method
        valid_methods = ["dbscan", "kmeans", "hierarchical"]
        if params["clustering_method"] not in valid_methods:
            raise ValueError(f"DCBS clustering_method must be one of {valid_methods}, got {params['clustering_method']}")
        
        # Validate assignments_path if provided
        if params["assignments_path"] is not None:
            import os
            if not os.path.exists(params["assignments_path"]):
                raise ValueError(f"DCBS assignments_path does not exist: {params['assignments_path']}")
    
    @property
    def is_deterministic(self) -> bool:
        """Check if current configuration produces deterministic results."""
        if not self.deterministic:
            return False
        return self.sampling_method in [SamplingMethod.GREEDY, SamplingMethod.DCBS]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "load_in_4bit": self.load_in_4bit,
            "max_new_tokens": self.max_new_tokens,
            "sampling_method": self.sampling_method.value,
            "sampling_params": self.sampling_params.copy(),
            "return_logprobs": self.return_logprobs,
            "return_token_ids": self.return_token_ids,
            "batch_size": self.batch_size,
            "deterministic": self.deterministic,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CompletionConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return (f"CompletionConfig(model={self.model_name}, "
                f"sampling={self.sampling_method.value}, "
                f"max_tokens={self.max_new_tokens}, "
                f"deterministic={self.deterministic})")