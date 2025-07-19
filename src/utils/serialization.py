"""
Enhanced JSON serialization utilities for DCBS evaluation system.

This module provides comprehensive type conversion utilities to handle
numpy types, torch tensors, and other non-JSON-serializable objects
that commonly appear in machine learning evaluation results.
"""

import json
import logging
from typing import Any, Dict, List, Union, Optional
import numpy as np

# Optional torch import (may not be available in all environments)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class SerializationError(Exception):
    """Custom exception for serialization errors with detailed context."""
    
    def __init__(self, message: str, obj_type: str, obj_path: str = ""):
        self.obj_type = obj_type
        self.obj_path = obj_path
        super().__init__(f"{message} (type: {obj_type}, path: {obj_path})")


class SerializationUtils:
    """Utilities for converting objects to JSON-serializable format."""
    
    @staticmethod
    def convert_to_json_serializable(obj: Any, path: str = "root") -> Any:
        """
        Recursively convert objects to JSON-serializable format.
        
        Args:
            obj: Object to convert
            path: Current path in object hierarchy for error reporting
            
        Returns:
            JSON-serializable version of the object
            
        Raises:
            SerializationError: If object cannot be serialized
        """
        try:
            # Handle None
            if obj is None:
                return None
            
            # Handle numpy types FIRST (before basic types, since numpy types inherit from basic types)
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return SerializationUtils.handle_numpy_array(obj, path)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            
            # Handle basic JSON-serializable types
            if isinstance(obj, (str, int, float, bool)):
                return obj
            
            # Handle torch types if available
            if TORCH_AVAILABLE and SerializationUtils._is_torch_type(obj):
                return SerializationUtils.handle_torch_types(obj, path)
            
            # Handle dictionaries
            if isinstance(obj, dict):
                return {
                    key: SerializationUtils.convert_to_json_serializable(
                        value, f"{path}.{key}"
                    )
                    for key, value in obj.items()
                }
            
            # Handle lists and tuples
            if isinstance(obj, (list, tuple)):
                return [
                    SerializationUtils.convert_to_json_serializable(
                        item, f"{path}[{i}]"
                    )
                    for i, item in enumerate(obj)
                ]
            
            # Handle sets
            if isinstance(obj, set):
                return list(obj)
            
            # Handle dataclass objects
            if hasattr(obj, '__dataclass_fields__'):
                return SerializationUtils.convert_to_json_serializable(
                    obj.__dict__, f"{path}.<dataclass>"
                )
            
            # Handle objects with __dict__ attribute
            if hasattr(obj, '__dict__'):
                try:
                    obj_dict = obj.__dict__
                    if obj_dict is not None:
                        return SerializationUtils.convert_to_json_serializable(
                            obj_dict, f"{path}.<object>"
                        )
                    # If __dict__ is None, fall through to string conversion
                except Exception:
                    # If __dict__ access fails, fall through to string conversion
                    pass
            
            # Try to convert to string as last resort
            try:
                str_repr = str(obj)
                # Only warn for non-trivial objects (not basic object())
                if type(obj).__name__ != 'object':
                    logger.warning(
                        f"Converting unknown type {type(obj).__name__} to string at {path}: {str_repr[:100]}"
                    )
                return str_repr
            except Exception:
                raise SerializationError(
                    f"Cannot serialize object", 
                    type(obj).__name__, 
                    path
                )
                
        except SerializationError:
            raise
        except Exception as e:
            raise SerializationError(
                f"Unexpected error during serialization: {e}",
                type(obj).__name__,
                path
            )
    
    @staticmethod
    def handle_numpy_array(arr: np.ndarray, path: str) -> Union[List, Dict]:
        """
        Convert numpy array to JSON-serializable format.
        
        Args:
            arr: Numpy array to convert
            path: Current path for error reporting
            
        Returns:
            List representation of array with metadata
        """
        try:
            # For small arrays, convert to list directly
            if arr.size <= 1000:  # Configurable threshold
                return {
                    "type": "numpy_array",
                    "dtype": str(arr.dtype),
                    "shape": list(arr.shape),
                    "data": arr.tolist()
                }
            else:
                # For large arrays, provide summary information
                return {
                    "type": "numpy_array_large",
                    "dtype": str(arr.dtype),
                    "shape": list(arr.shape),
                    "size": int(arr.size),
                    "summary": {
                        "min": float(arr.min()) if arr.size > 0 else None,
                        "max": float(arr.max()) if arr.size > 0 else None,
                        "mean": float(arr.mean()) if arr.size > 0 else None,
                    }
                }
        except Exception as e:
            raise SerializationError(
                f"Failed to serialize numpy array: {e}",
                "numpy.ndarray",
                path
            )
    
    @staticmethod
    def handle_torch_types(obj: Any, path: str) -> Union[Dict, List, float, int]:
        """
        Convert torch tensors and other torch types to JSON-serializable format.
        
        Args:
            obj: Torch object to convert
            path: Current path for error reporting
            
        Returns:
            JSON-serializable representation
        """
        if not TORCH_AVAILABLE:
            raise SerializationError(
                "Torch not available but torch object encountered",
                type(obj).__name__,
                path
            )
        
        try:
            if torch.is_tensor(obj):
                # Store original properties before any modifications
                original_device = str(obj.device)
                original_requires_grad = obj.requires_grad
                
                # Convert tensor to numpy first, then handle as numpy array
                if obj.requires_grad:
                    # Detach gradient-tracking tensors
                    obj = obj.detach()
                
                # Move to CPU if on GPU
                if obj.device.type != 'cpu':
                    obj = obj.cpu()
                
                numpy_array = obj.numpy()
                result = SerializationUtils.handle_numpy_array(numpy_array, path)
                
                # Add torch-specific metadata
                if isinstance(result, dict):
                    result["original_type"] = "torch_tensor"
                    result["device"] = original_device
                    result["requires_grad"] = original_requires_grad
                
                return result
            
            elif isinstance(obj, torch.dtype):
                return str(obj)
            
            elif isinstance(obj, torch.device):
                return str(obj)
            
            else:
                # Try to convert to string
                return str(obj)
                
        except Exception as e:
            raise SerializationError(
                f"Failed to serialize torch object: {e}",
                type(obj).__name__,
                path
            )
    
    @staticmethod
    def _is_torch_type(obj: Any) -> bool:
        """Check if object is a torch type."""
        if not TORCH_AVAILABLE:
            return False
        
        return (
            torch.is_tensor(obj) or
            isinstance(obj, (torch.dtype, torch.device)) or
            type(obj).__module__.startswith('torch')
        )
    
    @staticmethod
    def safe_json_dump(obj: Any, file_path: str, **kwargs) -> None:
        """
        Safely dump object to JSON file with enhanced serialization.
        
        Args:
            obj: Object to serialize
            file_path: Path to output file
            **kwargs: Additional arguments for json.dump
        """
        try:
            serializable_obj = SerializationUtils.convert_to_json_serializable(obj)
            
            with open(file_path, 'w') as f:
                json.dump(serializable_obj, f, **kwargs)
                
        except SerializationError as e:
            logger.error(f"Serialization failed for {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error writing to {file_path}: {e}")
            raise
    
    @staticmethod
    def safe_json_dumps(obj: Any, **kwargs) -> str:
        """
        Safely convert object to JSON string with enhanced serialization.
        
        Args:
            obj: Object to serialize
            **kwargs: Additional arguments for json.dumps
            
        Returns:
            JSON string representation
        """
        try:
            serializable_obj = SerializationUtils.convert_to_json_serializable(obj)
            return json.dumps(serializable_obj, **kwargs)
            
        except SerializationError as e:
            logger.error(f"Serialization failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during JSON serialization: {e}")
            raise


def convert_numpy_types(obj: Any) -> Any:
    """
    Legacy function for backward compatibility.
    
    This function maintains compatibility with existing code that uses
    the convert_numpy_types function from compare_methods.py.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    return SerializationUtils.convert_to_json_serializable(obj)


# Convenience functions for common use cases
def serialize_evaluation_results(results: Dict) -> Dict:
    """
    Serialize evaluation results with specific handling for common structures.
    
    Args:
        results: Evaluation results dictionary
        
    Returns:
        JSON-serializable results dictionary
    """
    return SerializationUtils.convert_to_json_serializable(results)


def serialize_checkpoint_state(state: Any) -> Dict:
    """
    Serialize checkpoint state with error recovery.
    
    Args:
        state: Checkpoint state object
        
    Returns:
        JSON-serializable state dictionary
    """
    try:
        return SerializationUtils.convert_to_json_serializable(state)
    except SerializationError as e:
        logger.error(f"Failed to serialize checkpoint state: {e}")
        # Return minimal state for recovery
        return {
            "serialization_error": str(e),
            "error_type": e.obj_type,
            "error_path": e.obj_path,
            "timestamp": getattr(state, 'timestamp', None),
            "run_id": getattr(state, 'run_id', None),
        }


def safe_json_dump(obj: Any, file_path: str, **kwargs) -> None:
    """
    Safely dump object to JSON file with enhanced serialization.
    
    This is an alias for SerializationUtils.safe_json_dump for backward compatibility.
    
    Args:
        obj: Object to serialize
        file_path: Path to output file
        **kwargs: Additional arguments for json.dump
    """
    return SerializationUtils.safe_json_dump(obj, file_path, **kwargs)