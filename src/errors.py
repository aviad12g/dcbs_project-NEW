"""
DCBS evaluation error handling utilities.

This module provides custom exception classes and error handling utilities
for the DCBS evaluation framework.
"""

import logging
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create the main logger
logger = logging.getLogger("dcbs")

# Component-specific loggers
eval_logger = logging.getLogger("dcbs.eval")
dcbs_logger = logging.getLogger("dcbs.algorithm")
vis_logger = logging.getLogger("dcbs.visualization")


class DCBSError(Exception):
    """Base exception class for DCBS errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, recoverable: bool = False):
        """Initialize error with message and optional details.

        Args:
            message: Error description
            details: Additional context about the error
            recoverable: Whether this error can be recovered from
        """
        self.message = message
        self.details = details or {}
        self.recoverable = recoverable
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "recoverable": self.recoverable,
        }


class DCBSRuntimeError(DCBSError):
    """Exception raised for errors during DCBS execution."""
    pass


class EvaluationError(DCBSError):
    """Exception raised for errors during evaluation."""
    pass


class ConfigurationError(DCBSError):
    """Exception raised for configuration errors."""
    pass


class VisualizationError(DCBSError):
    """Exception raised for errors during visualization."""
    pass


class DataError(DCBSError):
    """Exception raised for errors related to benchmark data."""
    pass


class ModelLoadError(DCBSError):
    """Exception raised for errors during model loading."""
    pass


class SamplingError(DCBSError):
    """Exception raised for errors during token sampling."""
    pass


class CacheError(DCBSError):
    """Exception raised for errors related to caching operations."""
    pass


class ValidationError(DCBSError):
    """Exception raised for validation failures."""
    pass


class ResourceError(DCBSError):
    """Exception raised for resource-related errors (memory, disk, etc.)."""
    pass


class TimeoutError(DCBSError):
    """Exception raised when operations exceed time limits."""
    pass


class ErrorRecoveryManager:
    """Manages error recovery strategies for different types of failures."""

    def __init__(self, logger_instance: logging.Logger = logger):
        self.logger = logger_instance
        self.recovery_strategies = {
            ModelLoadError: self._recover_model_load,
            CacheError: self._recover_cache_error,
            ResourceError: self._recover_resource_error,
            SamplingError: self._recover_sampling_error,
            DataError: self._recover_data_error,
        }

    def attempt_recovery(self, error: DCBSError, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Attempt to recover from an error.
        
        Args:
            error: The error to recover from
            context: Additional context for recovery
            
        Returns:
            True if recovery was successful, False otherwise
        """
        if not error.recoverable:
            self.logger.warning(f"Error {type(error).__name__} is not recoverable")
            return False

        error_type = type(error)
        if error_type in self.recovery_strategies:
            try:
                return self.recovery_strategies[error_type](error, context or {})
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed: {recovery_error}")
                return False
        else:
            self.logger.warning(f"No recovery strategy for {error_type.__name__}")
            return False

    def _recover_model_load(self, error: ModelLoadError, context: Dict[str, Any]) -> bool:
        """Attempt to recover from model loading errors."""
        self.logger.info("Attempting model load recovery...")
        
        # Try loading with reduced precision
        if "load_in_4bit" not in context or not context["load_in_4bit"]:
            self.logger.info("Retrying with 4-bit quantization")
            context["load_in_4bit"] = True
            return True
            
        # Try CPU fallback
        if context.get("device") != "cpu":
            self.logger.info("Falling back to CPU")
            context["device"] = "cpu"
            return True
            
        return False

    def _recover_cache_error(self, error: CacheError, context: Dict[str, Any]) -> bool:
        """Attempt to recover from cache errors."""
        self.logger.info("Attempting cache recovery...")
        
        # Clear cache and disable caching
        try:
            import shutil
            cache_dir = context.get("cache_dir", ".dcbs_cache")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                self.logger.info(f"Cleared cache directory: {cache_dir}")
            
            context["enable_caching"] = False
            self.logger.info("Disabled caching for this session")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return False

    def _recover_resource_error(self, error: ResourceError, context: Dict[str, Any]) -> bool:
        """Attempt to recover from resource errors."""
        self.logger.info("Attempting resource recovery...")
        
        # Reduce batch size
        current_batch_size = context.get("batch_size", 1)
        if current_batch_size > 1:
            new_batch_size = max(1, current_batch_size // 2)
            context["batch_size"] = new_batch_size
            self.logger.info(f"Reduced batch size from {current_batch_size} to {new_batch_size}")
            return True
            
        # Force garbage collection
        import gc
        gc.collect()
        self.logger.info("Forced garbage collection")
        return True

    def _recover_sampling_error(self, error: SamplingError, context: Dict[str, Any]) -> bool:
        """Attempt to recover from sampling errors."""
        self.logger.info("Attempting sampling recovery...")
        
        # Fall back to greedy sampling
        if context.get("sampler_type") != "greedy":
            context["sampler_type"] = "greedy"
            self.logger.info("Falling back to greedy sampling")
            return True
            
        return False

    def _recover_data_error(self, error: DataError, context: Dict[str, Any]) -> bool:
        """Attempt to recover from data errors."""
        self.logger.info("Attempting data recovery...")
        
        # Skip problematic examples
        if "skip_invalid" not in context:
            context["skip_invalid"] = True
            self.logger.info("Enabled skipping of invalid examples")
            return True
            
        return False


def with_error_recovery(recovery_manager: Optional[ErrorRecoveryManager] = None):
    """Decorator for functions that should attempt error recovery."""
    if recovery_manager is None:
        recovery_manager = ErrorRecoveryManager()
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            max_retries = kwargs.pop("max_retries", 3)
            retry_count = 0
            
            while retry_count <= max_retries:
                try:
                    return func(*args, **kwargs)
                except DCBSError as e:
                    if retry_count >= max_retries:
                        raise
                    
                    if recovery_manager.attempt_recovery(e, kwargs):
                        retry_count += 1
                        recovery_manager.logger.info(f"Retrying {func.__name__} (attempt {retry_count + 1})")
                        continue
                    else:
                        raise
                except Exception as e:
                    # Convert non-DCBS exceptions to DCBSError for consistency
                    dcbs_error = DCBSRuntimeError(f"Unexpected error in {func.__name__}: {str(e)}")
                    raise dcbs_error from e
            
        return wrapper
    return decorator


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    component_config: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Configure logging for DCBS evaluation.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, logs to console only)
        component_config: Component-specific logging configuration
    """
    level = getattr(logging, log_level.upper())

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers from all loggers to prevent duplicates
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # Configure DCBS root logger
    dcbs_logger = logging.getLogger("dcbs")
    dcbs_logger.setLevel(level)

    # Prevent propagation to avoid duplicate logs
    dcbs_logger.propagate = False

    # Clear existing handlers
    dcbs_logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    dcbs_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(
            log_file, mode="w"
        )  # 'w' mode to overwrite existing log
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        dcbs_logger.addHandler(file_handler)

    # Configure component loggers
    component_loggers = {
        "dcbs.eval": eval_logger,
        "dcbs.algorithm": dcbs_logger,
        "dcbs.visualization": vis_logger,
    }

    # Set up each component logger
    for logger_name, logger_instance in component_loggers.items():
        # Clear any existing handlers
        logger_instance.handlers = []

        # Set component-specific level if configured, otherwise use default level
        component_level = level
        if component_config and logger_name in component_config:
            component_level_name = component_config[logger_name].get("level", log_level)
            component_level = getattr(logging, component_level_name.upper())

        logger_instance.setLevel(component_level)

        # Prevent propagation to avoid duplicate logs
        logger_instance.propagate = False

        # Add the same handlers as the dcbs root logger
        for handler in dcbs_logger.handlers:
            # Create a new handler of the same type to avoid sharing handlers
            if isinstance(handler, logging.FileHandler) and log_file:
                new_handler = logging.FileHandler(
                    log_file, mode="a"
                )  # Append mode for component loggers
            else:
                new_handler = logging.StreamHandler(sys.stdout)

            new_handler.setLevel(component_level)
            new_handler.setFormatter(handler.formatter)
            logger_instance.addHandler(new_handler)

        # Log the component's logging level
        logger_instance.debug(
            f"Logger {logger_name} configured with level {logging.getLevelName(component_level)}"
        )


def log_exception(
    e: Exception, logger_instance: logging.Logger = logger, log_traceback: bool = True
) -> None:
    """Log exception details with appropriate formatting.

    Args:
        e: Exception instance
        logger_instance: Logger to use
        log_traceback: Whether to log full traceback
    """
    if isinstance(e, DCBSError) and hasattr(e, "details") and e.details:
        logger_instance.error(f"{type(e).__name__}: {str(e)}")
        for key, value in e.details.items():
            logger_instance.error(f"  {key}: {value}")
    else:
        logger_instance.error(f"{type(e).__name__}: {str(e)}")

    if log_traceback:
        logger_instance.debug("Traceback:", exc_info=True)


def report_memory_usage(
    operation_name: str,
    logger_instance: logging.Logger = logger,
    threshold_mb: int = 10,
    include_details: bool = False,
    warning_threshold_mb: int = 2000,
    gc_threshold_mb: int = 1000,
) -> float:
    """Report memory usage for an operation and trigger garbage collection if necessary.

    Args:
        operation_name: Name of the operation being monitored
        logger_instance: Logger to use
        threshold_mb: Only log if memory change exceeds this threshold (MB)
        include_details: Whether to include detailed memory statistics
        warning_threshold_mb: Log a warning if memory usage exceeds this threshold
        gc_threshold_mb: Trigger garbage collection when memory usage exceeds this threshold

    Returns:
        Current memory usage in MB
    """
    try:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)

        # Get memory change since last call
        if not hasattr(report_memory_usage, "last_memory"):
            report_memory_usage.last_memory = memory_mb
            memory_change = 0
        else:
            memory_change = memory_mb - report_memory_usage.last_memory
            report_memory_usage.last_memory = memory_mb

        # Run garbage collection if memory exceeds threshold
        if memory_mb > gc_threshold_mb:
            import gc

            logger_instance.info(
                f"Memory usage ({memory_mb:.2f} MB) exceeded threshold of {gc_threshold_mb} MB. Triggering garbage collection..."
            )

            # Record memory before gc
            pre_gc_memory = memory_mb

            # Run garbage collection
            gc.collect()

            # Re-measure memory after gc
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            gc_freed = pre_gc_memory - memory_mb

            logger_instance.info(
                f"Garbage collection complete. Freed {gc_freed:.2f} MB, current usage: {memory_mb:.2f} MB"
            )

        # Log if change exceeds threshold or memory usage is high
        log_message = False
        log_level = logging.DEBUG

        # Determine if and at what level we should log
        if abs(memory_change) >= threshold_mb:
            log_message = True

        if memory_mb >= warning_threshold_mb:
            log_message = True
            log_level = logging.WARNING

        if log_message:
            message = f"Memory usage during {operation_name}: {memory_mb:.2f} MB (Δ: {memory_change:+.2f} MB)"

            if log_level == logging.DEBUG:
                logger_instance.debug(message)
            elif log_level == logging.WARNING:
                logger_instance.warning(message)

            # Include detailed info if requested or if memory usage is high
            if include_details or log_level >= logging.WARNING:
                # Get overall system memory
                system_memory = psutil.virtual_memory()
                percent_used = memory_mb / (system_memory.total / (1024 * 1024)) * 100

                detail_message = (
                    f"  Process memory: {memory_mb:.2f} MB, "
                    f"System memory: {system_memory.percent:.1f}% used, "
                    f"Process share: {percent_used:.2f}%"
                )

                if log_level == logging.WARNING:
                    logger_instance.warning(detail_message)

        return memory_mb
    except ImportError as e:
        logger_instance.debug(f"Couldn't report memory usage: {str(e)}")
        return 0.0
