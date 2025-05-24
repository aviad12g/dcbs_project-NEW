"""
DCBS evaluation error handling utilities.

This module provides custom exception classes and error handling utilities
for the DCBS evaluation framework.
"""

import logging
import traceback
from typing import Optional, Dict, Any, List, Union
import sys
import os

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
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize error with message and optional details.
        
        Args:
            message: Error description
            details: Additional context about the error
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)


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


def setup_logging(
    log_level: str = "INFO", 
    log_file: Optional[str] = None,
    component_config: Optional[Dict[str, Dict[str, Any]]] = None
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
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
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
            
        file_handler = logging.FileHandler(log_file, mode='w')  # 'w' mode to overwrite existing log
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        dcbs_logger.addHandler(file_handler)
        
    # Configure component loggers
    component_loggers = {
        "dcbs.eval": eval_logger,
        "dcbs.algorithm": dcbs_logger,
        "dcbs.visualization": vis_logger
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
                new_handler = logging.FileHandler(log_file, mode='a')  # Append mode for component loggers
            else:
                new_handler = logging.StreamHandler(sys.stdout)
                
            new_handler.setLevel(component_level)
            new_handler.setFormatter(handler.formatter)
            logger_instance.addHandler(new_handler)
            
        # Log the component's logging level
        logger_instance.debug(f"Logger {logger_name} configured with level {logging.getLevelName(component_level)}")


def log_exception(
    e: Exception, 
    logger_instance: logging.Logger = logger, 
    log_traceback: bool = True
) -> None:
    """Log exception details with appropriate formatting.
    
    Args:
        e: Exception instance
        logger_instance: Logger to use
        log_traceback: Whether to log full traceback
    """
    if isinstance(e, DCBSError) and hasattr(e, 'details') and e.details:
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
    critical_threshold_mb: int = 3500,
    gc_threshold_mb: int = 1000
) -> float:
    """Report memory usage for an operation and trigger garbage collection if necessary.
    
    Args:
        operation_name: Name of the operation being monitored
        logger_instance: Logger to use
        threshold_mb: Only log if memory change exceeds this threshold (MB)
        include_details: Whether to include detailed memory statistics
        warning_threshold_mb: Log a warning if memory usage exceeds this threshold
        critical_threshold_mb: Log a critical warning if memory usage exceeds this threshold
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
            logger_instance.info(f"Memory usage ({memory_mb:.2f} MB) exceeded threshold of {gc_threshold_mb} MB. Triggering garbage collection...")
            
            # Record memory before gc
            pre_gc_memory = memory_mb
            
            # Run garbage collection
            gc.collect()
            
            # Re-measure memory after gc
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            gc_freed = pre_gc_memory - memory_mb
            
            logger_instance.info(f"Garbage collection complete. Freed {gc_freed:.2f} MB, current usage: {memory_mb:.2f} MB")
        
        # Log if change exceeds threshold or memory usage is high
        log_message = False
        log_level = logging.DEBUG
        
        # Determine if and at what level we should log
        if abs(memory_change) >= threshold_mb:
            log_message = True
        
        if memory_mb >= critical_threshold_mb:
            log_message = True
            log_level = logging.CRITICAL
        elif memory_mb >= warning_threshold_mb:
            log_message = True
            log_level = logging.WARNING
            
        if log_message:
            message = f"Memory usage during {operation_name}: {memory_mb:.2f} MB (Δ: {memory_change:+.2f} MB)"
            
            if log_level == logging.DEBUG:
                logger_instance.debug(message)
            elif log_level == logging.WARNING:
                logger_instance.warning(message)
            elif log_level == logging.CRITICAL:
                logger_instance.critical(message)
            
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
                
                if log_level == logging.DEBUG:
                    logger_instance.debug(detail_message)
                elif log_level == logging.WARNING:
                    logger_instance.warning(detail_message)
                elif log_level == logging.CRITICAL:
                    logger_instance.critical(detail_message)
                
        return memory_mb
    except ImportError as e:
        logger_instance.debug(f"Couldn't report memory usage: {str(e)}")
        return 0.0  


class MemoryProfiler:
    """Memory profiling utility for tracking memory usage during evaluation.
    
    This class provides tools to monitor memory usage during different phases
    of DCBS evaluation, recording peak memory usage and providing detailed
    statistics.
    """
    
    def __init__(
        self, 
        enabled: bool = False, 
        sampling_interval_ms: int = 1000,
        trace_allocations: bool = False,
        record_peak_for: List[str] = None,
        logger_instance: logging.Logger = logger
    ):
        """Initialize memory profiler.
        
        Args:
            enabled: Whether profiling is enabled
            sampling_interval_ms: How often to sample memory usage (ms)
            trace_allocations: Whether to trace Python object allocations
            record_peak_for: List of operation types to record peak memory for
            logger_instance: Logger to use for reporting
        """
        self.enabled = enabled
        self.sampling_interval_ms = sampling_interval_ms
        self.trace_allocations = trace_allocations
        self.record_peak_for = record_peak_for or []
        self.logger = logger_instance
        
        self.current_operation = None
        self.peak_memory = {}
        self.sampling_thread = None
        self.should_stop = False
        
        if self.enabled:
            self.logger.info(f"Memory profiling enabled with {sampling_interval_ms}ms sampling interval")
            if self.trace_allocations:
                try:
                    import tracemalloc
                    tracemalloc.start()
                    self.logger.info("Allocation tracing enabled with tracemalloc")
                except ImportError:
                    self.logger.warning("tracemalloc module not available, allocation tracing disabled")
                    self.trace_allocations = False
    
    def start_operation(self, operation_name: str) -> None:
        """Start tracking memory for a specific operation.
        
        Args:
            operation_name: Name of the operation to track
        """
        if not self.enabled:
            return
            
        self.current_operation = operation_name
        self.logger.debug(f"Starting memory tracking for operation: {operation_name}")
        
        if operation_name in self.record_peak_for:
            import threading
            import time
            import psutil
            
            def _sample_memory():
                process = psutil.Process()
                current_peak = 0
                
                while not self.should_stop:
                    try:
                        memory_info = process.memory_info()
                        memory_mb = memory_info.rss / (1024 * 1024)
                        
                        if memory_mb > current_peak:
                            current_peak = memory_mb
                            self.peak_memory[operation_name] = current_peak
                            
                        time.sleep(self.sampling_interval_ms / 1000)
                    except Exception as e:
                        self.logger.error(f"Error in memory sampling: {str(e)}")
                        break
            
            self.should_stop = False
            self.sampling_thread = threading.Thread(target=_sample_memory)
            self.sampling_thread.daemon = True
            self.sampling_thread.start()
    
    def end_operation(self, operation_name: str) -> None:
        """End tracking memory for a specific operation.
        
        Args:
            operation_name: Name of the operation to stop tracking
        """
        if not self.enabled or operation_name != self.current_operation:
            return
            
        if self.sampling_thread and self.sampling_thread.is_alive():
            self.should_stop = True
            self.sampling_thread.join(timeout=1.0)
            
        peak = self.peak_memory.get(operation_name, 0)
        if peak > 0:
            self.logger.info(f"Peak memory usage during {operation_name}: {peak:.2f} MB")
            
        self.current_operation = None
        
    def get_allocation_summary(self) -> str:
        """Get a summary of memory allocations if tracing is enabled.
        
        Returns:
            String with allocation summary
        """
        if not self.enabled or not self.trace_allocations:
            return "Allocation tracing not enabled"
            
        try:
            import tracemalloc
            
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            summary = ["Top 10 memory allocations:"]
            for i, stat in enumerate(top_stats[:10], 1):
                summary.append(f"#{i}: {stat.size / 1024:.1f} KB - {stat.traceback.format()[0]}")
                
            return "\n".join(summary)
        except ImportError:
            return "tracemalloc module not available"
    
    def report_summary(self) -> None:
        """Report a summary of memory usage statistics."""
        if not self.enabled:
            return
            
        self.logger.info("Memory profiling summary:")
        for operation, peak in self.peak_memory.items():
            self.logger.info(f"  Peak memory for {operation}: {peak:.2f} MB")
            
        if self.trace_allocations:
            allocation_summary = self.get_allocation_summary()
            self.logger.info(allocation_summary)
    
    def cleanup(self) -> None:
        """Clean up profiling resources."""
        if not self.enabled:
            return
            
        if self.trace_allocations:
            try:
                import tracemalloc
                tracemalloc.stop()
            except ImportError:
                pass  