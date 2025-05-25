# Error Handling

Module: `src.errors`

## Overview

DCBS evaluation error handling utilities.

This module provides custom exception classes and error handling utilities
for the DCBS evaluation framework.

## Classes

### CacheError

Exception raised for errors related to caching operations.

#### Constructor

```python
CacheError(message: str, details: Optional = None, recoverable: bool = False)
```

Initialize error with message and optional details.

Args:
    message: Error description
    details: Additional context about the error
    recoverable: Whether this error can be recovered from

#### Methods

##### to_dict

```python
to_dict()
```

Convert error to dictionary for serialization.


### ConfigurationError

Exception raised for configuration errors.

#### Constructor

```python
ConfigurationError(message: str, details: Optional = None, recoverable: bool = False)
```

Initialize error with message and optional details.

Args:
    message: Error description
    details: Additional context about the error
    recoverable: Whether this error can be recovered from

#### Methods

##### to_dict

```python
to_dict()
```

Convert error to dictionary for serialization.


### DCBSError

Base exception class for DCBS errors.

#### Constructor

```python
DCBSError(message: str, details: Optional = None, recoverable: bool = False)
```

Initialize error with message and optional details.

Args:
    message: Error description
    details: Additional context about the error
    recoverable: Whether this error can be recovered from

#### Methods

##### to_dict

```python
to_dict()
```

Convert error to dictionary for serialization.


### DCBSRuntimeError

Exception raised for errors during DCBS execution.

#### Constructor

```python
DCBSRuntimeError(message: str, details: Optional = None, recoverable: bool = False)
```

Initialize error with message and optional details.

Args:
    message: Error description
    details: Additional context about the error
    recoverable: Whether this error can be recovered from

#### Methods

##### to_dict

```python
to_dict()
```

Convert error to dictionary for serialization.


### DataError

Exception raised for errors related to benchmark data.

#### Constructor

```python
DataError(message: str, details: Optional = None, recoverable: bool = False)
```

Initialize error with message and optional details.

Args:
    message: Error description
    details: Additional context about the error
    recoverable: Whether this error can be recovered from

#### Methods

##### to_dict

```python
to_dict()
```

Convert error to dictionary for serialization.


### ErrorRecoveryManager

Manages error recovery strategies for different types of failures.

#### Methods

##### attempt_recovery

```python
attempt_recovery(error: DCBSError, context: Optional = None)
```

Attempt to recover from an error.

Args:
    error: The error to recover from
    context: Additional context for recovery
    
Returns:
    True if recovery was successful, False otherwise


### EvaluationError

Exception raised for errors during evaluation.

#### Constructor

```python
EvaluationError(message: str, details: Optional = None, recoverable: bool = False)
```

Initialize error with message and optional details.

Args:
    message: Error description
    details: Additional context about the error
    recoverable: Whether this error can be recovered from

#### Methods

##### to_dict

```python
to_dict()
```

Convert error to dictionary for serialization.


### MemoryProfiler

Memory profiling utility for tracking memory usage during evaluation.

This class provides tools to monitor memory usage during different phases
of DCBS evaluation, recording peak memory usage and providing detailed
statistics.

#### Constructor

```python
MemoryProfiler(enabled: bool = False, sampling_interval_ms: int = 1000, trace_allocations: bool = False, record_peak_for: List = None, logger_instance: Logger = <Logger dcbs (INFO)>)
```

Initialize memory profiler.

Args:
    enabled: Whether profiling is enabled
    sampling_interval_ms: How often to sample memory usage (ms)
    trace_allocations: Whether to trace Python object allocations
    record_peak_for: List of operation types to record peak memory for
    logger_instance: Logger to use for reporting

#### Methods

##### cleanup

```python
cleanup()
```

Clean up profiling resources.

##### end_operation

```python
end_operation(operation_name: str)
```

End tracking memory for a specific operation.

Args:
    operation_name: Name of the operation to stop tracking

##### get_allocation_summary

```python
get_allocation_summary()
```

Get a summary of memory allocations if tracing is enabled.

Returns:
    String with allocation summary

##### report_summary

```python
report_summary()
```

Report a summary of memory usage statistics.

##### start_operation

```python
start_operation(operation_name: str)
```

Start tracking memory for a specific operation.

Args:
    operation_name: Name of the operation to track


### ModelLoadError

Exception raised for errors during model loading.

#### Constructor

```python
ModelLoadError(message: str, details: Optional = None, recoverable: bool = False)
```

Initialize error with message and optional details.

Args:
    message: Error description
    details: Additional context about the error
    recoverable: Whether this error can be recovered from

#### Methods

##### to_dict

```python
to_dict()
```

Convert error to dictionary for serialization.


### ResourceError

Exception raised for resource-related errors (memory, disk, etc.).

#### Constructor

```python
ResourceError(message: str, details: Optional = None, recoverable: bool = False)
```

Initialize error with message and optional details.

Args:
    message: Error description
    details: Additional context about the error
    recoverable: Whether this error can be recovered from

#### Methods

##### to_dict

```python
to_dict()
```

Convert error to dictionary for serialization.


### SamplingError

Exception raised for errors during token sampling.

#### Constructor

```python
SamplingError(message: str, details: Optional = None, recoverable: bool = False)
```

Initialize error with message and optional details.

Args:
    message: Error description
    details: Additional context about the error
    recoverable: Whether this error can be recovered from

#### Methods

##### to_dict

```python
to_dict()
```

Convert error to dictionary for serialization.


### TimeoutError

Exception raised when operations exceed time limits.

#### Constructor

```python
TimeoutError(message: str, details: Optional = None, recoverable: bool = False)
```

Initialize error with message and optional details.

Args:
    message: Error description
    details: Additional context about the error
    recoverable: Whether this error can be recovered from

#### Methods

##### to_dict

```python
to_dict()
```

Convert error to dictionary for serialization.


### ValidationError

Exception raised for validation failures.

#### Constructor

```python
ValidationError(message: str, details: Optional = None, recoverable: bool = False)
```

Initialize error with message and optional details.

Args:
    message: Error description
    details: Additional context about the error
    recoverable: Whether this error can be recovered from

#### Methods

##### to_dict

```python
to_dict()
```

Convert error to dictionary for serialization.


### VisualizationError

Exception raised for errors during visualization.

#### Constructor

```python
VisualizationError(message: str, details: Optional = None, recoverable: bool = False)
```

Initialize error with message and optional details.

Args:
    message: Error description
    details: Additional context about the error
    recoverable: Whether this error can be recovered from

#### Methods

##### to_dict

```python
to_dict()
```

Convert error to dictionary for serialization.


## Functions

### log_exception

```python
log_exception(e: Exception, logger_instance: Logger = <Logger dcbs (INFO)>, log_traceback: bool = True)
```

Log exception details with appropriate formatting.

Args:
    e: Exception instance
    logger_instance: Logger to use
    log_traceback: Whether to log full traceback

### report_memory_usage

```python
report_memory_usage(operation_name: str, logger_instance: Logger = <Logger dcbs (INFO)>, threshold_mb: int = 10, include_details: bool = False, warning_threshold_mb: int = 2000, gc_threshold_mb: int = 1000)
```

Report memory usage for an operation and trigger garbage collection if necessary.

Args:
    operation_name: Name of the operation being monitored
    logger_instance: Logger to use
    threshold_mb: Only log if memory change exceeds this threshold (MB)
    include_details: Whether to include detailed memory statistics
    warning_threshold_mb: Log a warning if memory usage exceeds this threshold
    gc_threshold_mb: Trigger garbage collection when memory usage exceeds this threshold

Returns:
    Current memory usage in MB

### setup_logging

```python
setup_logging(log_level: str = INFO, log_file: Optional = None, component_config: Optional = None)
```

Configure logging for DCBS evaluation.

Args:
    log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file: Path to log file (if None, logs to console only)
    component_config: Component-specific logging configuration

### with_error_recovery

```python
with_error_recovery(recovery_manager: Optional = None)
```

Decorator for functions that should attempt error recovery.
