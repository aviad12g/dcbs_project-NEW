# Performance Profiling

Module: `src.profiler`

## Overview

Performance profiling and optimization utilities.

This module provides tools for profiling DCBS evaluation performance,
identifying bottlenecks, and suggesting optimizations.

## Classes

### CacheProfiler

Profiler for caching operations.

#### Methods

##### clear

```python
clear()
```

Clear all profiling data.

##### get_hit_rate

```python
get_hit_rate()
```

Get cache hit rate.

##### get_stats

```python
get_stats()
```

Get cache profiling statistics.

##### record_hit

```python
record_hit(access_time: float)
```

Record a cache hit.

##### record_miss

```python
record_miss(computation_time: float)
```

Record a cache miss.

##### record_size

```python
record_size(size_bytes: int)
```

Record cache size.


### ModelProfiler

Specialized profiler for model operations.

#### Methods

##### clear

```python
clear()
```

Clear all profiling data.

##### get_stats

```python
get_stats()
```

Get profiling statistics.

##### profile_forward_pass

```python
profile_forward_pass()
```

Profile a model forward pass.

##### profile_generation

```python
profile_generation()
```

Profile text generation.


### PerformanceProfiler

Comprehensive performance profiler for DCBS evaluation.

#### Methods

##### clear_data

```python
clear_data()
```

Clear all profiling data.

##### generate_report

```python
generate_report()
```

Generate a comprehensive performance report.

##### get_bottlenecks

```python
get_bottlenecks(threshold_seconds: float = 1.0)
```

Identify performance bottlenecks.

##### get_memory_summary

```python
get_memory_summary()
```

Get memory usage summary for all profiled sections.

##### get_profile_stats

```python
get_profile_stats(section_name: str, sort_by: str = cumulative)
```

Get formatted profiling statistics for a section.

##### get_timing_summary

```python
get_timing_summary()
```

Get timing summary for all profiled sections.

##### profile_section

```python
profile_section(section_name: str)
```

Context manager for profiling a code section.


## Functions

### get_global_profiler

```python
get_global_profiler()
```

Get the global profiler instance.

### profile_evaluation_run

```python
profile_evaluation_run(func: Callable)
```

Decorator for profiling entire evaluation runs.

### profile_function

```python
profile_function(profiler: Optional = None, section_name: Optional = None)
```

Decorator for profiling individual functions.
