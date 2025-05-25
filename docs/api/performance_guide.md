# Performance Optimization Guide

This guide provides tips and strategies for optimizing DCBS evaluation performance.

## Profiling and Monitoring

### Built-in Profiling

The DCBS framework includes comprehensive profiling tools:

```python
from src.profiler import PerformanceProfiler

profiler = PerformanceProfiler()

with profiler.profile_section('evaluation'):
    # Your evaluation code here
    pass

# Get detailed report
report = profiler.generate_report()
print(report)
```

### Memory Monitoring

Monitor memory usage to identify bottlenecks:

```python
from src.errors import report_memory_usage

# Monitor memory at key points
report_memory_usage('after_model_load')
# ... evaluation code ...
report_memory_usage('after_evaluation')
```

## Optimization Strategies

### Model Loading Optimization

1. **Use 4-bit Quantization**:
   ```yaml
   load_in_4bit: true
   ```

2. **CPU Fallback for Large Models**:
   ```python
   # Automatic fallback in error recovery
   from src.errors import with_error_recovery
   
   @with_error_recovery()
   def load_model():
       # Will automatically try CPU if GPU fails
       pass
   ```

### Memory Optimization

1. **Reduce Batch Size**:
   ```yaml
   performance:
     batch_size: 1  # Start with 1, increase if memory allows
   ```

2. **Set Memory Limits**:
   ```yaml
   performance:
     memory_limit_gb: 8.0  # Adjust based on available RAM
   ```

3. **Optimize Cache Size**:
   ```yaml
   dcbs_params:
     cache_size: 500  # Reduce if memory is limited
   ```

### DCBS Parameter Tuning

1. **Cluster Count (k)**:
   - Smaller k (4-8): Faster, less diverse
   - Larger k (16-32): Slower, more diverse

2. **Top-N Selection**:
   - Smaller top_n (20-50): Faster clustering
   - Larger top_n (100-200): Better token coverage

3. **Optimal Configurations**:
   ```yaml
   # Fast configuration
   dcbs_params:
     k: 4
     top_n: 20
   
   # Balanced configuration
   dcbs_params:
     k: 8
     top_n: 50
   
   # High-quality configuration
   dcbs_params:
     k: 16
     top_n: 100
   ```

### Caching Optimization

1. **Enable Caching for Repeated Evaluations**:
   ```yaml
   enable_caching: true
   dcbs_params:
     cache_size: 1000
   ```

2. **Disable Caching for One-off Evaluations**:
   ```yaml
   enable_caching: false
   ```

3. **Monitor Cache Performance**:
   ```python
   from src.profiler import CacheProfiler
   
   cache_profiler = CacheProfiler()
   # ... use during evaluation ...
   stats = cache_profiler.get_stats()
   print(f'Cache hit rate: {stats["hit_rate"]:.2%}')
   ```

## Hardware-Specific Optimizations

### GPU Optimization

1. **Use Mixed Precision**:
   ```python
   # Enable in model loading
   torch_dtype = torch.float16  # or torch.bfloat16
   ```

2. **Monitor GPU Memory**:
   ```python
   import torch
   print(f'GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f}GB')
   ```

### CPU Optimization

1. **Set Thread Count**:
   ```python
   import torch
   torch.set_num_threads(4)  # Adjust based on CPU cores
   ```

2. **Use CPU-Optimized Models**:
   ```yaml
   model_path: 'model-name-cpu-optimized'
   load_in_4bit: true
   ```

## Performance Benchmarks

### Typical Performance Ranges

| Configuration | Examples/Hour | Memory Usage | Quality |
|---------------|---------------|--------------|---------|
| Fast | 100-200 | 2-4GB | Good |
| Balanced | 50-100 | 4-8GB | Better |
| High-Quality | 20-50 | 8-16GB | Best |

### Bottleneck Identification

Common bottlenecks and solutions:

1. **Model Loading (>30s)**:
   - Use 4-bit quantization
   - Consider smaller models

2. **Token Generation (>5s per example)**:
   - Reduce max_new_tokens
   - Use greedy sampling for comparison

3. **DCBS Clustering (>2s per example)**:
   - Reduce k and top_n
   - Enable caching

4. **Memory Issues**:
   - Reduce batch_size
   - Enable 4-bit quantization
   - Set memory limits

## Monitoring and Alerts

### Performance Monitoring

```python
from src.profiler import get_global_profiler

profiler = get_global_profiler()

# Check for bottlenecks
bottlenecks = profiler.get_bottlenecks(threshold_seconds=5.0)
if bottlenecks:
    print('Performance bottlenecks detected:')
    for section, duration in bottlenecks:
        print(f'  {section}: {duration:.2f}s')
```

### Automated Optimization

```python
from src.errors import ErrorRecoveryManager

# Automatic performance optimization
recovery_manager = ErrorRecoveryManager()

# Will automatically reduce batch size, enable quantization, etc.
# when resource errors are detected
```