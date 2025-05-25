# Usage Examples

This document provides comprehensive usage examples for the DCBS evaluation framework.

## Basic Usage

### Running a Simple Evaluation

```python
from src.evaluation_core import EvaluationConfig, EvaluationRunner
from dcbs.samplers import DCBSSampler

# Create configuration
config = EvaluationConfig(
    model_name='meta-llama/Llama-3.2-1B',
    benchmark_path='data/arc_easy_processed.json',
    output_dir='results',
    limit=10  # Evaluate only 10 examples
)

# Run evaluation
runner = EvaluationRunner(config)
results = runner.run_evaluation()

print(f'Accuracy: {results.accuracy:.2%}')
```

### Using Different Samplers

```python
from dcbs.samplers import GreedySampler, TopPSampler, DCBSSampler

# Greedy sampling
greedy_sampler = GreedySampler()

# Top-p sampling
top_p_sampler = TopPSampler(p=0.9)

# DCBS sampling
dcbs_sampler = DCBSSampler.create_default(k=8, top_n=50)

# Use in evaluation
runner = EvaluationRunner(config)
runner.sampler_factory.register_sampler('custom_dcbs', dcbs_sampler)
```

## Configuration Management

### Using YAML Configuration

Create a configuration file `config.yaml`:

```yaml
model_path: 'meta-llama/Llama-3.2-1B'
benchmark: 'data/arc_easy_processed.json'
output_dir: 'results'
limit: 100
include_cot: true
log_level: 'INFO'

dcbs_params:
  k: 8
  top_n: 50
  cache_size: 1000

performance:
  batch_size: 1
  timeout_seconds: 3600
```

Load and use the configuration:

```python
from src.config_schema import validate_config_file
from src.config_builder import ConfigBuilder
import argparse

# Load configuration
config_dict = validate_config_file('config.yaml')

# Merge with command-line arguments
args = argparse.Namespace(limit=50)  # Override limit
config = ConfigBuilder.merge_config_with_args(config_dict, args)

# Run evaluation
runner = EvaluationRunner(config)
results = runner.run_evaluation()
```

### Environment Variables

You can override configuration values using environment variables:

```bash
export DCBS_MODEL_PATH='different-model'
export DCBS_K=16
export DCBS_LOG_LEVEL='DEBUG'

python compare_methods.py --config config.yaml
```

## Error Handling and Recovery

### Using Error Recovery

```python
from src.errors import ErrorRecoveryManager, with_error_recovery

# Create recovery manager
recovery_manager = ErrorRecoveryManager()

# Use decorator for automatic recovery
@with_error_recovery(recovery_manager)
def run_evaluation_with_recovery(**kwargs):
    runner = EvaluationRunner(config)
    return runner.run_evaluation()

# This will automatically retry with recovery strategies
results = run_evaluation_with_recovery(max_retries=3)
```

### Custom Error Handling

```python
from src.errors import ModelLoadError, ResourceError

try:
    runner = EvaluationRunner(config)
    results = runner.run_evaluation()
except ModelLoadError as e:
    print(f'Model loading failed: {e.message}')
    print(f'Details: {e.details}')
    if e.recoverable:
        print('This error might be recoverable')
except ResourceError as e:
    print(f'Resource error: {e.message}')
    # Handle resource constraints
```

## Performance Profiling

### Basic Profiling

```python
from src.profiler import PerformanceProfiler, profile_evaluation_run

# Create profiler
profiler = PerformanceProfiler()

# Profile specific sections
with profiler.profile_section('model_loading'):
    runner = EvaluationRunner(config)

with profiler.profile_section('evaluation'):
    results = runner.run_evaluation()

# Generate report
report = profiler.generate_report()
print(report)
```

### Automatic Profiling

```python
@profile_evaluation_run
def run_full_evaluation():
    runner = EvaluationRunner(config)
    return runner.run_evaluation()

# This will automatically profile and log performance
results = run_full_evaluation()
```

## Advanced Usage

### Custom Sampler Implementation

```python
from dcbs.samplers import Sampler, SamplingContext
import torch

class CustomSampler(Sampler):
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def sample(self, logits: torch.Tensor, 
               context: SamplingContext = None,
               filter_tokens: set = None) -> int:
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        
        # Apply filtering if specified
        if filter_tokens:
            mask = torch.ones_like(scaled_logits, dtype=torch.bool)
            for token_id in filter_tokens:
                if token_id < len(mask):
                    mask[token_id] = False
            scaled_logits = scaled_logits[mask]
            valid_indices = torch.nonzero(mask).squeeze()
        else:
            valid_indices = torch.arange(len(scaled_logits))

        # Sample from distribution
        probs = torch.softmax(scaled_logits, dim=-1)
        sampled_idx = torch.multinomial(probs, 1).item()
        
        return valid_indices[sampled_idx].item()

# Use custom sampler
custom_sampler = CustomSampler(temperature=0.8)
runner.sampler_factory.register_sampler('custom', custom_sampler)
```

### Batch Processing

```python
# Process multiple configurations
configs = [
    EvaluationConfig(model_name='model1', ...),
    EvaluationConfig(model_name='model2', ...),
]

results = []
for config in configs:
    runner = EvaluationRunner(config)
    result = runner.run_evaluation()
    results.append(result)

# Compare results
for i, result in enumerate(results):
    print(f'Config {i}: Accuracy = {result.accuracy:.2%}')
```