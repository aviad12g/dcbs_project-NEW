# DCBS API Documentation

Welcome to the Deterministic Category Based Sampling (DCBS) evaluation framework documentation.

## Quick Start

```python
from src.evaluation_core import EvaluationConfig, EvaluationRunner

config = EvaluationConfig(
    model_name='meta-llama/Llama-3.2-1B',
    benchmark_path='data/arc_easy_processed.json'
)

runner = EvaluationRunner(config)
results = runner.run_evaluation()
print(f'Accuracy: {results.accuracy:.2%}')
```

## Documentation Sections

### API Reference

- [Sampling Algorithms](dcbs_samplers.md) - Core sampling implementations
- [Evaluation Framework](src_evaluation_core.md) - Main evaluation components
- [Configuration Management](src_config_schema.md) - Configuration and validation
- [Error Handling](src_errors.md) - Error management and recovery
- [Performance Profiling](src_profiler.md) - Performance monitoring tools

### Guides

- [Usage Examples](usage_examples.md) - Comprehensive usage examples
- [Configuration Guide](configuration_guide.md) - Configuration options and best practices
- [Performance Guide](performance_guide.md) - Optimization tips and benchmarks

## Key Features

- **Multiple Sampling Methods**: Greedy, Top-p, Random, and DCBS
- **Comprehensive Configuration**: YAML-based with environment variable support
- **Error Recovery**: Automatic recovery from common failures
- **Performance Profiling**: Built-in profiling and optimization suggestions
- **Extensible Architecture**: Easy to add custom samplers and evaluators

## Architecture Overview

```
DCBS Framework
├── Samplers (dcbs.samplers)
│   ├── GreedySampler
│   ├── TopPSampler
│   ├── RandomSampler
│   └── DCBSSampler
├── Evaluation Core (src.evaluation_core)
│   ├── EvaluationConfig
│   ├── EvaluationRunner
│   ├── ModelManager
│   └── SamplerFactory
├── Configuration (src.config_schema)
│   ├── ConfigValidator
│   └── EnvironmentVariableResolver
├── Error Handling (src.errors)
│   ├── Custom Exceptions
│   └── ErrorRecoveryManager
└── Profiling (src.profiler)
    ├── PerformanceProfiler
    ├── ModelProfiler
    └── CacheProfiler
```

## Getting Help

- Check the [Usage Examples](usage_examples.md) for common use cases
- Review the [Configuration Guide](configuration_guide.md) for setup help
- See the [Performance Guide](performance_guide.md) for optimization tips
- Examine the API reference for detailed function documentation