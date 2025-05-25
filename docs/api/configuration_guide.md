# Configuration Guide

This guide explains how to configure the DCBS evaluation framework.

## Configuration Schema

The DCBS framework uses a comprehensive configuration schema with validation.

### Required Fields

- `model_path`: HuggingFace model name or local path
- `benchmark`: Path to the benchmark JSON file

### Optional Fields

#### Basic Settings

- `output_dir`: Output directory for results (default: 'results')
- `limit`: Limit number of examples (default: null - process all)
- `include_cot`: Enable chain-of-thought reasoning (default: true)
- `log_level`: Logging level (default: 'INFO')

#### Model Settings

- `load_in_4bit`: Use 4-bit quantization (default: false)

#### Caching

- `enable_caching`: Enable DCBS caching (default: true)

#### Sampling Parameters

- `p_values`: List of top-p values for nucleus sampling (default: [0.9])

#### DCBS Parameters

```yaml
dcbs_params:
  k: 8                    # Number of clusters (1-100)
  top_n: 50              # Top-n tokens to consider (1-1000)
  cache_size: 1000       # Maximum cache size (0+)
```

#### Performance Settings

```yaml
performance:
  batch_size: 1          # Batch size for processing (1-100)
  memory_limit_gb: null  # Memory limit in GB (0.1+)
  timeout_seconds: 3600  # Timeout for operations (1+)
```

## Environment Variables

All configuration values can be overridden using environment variables:

| Configuration Field | Environment Variable | Type |
|---------------------|---------------------|------|
| `model_path` | `DCBS_MODEL_PATH` | string |
| `benchmark` | `DCBS_BENCHMARK_PATH` | string |
| `output_dir` | `DCBS_OUTPUT_DIR` | string |
| `limit` | `DCBS_LIMIT` | integer |
| `include_cot` | `DCBS_INCLUDE_COT` | boolean |
| `log_level` | `DCBS_LOG_LEVEL` | string |
| `load_in_4bit` | `DCBS_LOAD_IN_4BIT` | boolean |
| `enable_caching` | `DCBS_ENABLE_CACHING` | boolean |
| `dcbs_params.k` | `DCBS_K` | integer |
| `dcbs_params.top_n` | `DCBS_TOP_N` | integer |
| `dcbs_params.cache_size` | `DCBS_CACHE_SIZE` | integer |
| `performance.batch_size` | `DCBS_BATCH_SIZE` | integer |
| `performance.memory_limit_gb` | `DCBS_MEMORY_LIMIT_GB` | float |
| `performance.timeout_seconds` | `DCBS_TIMEOUT_SECONDS` | integer |

### Boolean Environment Variables

Boolean values can be set using: `true`, `1`, `yes`, `on` (case-insensitive)

## Configuration Templates

### Development Configuration

```yaml
model_path: 'meta-llama/Llama-3.2-1B'
benchmark: 'data/arc_easy_processed.json'
output_dir: 'dev_results'
limit: 10
include_cot: true
log_level: 'DEBUG'
load_in_4bit: true

dcbs_params:
  k: 4
  top_n: 20
  cache_size: 100

performance:
  batch_size: 1
  timeout_seconds: 300
```

### Production Configuration

```yaml
model_path: 'meta-llama/Llama-3.2-1B'
benchmark: 'data/arc_easy_processed.json'
output_dir: 'production_results'
limit: null
include_cot: true
log_level: 'INFO'
load_in_4bit: false

dcbs_params:
  k: 8
  top_n: 50
  cache_size: 1000

performance:
  batch_size: 1
  memory_limit_gb: 16.0
  timeout_seconds: 7200
```

### High-Performance Configuration

```yaml
model_path: 'meta-llama/Llama-3.2-1B'
benchmark: 'data/arc_easy_processed.json'
output_dir: 'hp_results'
limit: null
include_cot: true
log_level: 'WARNING'
load_in_4bit: true

dcbs_params:
  k: 16
  top_n: 100
  cache_size: 5000

performance:
  batch_size: 4
  memory_limit_gb: 32.0
  timeout_seconds: 14400
```