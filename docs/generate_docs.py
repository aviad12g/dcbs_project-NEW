"""
Automatic API documentation generator for DCBS.

This script generates comprehensive API documentation including:
- Class and function documentation
- Usage examples
- Configuration guides
- Performance optimization tips
"""

import ast
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class DocGenerator:
    """Generates comprehensive API documentation."""

    def __init__(self, output_dir: str = "docs/api"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_docs(self):
        """Generate all documentation."""
        print("Generating DCBS API Documentation...")
        
        # Generate module documentation
        self._generate_module_docs()
        
        # Generate usage examples
        self._generate_usage_examples()
        
        # Generate configuration guide
        self._generate_config_guide()
        
        # Generate performance guide
        self._generate_performance_guide()
        
        # Generate main index
        self._generate_index()
        
        print(f"Documentation generated in {self.output_dir}")

    def _generate_module_docs(self):
        """Generate documentation for all modules."""
        modules = [
            ("dcbs.samplers", "Sampling Algorithms"),
            ("src.evaluation_core", "Evaluation Framework"),
            ("src.config_schema", "Configuration Management"),
            ("src.errors", "Error Handling"),
            ("src.profiler", "Performance Profiling"),
        ]
        
        for module_name, title in modules:
            try:
                self._generate_module_doc(module_name, title)
            except Exception as e:
                print(f"Warning: Could not generate docs for {module_name}: {e}")

    def _generate_module_doc(self, module_name: str, title: str):
        """Generate documentation for a specific module."""
        try:
            module = __import__(module_name, fromlist=[''])
        except ImportError as e:
            print(f"Could not import {module_name}: {e}")
            return

        doc_content = [
            f"# {title}",
            "",
            f"Module: `{module_name}`",
            "",
        ]

        # Add module docstring
        if module.__doc__:
            doc_content.extend([
                "## Overview",
                "",
                module.__doc__.strip(),
                "",
            ])

        # Document classes
        classes = self._get_classes(module)
        if classes:
            doc_content.extend(["## Classes", ""])
            for class_name, class_obj in classes:
                doc_content.extend(self._document_class(class_obj))

        # Document functions
        functions = self._get_functions(module)
        if functions:
            doc_content.extend(["## Functions", ""])
            for func_name, func_obj in functions:
                doc_content.extend(self._document_function(func_obj))

        # Write to file
        filename = module_name.replace(".", "_") + ".md"
        with open(self.output_dir / filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(doc_content))

    def _get_classes(self, module) -> List[Tuple[str, Any]]:
        """Get all classes from a module."""
        classes = []
        for name in dir(module):
            obj = getattr(module, name)
            if (inspect.isclass(obj) and 
                obj.__module__ == module.__name__ and
                not name.startswith('_')):
                classes.append((name, obj))
        return sorted(classes)

    def _get_functions(self, module) -> List[Tuple[str, Any]]:
        """Get all functions from a module."""
        functions = []
        for name in dir(module):
            obj = getattr(module, name)
            if (inspect.isfunction(obj) and 
                obj.__module__ == module.__name__ and
                not name.startswith('_')):
                functions.append((name, obj))
        return sorted(functions)

    def _document_class(self, class_obj) -> List[str]:
        """Generate documentation for a class."""
        doc = [
            f"### {class_obj.__name__}",
            "",
        ]

        # Class docstring
        if class_obj.__doc__:
            doc.extend([
                class_obj.__doc__.strip(),
                "",
            ])

        # Constructor
        try:
            init_method = getattr(class_obj, '__init__')
            if init_method and init_method.__doc__:
                doc.extend([
                    "#### Constructor",
                    "",
                    f"```python",
                    f"{class_obj.__name__}({self._get_signature(init_method)})",
                    f"```",
                    "",
                    init_method.__doc__.strip(),
                    "",
                ])
        except:
            pass

        # Methods
        methods = self._get_public_methods(class_obj)
        if methods:
            doc.extend(["#### Methods", ""])
            for method_name, method_obj in methods:
                doc.extend(self._document_method(method_obj, method_name))

        doc.append("")
        return doc

    def _get_public_methods(self, class_obj) -> List[Tuple[str, Any]]:
        """Get public methods of a class."""
        methods = []
        for name in dir(class_obj):
            if not name.startswith('_'):
                obj = getattr(class_obj, name)
                if inspect.ismethod(obj) or inspect.isfunction(obj):
                    methods.append((name, obj))
        return sorted(methods)

    def _document_method(self, method_obj, method_name: str) -> List[str]:
        """Generate documentation for a method."""
        doc = [
            f"##### {method_name}",
            "",
        ]

        # Signature
        try:
            signature = self._get_signature(method_obj)
            doc.extend([
                f"```python",
                f"{method_name}({signature})",
                f"```",
                "",
            ])
        except:
            pass

        # Docstring
        if method_obj.__doc__:
            doc.extend([
                method_obj.__doc__.strip(),
                "",
            ])

        return doc

    def _document_function(self, func_obj) -> List[str]:
        """Generate documentation for a function."""
        doc = [
            f"### {func_obj.__name__}",
            "",
        ]

        # Signature
        try:
            signature = self._get_signature(func_obj)
            doc.extend([
                f"```python",
                f"{func_obj.__name__}({signature})",
                f"```",
                "",
            ])
        except:
            pass

        # Docstring
        if func_obj.__doc__:
            doc.extend([
                func_obj.__doc__.strip(),
                "",
            ])

        return doc

    def _get_signature(self, func_obj) -> str:
        """Get function signature as string."""
        try:
            sig = inspect.signature(func_obj)
            params = []
            for param in sig.parameters.values():
                if param.name == 'self':
                    continue
                param_str = param.name
                if param.annotation != param.empty:
                    param_str += f": {param.annotation.__name__ if hasattr(param.annotation, '__name__') else param.annotation}"
                if param.default != param.empty:
                    param_str += f" = {param.default}"
                params.append(param_str)
            return ", ".join(params)
        except:
            return "..."

    def _generate_usage_examples(self):
        """Generate usage examples documentation."""
        content = [
            "# Usage Examples",
            "",
            "This document provides comprehensive usage examples for the DCBS evaluation framework.",
            "",
            "## Basic Usage",
            "",
            "### Running a Simple Evaluation",
            "",
            "```python",
            "from src.evaluation_core import EvaluationConfig, EvaluationRunner",
            "from dcbs.samplers import DCBSSampler",
            "",
            "# Create configuration",
            "config = EvaluationConfig(",
            "    model_name='meta-llama/Llama-3.2-1B',",
            "    benchmark_path='data/arc_easy_processed.json',",
            "    output_dir='results',",
            "    limit=10  # Evaluate only 10 examples",
            ")",
            "",
            "# Run evaluation",
            "runner = EvaluationRunner(config)",
            "results = runner.run_evaluation()",
            "",
            "print(f'Accuracy: {results.accuracy:.2%}')",
            "```",
            "",
            "### Using Different Samplers",
            "",
            "```python",
            "from dcbs.samplers import GreedySampler, TopPSampler, DCBSSampler",
            "",
            "# Greedy sampling",
            "greedy_sampler = GreedySampler()",
            "",
            "# Top-p sampling",
            "top_p_sampler = TopPSampler(p=0.9)",
            "",
            "# DCBS sampling",
            "dcbs_sampler = DCBSSampler.create_default(k=8, top_n=50)",
            "",
            "# Use in evaluation",
            "runner = EvaluationRunner(config)",
            "runner.sampler_factory.register_sampler('custom_dcbs', dcbs_sampler)",
            "```",
            "",
            "## Configuration Management",
            "",
            "### Using YAML Configuration",
            "",
            "Create a configuration file `config.yaml`:",
            "",
            "```yaml",
            "model_path: 'meta-llama/Llama-3.2-1B'",
            "benchmark: 'data/arc_easy_processed.json'",
            "output_dir: 'results'",
            "limit: 100",
            "include_cot: true",
            "log_level: 'INFO'",
            "",
            "dcbs_params:",
            "  k: 8",
            "  top_n: 50",
            "  cache_size: 1000",
            "",
            "performance:",
            "  batch_size: 1",
            "  timeout_seconds: 3600",
            "```",
            "",
            "Load and use the configuration:",
            "",
            "```python",
            "from src.config_schema import validate_config_file",
            "from src.config_builder import ConfigBuilder",
            "import argparse",
            "",
            "# Load configuration",
            "config_dict = validate_config_file('config.yaml')",
            "",
            "# Merge with command-line arguments",
            "args = argparse.Namespace(limit=50)  # Override limit",
            "config = ConfigBuilder.merge_config_with_args(config_dict, args)",
            "",
            "# Run evaluation",
            "runner = EvaluationRunner(config)",
            "results = runner.run_evaluation()",
            "```",
            "",
            "### Environment Variables",
            "",
            "You can override configuration values using environment variables:",
            "",
            "```bash",
            "export DCBS_MODEL_PATH='different-model'",
            "export DCBS_K=16",
            "export DCBS_LOG_LEVEL='DEBUG'",
            "",
            "python compare_methods.py --config config.yaml",
            "```",
            "",
            "## Error Handling and Recovery",
            "",
            "### Using Error Recovery",
            "",
            "```python",
            "from src.errors import ErrorRecoveryManager, with_error_recovery",
            "",
            "# Create recovery manager",
            "recovery_manager = ErrorRecoveryManager()",
            "",
            "# Use decorator for automatic recovery",
            "@with_error_recovery(recovery_manager)",
            "def run_evaluation_with_recovery(**kwargs):",
            "    runner = EvaluationRunner(config)",
            "    return runner.run_evaluation()",
            "",
            "# This will automatically retry with recovery strategies",
            "results = run_evaluation_with_recovery(max_retries=3)",
            "```",
            "",
            "### Custom Error Handling",
            "",
            "```python",
            "from src.errors import ModelLoadError, ResourceError",
            "",
            "try:",
            "    runner = EvaluationRunner(config)",
            "    results = runner.run_evaluation()",
            "except ModelLoadError as e:",
            "    print(f'Model loading failed: {e.message}')",
            "    print(f'Details: {e.details}')",
            "    if e.recoverable:",
            "        print('This error might be recoverable')",
            "except ResourceError as e:",
            "    print(f'Resource error: {e.message}')",
            "    # Handle resource constraints",
            "```",
            "",
            "## Performance Profiling",
            "",
            "### Basic Profiling",
            "",
            "```python",
            "from src.profiler import PerformanceProfiler, profile_evaluation_run",
            "",
            "# Create profiler",
            "profiler = PerformanceProfiler()",
            "",
            "# Profile specific sections",
            "with profiler.profile_section('model_loading'):",
            "    runner = EvaluationRunner(config)",
            "",
            "with profiler.profile_section('evaluation'):",
            "    results = runner.run_evaluation()",
            "",
            "# Generate report",
            "report = profiler.generate_report()",
            "print(report)",
            "```",
            "",
            "### Automatic Profiling",
            "",
            "```python",
            "@profile_evaluation_run",
            "def run_full_evaluation():",
            "    runner = EvaluationRunner(config)",
            "    return runner.run_evaluation()",
            "",
            "# This will automatically profile and log performance",
            "results = run_full_evaluation()",
            "```",
            "",
            "## Advanced Usage",
            "",
            "### Custom Sampler Implementation",
            "",
            "```python",
            "from dcbs.samplers import Sampler, SamplingContext",
            "import torch",
            "",
            "class CustomSampler(Sampler):",
            "    def __init__(self, temperature: float = 1.0):",
            "        self.temperature = temperature",
            "",
            "    def sample(self, logits: torch.Tensor, ",
            "               context: SamplingContext = None,",
            "               filter_tokens: set = None) -> int:",
            "        # Apply temperature scaling",
            "        scaled_logits = logits / self.temperature",
            "        ",
            "        # Apply filtering if specified",
            "        if filter_tokens:",
            "            mask = torch.ones_like(scaled_logits, dtype=torch.bool)",
            "            for token_id in filter_tokens:",
            "                if token_id < len(mask):",
            "                    mask[token_id] = False",
            "            scaled_logits = scaled_logits[mask]",
            "            valid_indices = torch.nonzero(mask).squeeze()",
            "        else:",
            "            valid_indices = torch.arange(len(scaled_logits))",
            "",
            "        # Sample from distribution",
            "        probs = torch.softmax(scaled_logits, dim=-1)",
            "        sampled_idx = torch.multinomial(probs, 1).item()",
            "        ",
            "        return valid_indices[sampled_idx].item()",
            "",
            "# Use custom sampler",
            "custom_sampler = CustomSampler(temperature=0.8)",
            "runner.sampler_factory.register_sampler('custom', custom_sampler)",
            "```",
            "",
            "### Batch Processing",
            "",
            "```python",
            "# Process multiple configurations",
            "configs = [",
            "    EvaluationConfig(model_name='model1', ...),",
            "    EvaluationConfig(model_name='model2', ...),",
            "]",
            "",
            "results = []",
            "for config in configs:",
            "    runner = EvaluationRunner(config)",
            "    result = runner.run_evaluation()",
            "    results.append(result)",
            "",
            "# Compare results",
            "for i, result in enumerate(results):",
            "    print(f'Config {i}: Accuracy = {result.accuracy:.2%}')",
            "```",
        ]

        with open(self.output_dir / "usage_examples.md", 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))

    def _generate_config_guide(self):
        """Generate configuration guide."""
        content = [
            "# Configuration Guide",
            "",
            "This guide explains how to configure the DCBS evaluation framework.",
            "",
            "## Configuration Schema",
            "",
            "The DCBS framework uses a comprehensive configuration schema with validation.",
            "",
            "### Required Fields",
            "",
            "- `model_path`: HuggingFace model name or local path",
            "- `benchmark`: Path to the benchmark JSON file",
            "",
            "### Optional Fields",
            "",
            "#### Basic Settings",
            "",
            "- `output_dir`: Output directory for results (default: 'results')",
            "- `limit`: Limit number of examples (default: null - process all)",
            "- `include_cot`: Enable chain-of-thought reasoning (default: true)",
            "- `log_level`: Logging level (default: 'INFO')",
            "",
            "#### Model Settings",
            "",
            "- `load_in_4bit`: Use 4-bit quantization (default: false)",
            "",
            "#### Caching",
            "",
            "- `enable_caching`: Enable DCBS caching (default: true)",
            "",
            "#### Sampling Parameters",
            "",
            "- `p_values`: List of top-p values for nucleus sampling (default: [0.9])",
            "",
            "#### DCBS Parameters",
            "",
            "```yaml",
            "dcbs_params:",
            "  k: 8                    # Number of clusters (1-100)",
            "  top_n: 50              # Top-n tokens to consider (1-1000)",
            "  cache_size: 1000       # Maximum cache size (0+)",
            "```",
            "",
            "#### Performance Settings",
            "",
            "```yaml",
            "performance:",
            "  batch_size: 1          # Batch size for processing (1-100)",
            "  memory_limit_gb: null  # Memory limit in GB (0.1+)",
            "  timeout_seconds: 3600  # Timeout for operations (1+)",
            "```",
            "",
            "## Environment Variables",
            "",
            "All configuration values can be overridden using environment variables:",
            "",
            "| Configuration Field | Environment Variable | Type |",
            "|---------------------|---------------------|------|",
            "| `model_path` | `DCBS_MODEL_PATH` | string |",
            "| `benchmark` | `DCBS_BENCHMARK_PATH` | string |",
            "| `output_dir` | `DCBS_OUTPUT_DIR` | string |",
            "| `limit` | `DCBS_LIMIT` | integer |",
            "| `include_cot` | `DCBS_INCLUDE_COT` | boolean |",
            "| `log_level` | `DCBS_LOG_LEVEL` | string |",
            "| `load_in_4bit` | `DCBS_LOAD_IN_4BIT` | boolean |",
            "| `enable_caching` | `DCBS_ENABLE_CACHING` | boolean |",
            "| `dcbs_params.k` | `DCBS_K` | integer |",
            "| `dcbs_params.top_n` | `DCBS_TOP_N` | integer |",
            "| `dcbs_params.cache_size` | `DCBS_CACHE_SIZE` | integer |",
            "| `performance.batch_size` | `DCBS_BATCH_SIZE` | integer |",
            "| `performance.memory_limit_gb` | `DCBS_MEMORY_LIMIT_GB` | float |",
            "| `performance.timeout_seconds` | `DCBS_TIMEOUT_SECONDS` | integer |",
            "",
            "### Boolean Environment Variables",
            "",
            "Boolean values can be set using: `true`, `1`, `yes`, `on` (case-insensitive)",
            "",
            "## Configuration Templates",
            "",
            "### Development Configuration",
            "",
            "```yaml",
            "model_path: 'meta-llama/Llama-3.2-1B'",
            "benchmark: 'data/arc_easy_processed.json'",
            "output_dir: 'dev_results'",
            "limit: 10",
            "include_cot: true",
            "log_level: 'DEBUG'",
            "load_in_4bit: true",
            "",
            "dcbs_params:",
            "  k: 4",
            "  top_n: 20",
            "  cache_size: 100",
            "",
            "performance:",
            "  batch_size: 1",
            "  timeout_seconds: 300",
            "```",
            "",
            "### Production Configuration",
            "",
            "```yaml",
            "model_path: 'meta-llama/Llama-3.2-1B'",
            "benchmark: 'data/arc_easy_processed.json'",
            "output_dir: 'production_results'",
            "limit: null",
            "include_cot: true",
            "log_level: 'INFO'",
            "load_in_4bit: false",
            "",
            "dcbs_params:",
            "  k: 8",
            "  top_n: 50",
            "  cache_size: 1000",
            "",
            "performance:",
            "  batch_size: 1",
            "  memory_limit_gb: 16.0",
            "  timeout_seconds: 7200",
            "```",
            "",
            "### High-Performance Configuration",
            "",
            "```yaml",
            "model_path: 'meta-llama/Llama-3.2-1B'",
            "benchmark: 'data/arc_easy_processed.json'",
            "output_dir: 'hp_results'",
            "limit: null",
            "include_cot: true",
            "log_level: 'WARNING'",
            "load_in_4bit: true",
            "",
            "dcbs_params:",
            "  k: 16",
            "  top_n: 100",
            "  cache_size: 5000",
            "",
            "performance:",
            "  batch_size: 4",
            "  memory_limit_gb: 32.0",
            "  timeout_seconds: 14400",
            "```",
        ]

        with open(self.output_dir / "configuration_guide.md", 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))

    def _generate_performance_guide(self):
        """Generate performance optimization guide."""
        content = [
            "# Performance Optimization Guide",
            "",
            "This guide provides tips and strategies for optimizing DCBS evaluation performance.",
            "",
            "## Profiling and Monitoring",
            "",
            "### Built-in Profiling",
            "",
            "The DCBS framework includes comprehensive profiling tools:",
            "",
            "```python",
            "from src.profiler import PerformanceProfiler",
            "",
            "profiler = PerformanceProfiler()",
            "",
            "with profiler.profile_section('evaluation'):",
            "    # Your evaluation code here",
            "    pass",
            "",
            "# Get detailed report",
            "report = profiler.generate_report()",
            "print(report)",
            "```",
            "",
            "### Memory Monitoring",
            "",
            "Monitor memory usage to identify bottlenecks:",
            "",
            "```python",
            "from src.errors import report_memory_usage",
            "",
            "# Monitor memory at key points",
            "report_memory_usage('after_model_load')",
            "# ... evaluation code ...",
            "report_memory_usage('after_evaluation')",
            "```",
            "",
            "## Optimization Strategies",
            "",
            "### Model Loading Optimization",
            "",
            "1. **Use 4-bit Quantization**:",
            "   ```yaml",
            "   load_in_4bit: true",
            "   ```",
            "",
            "2. **CPU Fallback for Large Models**:",
            "   ```python",
            "   # Automatic fallback in error recovery",
            "   from src.errors import with_error_recovery",
            "   ",
            "   @with_error_recovery()",
            "   def load_model():",
            "       # Will automatically try CPU if GPU fails",
            "       pass",
            "   ```",
            "",
            "### Memory Optimization",
            "",
            "1. **Reduce Batch Size**:",
            "   ```yaml",
            "   performance:",
            "     batch_size: 1  # Start with 1, increase if memory allows",
            "   ```",
            "",
            "2. **Set Memory Limits**:",
            "   ```yaml",
            "   performance:",
            "     memory_limit_gb: 8.0  # Adjust based on available RAM",
            "   ```",
            "",
            "3. **Optimize Cache Size**:",
            "   ```yaml",
            "   dcbs_params:",
            "     cache_size: 500  # Reduce if memory is limited",
            "   ```",
            "",
            "### DCBS Parameter Tuning",
            "",
            "1. **Cluster Count (k)**:",
            "   - Smaller k (4-8): Faster, less diverse",
            "   - Larger k (16-32): Slower, more diverse",
            "",
            "2. **Top-N Selection**:",
            "   - Smaller top_n (20-50): Faster clustering",
            "   - Larger top_n (100-200): Better token coverage",
            "",
            "3. **Optimal Configurations**:",
            "   ```yaml",
            "   # Fast configuration",
            "   dcbs_params:",
            "     k: 4",
            "     top_n: 20",
            "   ",
            "   # Balanced configuration",
            "   dcbs_params:",
            "     k: 8",
            "     top_n: 50",
            "   ",
            "   # High-quality configuration",
            "   dcbs_params:",
            "     k: 16",
            "     top_n: 100",
            "   ```",
            "",
            "### Caching Optimization",
            "",
            "1. **Enable Caching for Repeated Evaluations**:",
            "   ```yaml",
            "   enable_caching: true",
            "   dcbs_params:",
            "     cache_size: 1000",
            "   ```",
            "",
            "2. **Disable Caching for One-off Evaluations**:",
            "   ```yaml",
            "   enable_caching: false",
            "   ```",
            "",
            "3. **Monitor Cache Performance**:",
            "   ```python",
            "   from src.profiler import CacheProfiler",
            "   ",
            "   cache_profiler = CacheProfiler()",
            "   # ... use during evaluation ...",
            "   stats = cache_profiler.get_stats()",
            "   print(f'Cache hit rate: {stats[\"hit_rate\"]:.2%}')",
            "   ```",
            "",
            "## Hardware-Specific Optimizations",
            "",
            "### GPU Optimization",
            "",
            "1. **Use Mixed Precision**:",
            "   ```python",
            "   # Enable in model loading",
            "   torch_dtype = torch.float16  # or torch.bfloat16",
            "   ```",
            "",
            "2. **Monitor GPU Memory**:",
            "   ```python",
            "   import torch",
            "   print(f'GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f}GB')",
            "   ```",
            "",
            "### CPU Optimization",
            "",
            "1. **Set Thread Count**:",
            "   ```python",
            "   import torch",
            "   torch.set_num_threads(4)  # Adjust based on CPU cores",
            "   ```",
            "",
            "2. **Use CPU-Optimized Models**:",
            "   ```yaml",
            "   model_path: 'model-name-cpu-optimized'",
            "   load_in_4bit: true",
            "   ```",
            "",
            "## Performance Benchmarks",
            "",
            "### Typical Performance Ranges",
            "",
            "| Configuration | Examples/Hour | Memory Usage | Quality |",
            "|---------------|---------------|--------------|---------|",
            "| Fast | 100-200 | 2-4GB | Good |",
            "| Balanced | 50-100 | 4-8GB | Better |",
            "| High-Quality | 20-50 | 8-16GB | Best |",
            "",
            "### Bottleneck Identification",
            "",
            "Common bottlenecks and solutions:",
            "",
            "1. **Model Loading (>30s)**:",
            "   - Use 4-bit quantization",
            "   - Consider smaller models",
            "",
            "2. **Token Generation (>5s per example)**:",
            "   - Reduce max_new_tokens",
            "   - Use greedy sampling for comparison",
            "",
            "3. **DCBS Clustering (>2s per example)**:",
            "   - Reduce k and top_n",
            "   - Enable caching",
            "",
            "4. **Memory Issues**:",
            "   - Reduce batch_size",
            "   - Enable 4-bit quantization",
            "   - Set memory limits",
            "",
            "## Monitoring and Alerts",
            "",
            "### Performance Monitoring",
            "",
            "```python",
            "from src.profiler import get_global_profiler",
            "",
            "profiler = get_global_profiler()",
            "",
            "# Check for bottlenecks",
            "bottlenecks = profiler.get_bottlenecks(threshold_seconds=5.0)",
            "if bottlenecks:",
            "    print('Performance bottlenecks detected:')",
            "    for section, duration in bottlenecks:",
            "        print(f'  {section}: {duration:.2f}s')",
            "```",
            "",
            "### Automated Optimization",
            "",
            "```python",
            "from src.errors import ErrorRecoveryManager",
            "",
            "# Automatic performance optimization",
            "recovery_manager = ErrorRecoveryManager()",
            "",
            "# Will automatically reduce batch size, enable quantization, etc.",
            "# when resource errors are detected",
            "```",
        ]

        with open(self.output_dir / "performance_guide.md", 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))

    def _generate_index(self):
        """Generate main documentation index."""
        content = [
            "# DCBS API Documentation",
            "",
            "Welcome to the Deterministic Category Based Sampling (DCBS) evaluation framework documentation.",
            "",
            "## Quick Start",
            "",
            "```python",
            "from src.evaluation_core import EvaluationConfig, EvaluationRunner",
            "",
            "config = EvaluationConfig(",
            "    model_name='meta-llama/Llama-3.2-1B',",
            "    benchmark_path='data/arc_easy_processed.json'",
            ")",
            "",
            "runner = EvaluationRunner(config)",
            "results = runner.run_evaluation()",
            "print(f'Accuracy: {results.accuracy:.2%}')",
            "```",
            "",
            "## Documentation Sections",
            "",
            "### API Reference",
            "",
            "- [Sampling Algorithms](dcbs_samplers.md) - Core sampling implementations",
            "- [Evaluation Framework](src_evaluation_core.md) - Main evaluation components",
            "- [Configuration Management](src_config_schema.md) - Configuration and validation",
            "- [Error Handling](src_errors.md) - Error management and recovery",
            "- [Performance Profiling](src_profiler.md) - Performance monitoring tools",
            "",
            "### Guides",
            "",
            "- [Usage Examples](usage_examples.md) - Comprehensive usage examples",
            "- [Configuration Guide](configuration_guide.md) - Configuration options and best practices",
            "- [Performance Guide](performance_guide.md) - Optimization tips and benchmarks",
            "",
            "## Key Features",
            "",
            "- **Multiple Sampling Methods**: Greedy, Top-p, Random, and DCBS",
            "- **Comprehensive Configuration**: YAML-based with environment variable support",
            "- **Error Recovery**: Automatic recovery from common failures",
            "- **Performance Profiling**: Built-in profiling and optimization suggestions",
            "- **Extensible Architecture**: Easy to add custom samplers and evaluators",
            "",
            "## Architecture Overview",
            "",
            "```",
            "DCBS Framework",
            "├── Samplers (dcbs.samplers)",
            "│   ├── GreedySampler",
            "│   ├── TopPSampler", 
            "│   ├── RandomSampler",
            "│   └── DCBSSampler",
            "├── Evaluation Core (src.evaluation_core)",
            "│   ├── EvaluationConfig",
            "│   ├── EvaluationRunner",
            "│   ├── ModelManager",
            "│   └── SamplerFactory",
            "├── Configuration (src.config_schema)",
            "│   ├── ConfigValidator",
            "│   └── EnvironmentVariableResolver",
            "├── Error Handling (src.errors)",
            "│   ├── Custom Exceptions",
            "│   └── ErrorRecoveryManager",
            "└── Profiling (src.profiler)",
            "    ├── PerformanceProfiler",
            "    ├── ModelProfiler",
            "    └── CacheProfiler",
            "```",
            "",
            "## Getting Help",
            "",
            "- Check the [Usage Examples](usage_examples.md) for common use cases",
            "- Review the [Configuration Guide](configuration_guide.md) for setup help",
            "- See the [Performance Guide](performance_guide.md) for optimization tips",
            "- Examine the API reference for detailed function documentation",
        ]

        with open(self.output_dir / "index.md", 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))


if __name__ == "__main__":
    generator = DocGenerator()
    generator.generate_all_docs() 