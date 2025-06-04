"""
Configuration schema validation and environment variable support.

This module provides schema validation for YAML configuration files
and support for environment variable substitution.
"""

import yaml
from typing import Any, Dict

from src.config_validation import ConfigValidator
from src.env_resolver import EnvironmentVariableResolver
from src.errors import ConfigurationError, ValidationError


class ConfigSchema:
    """Defines the schema for DCBS configuration files."""

    SCHEMA = {
        "model_path": {
            "type": str,
            "required": True,
            "description": "HuggingFace model name or path",
            "env_var": "DCBS_MODEL_PATH",
        },
        "benchmark": {
            "type": str,
            "required": True,
            "description": "Path to benchmark JSON file",
            "env_var": "DCBS_BENCHMARK_PATH",
        },
        "output_dir": {
            "type": str,
            "required": False,
            "default": "results",
            "description": "Output directory for results",
            "env_var": "DCBS_OUTPUT_DIR",
        },
        "limit": {
            "type": int,
            "required": False,
            "default": None,
            "description": "Limit number of examples for testing",
            "min_value": 1,
            "env_var": "DCBS_LIMIT",
        },
        "include_cot": {
            "type": bool,
            "required": False,
            "default": True,
            "description": "Enable chain-of-thought reasoning",
            "env_var": "DCBS_INCLUDE_COT",
        },
        "log_level": {
            "type": str,
            "required": False,
            "default": "INFO",
            "description": "Logging level",
            "choices": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "env_var": "DCBS_LOG_LEVEL",
        },
        "load_in_4bit": {
            "type": bool,
            "required": False,
            "default": False,
            "description": "Load model with 4-bit quantization",
            "env_var": "DCBS_LOAD_IN_4BIT",
        },
        "enable_caching": {
            "type": bool,
            "required": False,
            "default": True,
            "description": "Enable DCBS caching",
            "env_var": "DCBS_ENABLE_CACHING",
        },
        "debug_mode": {
            "type": bool,
            "required": False,
            "default": False,
            "description": "Enable debug mode for DCBS sampler",
            "env_var": "DCBS_DEBUG_MODE",
        },
        "enable_cluster_history": {
            "type": bool,
            "required": False,
            "default": False,
            "description": "Record cluster history and probabilities",
            "env_var": "DCBS_ENABLE_CLUSTER_HISTORY",
        },
        "p_values": {
            "type": list,
            "required": False,
            "default": [0.9],
            "description": "Top-p values for nucleus sampling",
            "item_type": float,
            "min_value": 0.0,
            "max_value": 1.0,
        },
        "dcbs_params": {
            "type": dict,
            "required": False,
            "default": {},
            "description": "DCBS-specific parameters",
            "schema": {
                "k": {
                    "type": int,
                    "required": False,
                    "default": 8,
                    "description": "Number of clusters for DCBS",
                    "min_value": 1,
                    "max_value": 100,
                    "env_var": "DCBS_K",
                },
                "top_n": {
                    "type": int,
                    "required": False,
                    "default": 50,
                    "description": "Top-n tokens to consider for clustering",
                    "min_value": 1,
                    "max_value": 1000,
                    "env_var": "DCBS_TOP_N",
                },
                "cache_size": {
                    "type": int,
                    "required": False,
                    "default": 1000,
                    "description": "Maximum cache size",
                    "min_value": 0,
                    "env_var": "DCBS_CACHE_SIZE",
                },
            },
        },
        "clustering_method": {
            "type": str,
            "required": False,
            "default": "dbscan",
            "description": "Clustering method for DCBS",
            "choices": ["kmeans", "dbscan", "hierarchical"],
            "env_var": "DCBS_CLUSTERING_METHOD",
        },
        "dbscan_eps": {
            "type": float,
            "required": False,
            "default": 0.3,
            "description": "DBSCAN epsilon parameter",
            "min_value": 0.01,
            "max_value": 10.0,
            "env_var": "DCBS_DBSCAN_EPS",
        },
        "dbscan_min_samples": {
            "type": int,
            "required": False,
            "default": 2,
            "description": "DBSCAN minimum samples parameter",
            "min_value": 1,
            "max_value": 100,
            "env_var": "DCBS_DBSCAN_MIN_SAMPLES",
        },
        "hierarchical_linkage": {
            "type": str,
            "required": False,
            "default": "average",
            "description": "Hierarchical clustering linkage method",
            "choices": ["average", "single", "complete", "ward"],
            "env_var": "DCBS_HIERARCHICAL_LINKAGE",
        },
        "performance": {
            "type": dict,
            "required": False,
            "default": {},
            "description": "Performance-related settings",
            "schema": {
                "batch_size": {
                    "type": int,
                    "required": False,
                    "default": 1,
                    "description": "Batch size for processing",
                    "min_value": 1,
                    "max_value": 100,
                    "env_var": "DCBS_BATCH_SIZE",
                },
                "memory_limit_gb": {
                    "type": float,
                    "required": False,
                    "default": None,
                    "description": "Memory limit in GB",
                    "min_value": 0.1,
                    "env_var": "DCBS_MEMORY_LIMIT_GB",
                },
                "timeout_seconds": {
                    "type": int,
                    "required": False,
                    "default": 3600,
                    "description": "Timeout for operations in seconds",
                    "min_value": 1,
                    "env_var": "DCBS_TIMEOUT_SECONDS",
                },
            },
        },
    }


def validate_config(config: Dict[str, Any], schema: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Validate configuration with environment variable resolution.
    
    Args:
        config: Configuration dictionary to validate
        schema: Optional custom schema (defaults to ConfigSchema.SCHEMA)
        
    Returns:
        Validated configuration with environment variables resolved
        
    Raises:
        ValidationError: If validation fails
        ConfigurationError: If environment variable resolution fails
    """
    if schema is None:
        schema = ConfigSchema.SCHEMA
    
    # Apply environment variable resolution
    config = EnvironmentVariableResolver.resolve_env_vars(config)
    
    # Apply environment variable overrides
    config = EnvironmentVariableResolver.apply_env_var_overrides(config, schema)
    
    # Validate configuration
    validator = ConfigValidator(schema)
    return validator.validate(config)


def generate_config_template() -> str:
    """Generate a template configuration file with documentation."""
    template = """# DCBS Evaluation Configuration Template
# This file contains all available configuration options with their descriptions
# Environment variables can be used with ${VAR_NAME} or ${VAR_NAME:default_value}

# Model configuration
model_path: "meta-llama/Llama-3.2-1B"  # HuggingFace model name or path
# Environment variable: DCBS_MODEL_PATH

# Benchmark data
benchmark: "data/arc_easy_full.json"  # Path to benchmark JSON file
# Environment variable: DCBS_BENCHMARK_PATH

# Output settings
output_dir: "results"  # Output directory for results
# Environment variable: DCBS_OUTPUT_DIR

# Evaluation settings
limit: null  # Limit number of examples (null for all)
# Environment variable: DCBS_LIMIT

include_cot: true  # Enable chain-of-thought reasoning
# Environment variable: DCBS_INCLUDE_COT

# Logging
log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
# Environment variable: DCBS_LOG_LEVEL

# Model loading
load_in_4bit: false  # Load model with 4-bit quantization
# Environment variable: DCBS_LOAD_IN_4BIT

# Caching
enable_caching: true  # Enable DCBS caching
# Environment variable: DCBS_ENABLE_CACHING

# Sampling parameters
p_values: [0.9]  # Top-p values for nucleus sampling

# DCBS-specific parameters
dcbs_params:
  k: 8  # Number of clusters
  # Environment variable: DCBS_K
  
  top_n: 50  # Top-n tokens to consider
  # Environment variable: DCBS_TOP_N
  
  cache_size: 1000  # Maximum cache size
  # Environment variable: DCBS_CACHE_SIZE

# Performance settings
performance:
  batch_size: 1  # Batch size for processing
  # Environment variable: DCBS_BATCH_SIZE
  
  memory_limit_gb: null  # Memory limit in GB (null for no limit)
  # Environment variable: DCBS_MEMORY_LIMIT_GB
  
  timeout_seconds: 3600  # Timeout for operations
  # Environment variable: DCBS_TIMEOUT_SECONDS
"""
    return template


def validate_config_file(config_path: str) -> Dict[str, Any]:
    """
    Load and validate a configuration file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ConfigurationError: If file cannot be loaded
        ValidationError: If validation fails
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in configuration file: {e}")
    
    if not isinstance(config, dict):
        raise ConfigurationError("Configuration file must contain a YAML dictionary")
    
    return validate_config(config) 