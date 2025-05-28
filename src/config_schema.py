"""
Configuration schema validation and environment variable support.

This module provides schema validation for YAML configuration files
and support for environment variable substitution.
"""

import os
import re
from typing import Any, Dict, List, Optional, Union

import yaml

from src.errors import ConfigurationError, ValidationError, eval_logger as logger


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


class EnvironmentVariableResolver:
    """Handles environment variable substitution in configuration values."""

    ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')

    @classmethod
    def resolve_env_vars(cls, value: Any) -> Any:
        """
        Resolve environment variables in configuration values.
        
        Args:
            value: Configuration value that may contain environment variables
            
        Returns:
            Value with environment variables resolved
        """
        if isinstance(value, str):
            return cls._resolve_string_env_vars(value)
        elif isinstance(value, dict):
            return {k: cls.resolve_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [cls.resolve_env_vars(item) for item in value]
        else:
            return value

    @classmethod
    def _resolve_string_env_vars(cls, value: str) -> str:
        """Resolve environment variables in a string value."""
        def replace_env_var(match):
            env_var = match.group(1)
            # Support default values: ${VAR:default}
            if ':' in env_var:
                var_name, default_value = env_var.split(':', 1)
                return os.getenv(var_name, default_value)
            else:
                env_value = os.getenv(env_var)
                if env_value is None:
                    raise ConfigurationError(
                        f"Environment variable '{env_var}' not found",
                        details={"variable": env_var, "value": value}
                    )
                return env_value

        return cls.ENV_VAR_PATTERN.sub(replace_env_var, value)

    @classmethod
    def apply_env_var_overrides(cls, config: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.
        
        Args:
            config: Current configuration
            schema: Configuration schema
            
        Returns:
            Configuration with environment variable overrides applied
        """
        result = config.copy()
        
        for key, field_schema in schema.items():
            env_var = field_schema.get("env_var")
            if env_var and env_var in os.environ:
                env_value = os.environ[env_var]
                
                # Convert environment variable value to appropriate type
                try:
                    if field_schema["type"] == bool:
                        result[key] = env_value.lower() in ("true", "1", "yes", "on")
                    elif field_schema["type"] == int:
                        result[key] = int(env_value)
                    elif field_schema["type"] == float:
                        result[key] = float(env_value)
                    elif field_schema["type"] == list:
                        # Assume comma-separated values
                        result[key] = [item.strip() for item in env_value.split(",")]
                    else:
                        result[key] = env_value
                        
                    logger.info(f"Applied environment override: {key} = {result[key]} (from {env_var})")
                except (ValueError, TypeError) as e:
                    raise ConfigurationError(
                        f"Invalid value for environment variable {env_var}: {env_value}",
                        details={"env_var": env_var, "value": env_value, "expected_type": field_schema["type"]}
                    )
            
            # Recursively apply to nested schemas
            if field_schema["type"] == dict and "schema" in field_schema:
                if key in result and isinstance(result[key], dict):
                    result[key] = cls.apply_env_var_overrides(result[key], field_schema["schema"])
        
        return result


class ConfigValidator:
    """Validates configuration against schema."""

    def __init__(self, schema: Dict[str, Any] = None):
        self.schema = schema or ConfigSchema.SCHEMA

    def validate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validated and normalized configuration
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            # Apply environment variable resolution
            config = EnvironmentVariableResolver.resolve_env_vars(config)
            
            # Apply environment variable overrides
            config = EnvironmentVariableResolver.apply_env_var_overrides(config, self.schema)
            
            # Validate against schema
            validated_config = self._validate_dict(config, self.schema, "root")
            
            logger.info("Configuration validation successful")
            return validated_config
            
        except Exception as e:
            if isinstance(e, (ValidationError, ConfigurationError)):
                raise
            else:
                raise ValidationError(f"Configuration validation failed: {str(e)}") from e

    def _validate_dict(self, config: Dict[str, Any], schema: Dict[str, Any], path: str) -> Dict[str, Any]:
        """Validate a dictionary against its schema."""
        result = {}
        
        # Check required fields
        for key, field_schema in schema.items():
            if field_schema.get("required", False) and key not in config:
                raise ValidationError(
                    f"Required field '{key}' missing",
                    details={"path": path, "field": key}
                )
        
        # Validate each field
        for key, field_schema in schema.items():
            if key in config:
                value = config[key]
                result[key] = self._validate_field(value, field_schema, f"{path}.{key}")
            elif "default" in field_schema:
                result[key] = field_schema["default"]
        
        # Check for unknown fields
        unknown_fields = set(config.keys()) - set(schema.keys())
        if unknown_fields:
            logger.warning(f"Unknown configuration fields: {unknown_fields}")
        
        return result

    def _validate_field(self, value: Any, field_schema: Dict[str, Any], path: str) -> Any:
        """Validate a single field against its schema."""
        expected_type = field_schema["type"]
        
        # Handle None values for optional fields with None defaults
        if value is None and field_schema.get("default") is None and not field_schema.get("required", False):
            return None
        
        # Type validation (skip for None values that are allowed)
        if value is not None:
            if expected_type == dict and not isinstance(value, dict):
                raise ValidationError(
                    f"Field '{path}' must be a dictionary",
                    details={"path": path, "value": value, "expected_type": "dict"}
                )
            elif expected_type == list and not isinstance(value, list):
                raise ValidationError(
                    f"Field '{path}' must be a list",
                    details={"path": path, "value": value, "expected_type": "list"}
                )
            elif expected_type in (str, int, float, bool) and not isinstance(value, expected_type):
                raise ValidationError(
                    f"Field '{path}' must be of type {expected_type.__name__}",
                    details={"path": path, "value": value, "expected_type": expected_type.__name__}
                )
        
        # Choice validation (skip for None values)
        if value is not None and "choices" in field_schema and value not in field_schema["choices"]:
            raise ValidationError(
                f"Field '{path}' must be one of {field_schema['choices']}",
                details={"path": path, "value": value, "choices": field_schema["choices"]}
            )
        
        # Range validation (only for non-list types and non-None values)
        if value is not None and expected_type != list:
            if "min_value" in field_schema and value < field_schema["min_value"]:
                raise ValidationError(
                    f"Field '{path}' must be >= {field_schema['min_value']}",
                    details={"path": path, "value": value, "min_value": field_schema["min_value"]}
                )
            
            if "max_value" in field_schema and value > field_schema["max_value"]:
                raise ValidationError(
                    f"Field '{path}' must be <= {field_schema['max_value']}",
                    details={"path": path, "value": value, "max_value": field_schema["max_value"]}
                )
        
        # Nested validation (skip for None values)
        if value is not None:
            if expected_type == dict and "schema" in field_schema:
                return self._validate_dict(value, field_schema["schema"], path)
            elif expected_type == list and "item_type" in field_schema:
                return self._validate_list(value, field_schema, path)
        
        return value

    def _validate_list(self, value: List[Any], field_schema: Dict[str, Any], path: str) -> List[Any]:
        """Validate a list field."""
        item_type = field_schema["item_type"]
        result = []
        
        for i, item in enumerate(value):
            item_path = f"{path}[{i}]"
            
            if not isinstance(item, item_type):
                raise ValidationError(
                    f"List item at '{item_path}' must be of type {item_type.__name__}",
                    details={"path": item_path, "value": item, "expected_type": item_type.__name__}
                )
            
            # Range validation for list items
            if "min_value" in field_schema and item < field_schema["min_value"]:
                raise ValidationError(
                    f"List item at '{item_path}' must be >= {field_schema['min_value']}",
                    details={"path": item_path, "value": item, "min_value": field_schema["min_value"]}
                )
            
            if "max_value" in field_schema and item > field_schema["max_value"]:
                raise ValidationError(
                    f"List item at '{item_path}' must be <= {field_schema['max_value']}",
                    details={"path": item_path, "value": item, "max_value": field_schema["max_value"]}
                )
            
            result.append(item)
        
        return result


def generate_config_template() -> str:
    """Generate a template configuration file with documentation."""
    template = """# DCBS Evaluation Configuration Template
# This file contains all available configuration options with their descriptions
# Environment variables can be used with ${VAR_NAME} or ${VAR_NAME:default_value}

# Model configuration
model_path: "meta-llama/Llama-3.2-1B"  # HuggingFace model name or path
# Environment variable: DCBS_MODEL_PATH

# Benchmark data
benchmark: "data/arc_easy_processed.json"  # Path to benchmark JSON file
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
    
    validator = ConfigValidator()
    return validator.validate(config) 