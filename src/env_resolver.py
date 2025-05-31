"""
Environment variable resolution utilities.

This module provides functionality for resolving environment variables
in configuration values and applying environment variable overrides.
"""

import os
import re
from typing import Any, Dict

from src.errors import ConfigurationError, eval_logger as logger


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