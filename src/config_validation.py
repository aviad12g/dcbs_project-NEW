"""
Configuration validation utilities.

This module provides validation logic for configuration values and schemas.
"""

from typing import Any, Dict, List

from src.errors import ValidationError, eval_logger as logger


class ConfigValidator:
    """Validates configuration against schema."""

    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema

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
            # Validate against schema
            validated_config = self._validate_dict(config, self.schema, "root")
            
            logger.info("Configuration validation successful")
            return validated_config
            
        except Exception as e:
            if isinstance(e, ValidationError):
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