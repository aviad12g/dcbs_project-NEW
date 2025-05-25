"""
Unit tests for configuration schema validation.

Tests the ConfigValidator, EnvironmentVariableResolver, and related
configuration validation functionality.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from src.config_schema import (
    ConfigValidator,
    EnvironmentVariableResolver,
    generate_config_template,
    validate_config_file,
)
from src.errors import ConfigurationError, ValidationError


class TestEnvironmentVariableResolver(unittest.TestCase):
    """Test environment variable resolution functionality."""

    def test_resolve_simple_env_var(self):
        """Test resolving a simple environment variable."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = EnvironmentVariableResolver.resolve_env_vars("${TEST_VAR}")
            self.assertEqual(result, "test_value")

    def test_resolve_env_var_with_default(self):
        """Test resolving environment variable with default value."""
        result = EnvironmentVariableResolver.resolve_env_vars("${NONEXISTENT_VAR:default}")
        self.assertEqual(result, "default")

    def test_resolve_missing_env_var_raises_error(self):
        """Test that missing environment variable raises ConfigurationError."""
        with self.assertRaises(ConfigurationError):
            EnvironmentVariableResolver.resolve_env_vars("${NONEXISTENT_VAR}")

    def test_resolve_nested_structures(self):
        """Test resolving environment variables in nested data structures."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            data = {
                "key1": "${TEST_VAR}",
                "key2": ["${TEST_VAR}", "static"],
                "key3": {"nested": "${TEST_VAR}"}
            }
            result = EnvironmentVariableResolver.resolve_env_vars(data)
            expected = {
                "key1": "test_value",
                "key2": ["test_value", "static"],
                "key3": {"nested": "test_value"}
            }
            self.assertEqual(result, expected)

    def test_apply_env_var_overrides(self):
        """Test applying environment variable overrides."""
        schema = {
            "test_field": {
                "type": str,
                "env_var": "TEST_OVERRIDE"
            },
            "int_field": {
                "type": int,
                "env_var": "INT_OVERRIDE"
            },
            "bool_field": {
                "type": bool,
                "env_var": "BOOL_OVERRIDE"
            }
        }
        
        config = {"test_field": "original"}
        
        with patch.dict(os.environ, {
            "TEST_OVERRIDE": "overridden",
            "INT_OVERRIDE": "42",
            "BOOL_OVERRIDE": "true"
        }):
            result = EnvironmentVariableResolver.apply_env_var_overrides(config, schema)
            
        self.assertEqual(result["test_field"], "overridden")
        self.assertEqual(result["int_field"], 42)
        self.assertEqual(result["bool_field"], True)


class TestConfigValidator(unittest.TestCase):
    """Test configuration validation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = ConfigValidator()
        self.minimal_valid_config = {
            "model_path": "test-model",
            "benchmark": "test-benchmark.json"
        }

    def test_validate_minimal_config(self):
        """Test validation of minimal valid configuration."""
        result = self.validator.validate(self.minimal_valid_config)
        self.assertIn("model_path", result)
        self.assertIn("benchmark", result)
        self.assertEqual(result["model_path"], "test-model")

    def test_validate_missing_required_field(self):
        """Test validation fails for missing required fields."""
        config = {"model_path": "test-model"}  # Missing benchmark
        with self.assertRaises(ValidationError) as cm:
            self.validator.validate(config)
        self.assertIn("benchmark", str(cm.exception))

    def test_validate_type_errors(self):
        """Test validation fails for incorrect types."""
        config = {
            "model_path": "test-model",
            "benchmark": "test-benchmark.json",
            "limit": "not_an_int"  # Should be int
        }
        with self.assertRaises(ValidationError):
            self.validator.validate(config)

    def test_validate_range_constraints(self):
        """Test validation of range constraints."""
        config = {
            "model_path": "test-model",
            "benchmark": "test-benchmark.json",
            "limit": -1  # Should be >= 1
        }
        with self.assertRaises(ValidationError):
            self.validator.validate(config)

    def test_validate_choice_constraints(self):
        """Test validation of choice constraints."""
        config = {
            "model_path": "test-model",
            "benchmark": "test-benchmark.json",
            "log_level": "INVALID"  # Should be one of the valid choices
        }
        with self.assertRaises(ValidationError):
            self.validator.validate(config)

    def test_validate_nested_structures(self):
        """Test validation of nested configuration structures."""
        config = {
            "model_path": "test-model",
            "benchmark": "test-benchmark.json",
            "dcbs_params": {
                "k": 5,
                "top_n": 25
            }
        }
        result = self.validator.validate(config)
        self.assertEqual(result["dcbs_params"]["k"], 5)
        self.assertEqual(result["dcbs_params"]["top_n"], 25)

    def test_validate_list_fields(self):
        """Test validation of list fields."""
        config = {
            "model_path": "test-model",
            "benchmark": "test-benchmark.json",
            "p_values": [0.5, 0.9]
        }
        result = self.validator.validate(config)
        self.assertEqual(result["p_values"], [0.5, 0.9])

    def test_validate_list_item_constraints(self):
        """Test validation of list item constraints."""
        config = {
            "model_path": "test-model",
            "benchmark": "test-benchmark.json",
            "p_values": [1.5]  # Should be <= 1.0
        }
        with self.assertRaises(ValidationError):
            self.validator.validate(config)

    def test_default_values_applied(self):
        """Test that default values are applied for missing optional fields."""
        result = self.validator.validate(self.minimal_valid_config)
        self.assertEqual(result["output_dir"], "results")
        self.assertEqual(result["include_cot"], True)
        self.assertEqual(result["log_level"], "INFO")


class TestConfigFileValidation(unittest.TestCase):
    """Test configuration file loading and validation."""

    def test_validate_valid_config_file(self):
        """Test validation of a valid configuration file."""
        config_content = """
model_path: "test-model"
benchmark: "test-benchmark.json"
output_dir: "test-results"
limit: 10
include_cot: true
log_level: "DEBUG"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            temp_path = f.name
            
        try:
            result = validate_config_file(temp_path)
            self.assertEqual(result["model_path"], "test-model")
            self.assertEqual(result["limit"], 10)
            self.assertEqual(result["log_level"], "DEBUG")
        finally:
            try:
                os.unlink(temp_path)
            except (OSError, PermissionError):
                pass  # Ignore cleanup errors on Windows

    def test_validate_invalid_yaml_file(self):
        """Test validation fails for invalid YAML."""
        config_content = """
model_path: "test-model"
benchmark: "test-benchmark.json"
invalid_yaml: [unclosed list
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            temp_path = f.name
            
        try:
            with self.assertRaises(ConfigurationError):
                validate_config_file(temp_path)
        finally:
            try:
                os.unlink(temp_path)
            except (OSError, PermissionError):
                pass  # Ignore cleanup errors on Windows

    def test_validate_nonexistent_file(self):
        """Test validation fails for nonexistent file."""
        with self.assertRaises(ConfigurationError):
            validate_config_file("nonexistent_file.yaml")

    def test_validate_non_dict_config(self):
        """Test validation fails for non-dictionary configuration."""
        config_content = "- item1\n- item2"  # YAML list instead of dict
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            temp_path = f.name
            
        try:
            with self.assertRaises(ConfigurationError):
                validate_config_file(temp_path)
        finally:
            try:
                os.unlink(temp_path)
            except (OSError, PermissionError):
                pass  # Ignore cleanup errors on Windows


class TestConfigTemplate(unittest.TestCase):
    """Test configuration template generation."""

    def test_generate_config_template(self):
        """Test that configuration template is generated correctly."""
        template = generate_config_template()
        self.assertIsInstance(template, str)
        self.assertIn("model_path:", template)
        self.assertIn("benchmark:", template)
        self.assertIn("DCBS_MODEL_PATH", template)
        self.assertIn("Environment variable:", template)


if __name__ == "__main__":
    unittest.main() 