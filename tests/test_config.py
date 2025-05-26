"""
Unit tests for configuration loading and validation.

This module tests the configuration loading functionality
used by the evaluation framework.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation_core import load_benchmark_data
from src.config_builder import ConfigBuilder


class TestConfigLoading(unittest.TestCase):
    """Test configuration loading functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config_file = self.test_dir / "test_config.yaml"

    def test_load_benchmark_data(self):
        """Test that benchmark data loading works correctly."""
        # Create test benchmark data
        test_data = [
            {
                "id": "test_1", 
                "question": "What is 2+2?",
                "options": ["3", "4", "5", "6"],
                "correct_option": "2"
            }
        ]
        
        import json
        benchmark_file = self.test_dir / "test_benchmark.json"
        with open(benchmark_file, 'w') as f:
            json.dump(test_data, f)
        
        # Test loading
        loaded_data = load_benchmark_data(str(benchmark_file))
        self.assertEqual(len(loaded_data), 1)
        self.assertEqual(loaded_data[0]["id"], "test_1")

    def test_config_builder(self):
        """Test that config builder works correctly."""
        # Create test config with all required fields
        config_content = """
model_path: "test-model"
benchmark: "test_benchmark.json"

model:
  name: "test-model"

sampling:
  k_values: [8]
  top_n_values: [50]
  p_values: [0.9]
        """
        
        with open(self.config_file, 'w') as f:
            f.write(config_content)
        
        # Mock args object
        class MockArgs:
            def __init__(self):
                self.model = None
                self.top_p = None
                self.k = None
                self.top_n = None
                self.benchmark = None
                self.limit = None
                self.enable_caching = None
                self.load_in_4bit = None
                self.log_level = None
                self.no_cot = False
                self.output_dir = None
        
        args = MockArgs()
        config = ConfigBuilder.from_yaml_and_args(str(self.config_file), args)
        
        self.assertIsNotNone(config)
        self.assertEqual(config.model_name, "test-model")
