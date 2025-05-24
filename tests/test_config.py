"""
Tests for configuration loading
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

# Add the parent directory to the path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_dcbs_eval import load_config


class TestConfig(unittest.TestCase):
    def test_valid_config(self):
        """Test loading a valid configuration file"""
        # Create a temporary config file
        config_data = {
            "model_path": "test-model",
            "clusters": 5,
            "top_n": 10,
            "benchmark": "test-bench.json",
            "output_file": "test-output.csv",
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as temp:
            yaml.dump(config_data, temp)
            temp_path = temp.name

        try:
            # Load the config
            config = load_config(temp_path)

            # Verify all fields are present
            self.assertEqual(config["model_path"], "test-model")
            self.assertEqual(config["clusters"], 5)
            self.assertEqual(config["top_n"], 10)
            self.assertEqual(config["benchmark"], "test-bench.json")
            self.assertEqual(config["output_file"], "test-output.csv")
        finally:
            # Clean up the temporary file
            os.unlink(temp_path)

    def test_invalid_config(self):
        """Test loading an invalid configuration file"""
        # Create a non-existent file path
        temp_path = "/path/to/nonexistent/config.yaml"

        # This should raise an exception
        with self.assertRaises(SystemExit):
            load_config(temp_path)


if __name__ == "__main__":
    unittest.main()
