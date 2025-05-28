#!/usr/bin/env python3
"""
Unit tests for the --limit parameter in the DCBS evaluation framework.
Tests to ensure that the --limit parameter correctly limits the number of examples processed.
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Add the parent directory to the path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import shutil

import pandas as pd

from src.evaluation_core import EvaluationConfig, EvaluationRunner, load_benchmark_data


class TestLimit(unittest.TestCase):
    """Test case for the limit parameter in the evaluation framework."""

    def setUp(self):
        """Set up temporary files for testing."""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()

        # Set up paths for test files
        self.output_csv = os.path.join(self.temp_dir, "test_output.csv")

    def tearDown(self):
        """Clean up temporary files after testing."""
        # Clean up temporary directory and files
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_evaluation_runner_with_limit(self):
        """Test that EvaluationRunner respects the limit parameter."""
        # Create test data with more examples than our limit
        test_data = [
            {
                "id": f"test_{i}",
                "question": f"What is {i}+1?",
                "options": [str(i), str(i+1), str(i+2), str(i+3)],
                "correct_option": "2"
            }
            for i in range(10)
        ]
        
        # Create test benchmark file
        benchmark_file = os.path.join(self.temp_dir, "test_benchmark.json")
        with open(benchmark_file, 'w') as f:
            json.dump(test_data, f)
        
        # Create config with limit
        limit = 3
        config = EvaluationConfig(
            model_name="test-model",
            benchmark_path=benchmark_file,
            output_dir=self.temp_dir,
            limit=limit
        )
        
        # Mock the model loading and processing
        with patch('src.evaluation_core.runner.ModelManager') as mock_model_mgr, \
             patch('src.evaluation_core.runner.ExampleProcessor') as mock_processor:
            
            # Setup mocks
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_context = Mock()
            mock_model_mgr.return_value.load_model.return_value = (mock_model, mock_tokenizer, mock_context)
            
            mock_processor_instance = Mock()
            mock_processor.return_value = mock_processor_instance
            mock_processor_instance.process_example.return_value = {
                "id": "test_1",
                "logits": Mock(),
                "filter_tokens": {1, 2, 3, 4},
                "correct_id": 2,
                "correct_answer": "B",
                "answer_probs": {"A": 0.2, "B": 0.5, "C": 0.2, "D": 0.1}
            }
            
            # Run evaluation
            runner = EvaluationRunner(config)
            results = runner.run_evaluation(test_data)
            
            # Verify that statistics reflect the limited number of examples
            for method_stats in results["statistics"].values():
                self.assertEqual(method_stats["total"], limit)

    def test_load_benchmark_data_functionality(self):
        """Test that benchmark data loading works correctly."""
        # Create test data
        test_data = [
            {
                "id": "test_1",
                "question": "What is 2+2?",
                "options": ["3", "4", "5", "6"],
                "correct_option": "2"
            },
            {
                "id": "test_2", 
                "question": "What is 3+3?",
                "options": ["5", "6", "7", "8"],
                "correct_option": "2"
            }
        ]
        
        # Save to temporary file
        benchmark_file = os.path.join(self.temp_dir, "test_data.json")
        with open(benchmark_file, 'w') as f:
            json.dump(test_data, f)
        
        # Load and verify
        loaded_data = load_benchmark_data(benchmark_file)
        self.assertEqual(len(loaded_data), 2)
        self.assertEqual(loaded_data[0]["id"], "test_1")
        self.assertEqual(loaded_data[1]["question"], "What is 3+3?")

    def test_config_limit_parameter(self):
        """Test that the EvaluationConfig correctly handles limit parameter."""
        config = EvaluationConfig(
            model_name="test-model",
            benchmark_path="test.json",
            output_dir="test_output",
            limit=5
        )
        
        self.assertEqual(config.limit, 5)
        
        # Test without limit
        config_no_limit = EvaluationConfig(
            model_name="test-model",
            benchmark_path="test.json",
            output_dir="test_output"
        )
        
        self.assertIsNone(config_no_limit.limit)


if __name__ == "__main__":
    unittest.main()
