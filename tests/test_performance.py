"""
Unit tests for performance characteristics of sampling methods.

This module tests the performance and timing behavior of different
sampling strategies in the evaluation pipeline.
"""

import os
import sys
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

# Add parent directory to path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation_core.runner import EvaluationRunner
from src.evaluation_core.example_processor import ExampleProcessor
from src.evaluation_core.config import EvaluationConfig
from dcbs import SamplingContext, GreedySampler, DCBSSampler


class TestPerformance(unittest.TestCase):
    """Test performance characteristics of different sampling methods."""

    def setUp(self):
        """Set up test environment."""
        self.test_config = EvaluationConfig(
            model_name="test-model",
            benchmark_path="test.json",
            limit=10,
            top_p=0.9,
            k=8,
            top_n=50,
        )

    def test_evaluation_runner_performance(self):
        """Test that evaluation runner handles performance measurement correctly."""
        # Mock data
        mock_data = [
            {
                "id": "test_1",
                "question": "Test question?",
                "options": ["A", "B", "C", "D"],
                "correct_option": "1"
            }
        ]
        
        # Mock the runner components
        with patch('src.evaluation_core.runner.ModelManager') as mock_model_mgr, \
             patch('src.evaluation_core.runner.ExampleProcessor') as mock_processor:
            
            # Setup mocks
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_context = SamplingContext()
            mock_model_mgr.return_value.load_model.return_value = (mock_model, mock_tokenizer, mock_context)
            
            mock_processor_instance = Mock()
            mock_processor.return_value = mock_processor_instance
            mock_processor_instance.process_example.return_value = {
                "id": "test_1",
                "logits": torch.randn(100),
                "filter_tokens": {1, 2, 3, 4},
                "correct_id": 1,
                "correct_answer": "A",
                "answer_probs": {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1}
            }
            
            # Create runner and test
            runner = EvaluationRunner(self.test_config)
            results = runner.run_evaluation(mock_data)
            
            # Verify results structure
            self.assertIn("statistics", results)
            self.assertIn("config", results)

    def test_example_processor_timing(self):
        """Test that example processing timing is reasonable."""
        # Mock model components
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_context = SamplingContext()
        mock_sampler = GreedySampler()
        
        # Mock tokenizer methods
        mock_tokenizer.apply_chat_template.return_value = "test prompt"
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "A"
        
        # Mock model forward pass
        mock_outputs = Mock()
        mock_outputs.logits = torch.randn(1, 10, 100)
        mock_model.return_value = mock_outputs
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        
        processor = ExampleProcessor(mock_model, mock_tokenizer, mock_context, mock_sampler)
        
        # Test processing timing
        test_example = {
            "id": "timing_test",
            "question": "What is 2+2?",
            "options": ["3", "4", "5", "6"],
            "correct_option": "2"
        }
        
        start_time = time.time()
        result = processor.process_example(test_example, include_cot=False)
        elapsed = time.time() - start_time
        
        # Verify reasonable timing (should be very fast with mocks)
        self.assertLess(elapsed, 1.0)  # Should complete in under 1 second
        self.assertIn("logits", result)

    def test_sampler_performance_comparison(self):
        """Test relative performance of different samplers."""
        # Create test logits
        logits = torch.randn(1000)
        filter_tokens = set(range(10))
        context = SamplingContext()
        
        # Time greedy sampling
        greedy_sampler = GreedySampler()
        start_time = time.time()
        for _ in range(100):
            greedy_sampler.sample(logits, filter_tokens)
        greedy_time = time.time() - start_time
        
        # Verify greedy is fast
        self.assertLess(greedy_time, 1.0)  # Should be very fast


if __name__ == "__main__":
    unittest.main()
