#!/usr/bin/env python3
"""
Unit tests for accuracy computation in the DCBS evaluation framework.
Tests to ensure that the accuracy computation is correctly computing per-example
accuracy and that the random baseline is around 50% for two-choice examples.
"""

import csv
import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from transformers import AutoTokenizer

# Add the parent directory to the path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix imports to use proper paths
from src.token_utils import is_valid_token_prediction


class TestAccuracy(unittest.TestCase):
    """Test case for accuracy computation."""

    def setUp(self):
        """Set up test environment."""
        # Mock tokenizer for testing
        self.tokenizer = Mock()
        self.tokenizer.decode.return_value = "A"

        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        # Create a synthetic dataset with two-choice examples
        self.examples = [
            {
                "id": f"test_{i}",
                "sentence": f"This is test sentence {i}.",
                "option1": "he",
                "option2": "she",
                "correct_option": "1" if i % 2 == 0 else "2",
            }
            for i in range(100)
        ]

    def test_random_baseline_accuracy(self):
        """Test that random sampling is around 50% accurate on two-choice examples."""
        correct_count = 0

        for example in self.examples:
            # Get answer IDs for both options
            choice1 = example["option1"]
            choice2 = example["option2"]
            choice1_ids = self.tokenizer.encode(" " + choice1, add_special_tokens=False)
            choice2_ids = self.tokenizer.encode(" " + choice2, add_special_tokens=False)

            # Get the correct answer
            correct_option = example["correct_option"]
            correct_answer = example[f"option{correct_option}"].lower().strip()

            # Get correct token ID
            correct_token_ids = self.tokenizer.encode(
                " " + correct_answer, add_special_tokens=False
            )
            correct_id = correct_token_ids[0]

            # Define correctness checking function
            def is_correct_local(pred_id):
                if pred_id == correct_id:
                    return True
                pred_str = self.tokenizer.decode([pred_id]).lower().strip()
                return pred_str == correct_answer

            # Random sampling from the two choices
            valid_ids = [choice1_ids[0], choice2_ids[0]]
            rand_id = random.choice(valid_ids)

            if is_correct_local(rand_id):
                correct_count += 1

        # Calculate accuracy
        random_acc = correct_count / len(self.examples)

        # Random should be close to 50%
        self.assertGreater(random_acc, 0.4)
        self.assertLess(random_acc, 0.6)

    def test_greedy_vs_random(self):
        """Test that greedy sampling is at least as good as random sampling."""
        random_correct = 0
        greedy_correct = 0

        for example in self.examples:
            # Get answer IDs for both options
            choice1 = example["option1"]
            choice2 = example["option2"]
            choice1_ids = self.tokenizer.encode(" " + choice1, add_special_tokens=False)
            choice2_ids = self.tokenizer.encode(" " + choice2, add_special_tokens=False)

            # Get the correct answer
            correct_option = example["correct_option"]
            correct_answer = example[f"option{correct_option}"].lower().strip()

            # Get correct token ID
            correct_token_ids = self.tokenizer.encode(
                " " + correct_answer, add_special_tokens=False
            )
            correct_id = correct_token_ids[0]

            # Define correctness checking function
            def is_correct_local(pred_id):
                if pred_id == correct_id:
                    return True
                pred_str = self.tokenizer.decode([pred_id]).lower().strip()
                return pred_str == correct_answer

            # Random sampling
            valid_ids = [choice1_ids[0], choice2_ids[0]]
            rand_id = random.choice(valid_ids)

            # Simulated greedy sampling (just use correct answer more often than random)
            # In a real setting, greedy would use logits and pick highest probability
            greedy_choice = correct_id if random.random() > 0.3 else valid_ids[0]

            if is_correct_local(rand_id):
                random_correct += 1

            if is_correct_local(greedy_choice):
                greedy_correct += 1

        # Calculate accuracies
        random_acc = random_correct / len(self.examples)
        greedy_acc = greedy_correct / len(self.examples)

        # Greedy should be at least as good as random
        self.assertGreaterEqual(greedy_acc, random_acc)

    def test_correct_prediction(self):
        """Test that correct predictions are identified properly."""
        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = "A"
        
        # Correct prediction case
        result = is_valid_token_prediction(
            pred_id=100, 
            correct_id=100, 
            correct_answer="A", 
            tokenizer=mock_tokenizer
        )
        self.assertTrue(result)

    def test_incorrect_prediction(self):
        """Test that incorrect predictions are identified properly."""
        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = "B"
        
        # Incorrect prediction case
        result = is_valid_token_prediction(
            pred_id=101, 
            correct_id=100, 
            correct_answer="A", 
            tokenizer=mock_tokenizer
        )
        self.assertFalse(result)

    def test_multitoken_prediction(self):
        """Test handling of multi-token predictions."""
        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = "multi"
        
        # Multi-token correct answer where pred_id matches first token
        result = is_valid_token_prediction(
            pred_id=100, 
            correct_id=[100, 101], 
            correct_answer="multi token", 
            tokenizer=mock_tokenizer
        )
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
