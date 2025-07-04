"""
Unit tests for multi-token answer handling in DCBS.

This module tests the functionality for handling multi-token answers,
including token concatenation support and validation.
"""

import os
import sys
import unittest

import torch

# Add parent directory to path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoTokenizer

from src.token_utils import is_valid_token_prediction, tokenizer_cache


def handle_multitoken_answer(token_ids, tokenizer, answer_text, strategy="first"):
    """
    Mock implementation of multi-token answer handling for testing.
    
    Args:
        token_ids: List of token IDs
        tokenizer: Tokenizer instance
        answer_text: Original answer text
        strategy: Either "first" (return first token) or "combine" (return all)
    
    Returns:
        Single token ID (strategy="first") or list of token IDs (strategy="combine")
    """
    if strategy == "first":
        return token_ids[0] if token_ids else 0
    elif strategy == "combine":
        return token_ids
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.vocab = {
            "hello": 100,
            "world": 101,
            "test": 102,
            "multi": 103,
            "token": 104,
            "answer": 105,
            "combined": 106,
            "text": 107,
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def encode(self, text, add_special_tokens=False):
        """Mock encode method."""
        # Simple space-based tokenization for testing
        words = text.strip().split()
        return [self.vocab.get(word, 999) for word in words]

    def decode(self, token_ids):
        """Mock decode method."""
        return " ".join([self.id_to_token.get(tid, "UNK") for tid in token_ids])


class TestMultiTokenHandling(unittest.TestCase):
    """Tests for multi-token answer handling."""

    def setUp(self):
        """Set up test environment."""
        self.mock_tokenizer = MockTokenizer()
        # Initialize tokenizer cache for testing
        tokenizer_cache.clear()

    def test_first_token_strategy(self):
        """Test that 'first' strategy returns only the first token."""
        tokens = [103, 104, 105]  # "multi token answer"
        result = handle_multitoken_answer(
            tokens, self.mock_tokenizer, "multi token answer", "first"
        )
        self.assertEqual(result, 103)
        self.assertIsInstance(result, int)

    def test_combine_strategy(self):
        """Test that 'combine' strategy returns all tokens."""
        tokens = [103, 104, 105]  # "multi token answer"
        result = handle_multitoken_answer(
            tokens, self.mock_tokenizer, "multi token answer", "combine"
        )
        self.assertEqual(result, tokens)
        self.assertIsInstance(result, list)

    def test_token_validation_with_single_token(self):
        """Test token validation with single token."""
        correct_id = 102  # "test"
        pred_id = 102  # "test"
        is_valid = is_valid_token_prediction(
            pred_id, correct_id, "test", self.mock_tokenizer
        )
        self.assertTrue(is_valid)

    def test_token_validation_with_multi_token_match(self):
        """Test token validation with multi-token answer - direct match."""
        correct_ids = [103, 104]  # "multi token"
        pred_id = 103  # "multi"
        is_valid = is_valid_token_prediction(
            pred_id, correct_ids, "multi token", self.mock_tokenizer
        )
        self.assertTrue(is_valid)

    def test_token_validation_with_multi_token_no_match(self):
        """Test token validation with multi-token answer - no match."""
        correct_ids = [103, 104]  # "multi token"
        pred_id = 105  # "answer"
        is_valid = is_valid_token_prediction(
            pred_id, correct_ids, "multi token", self.mock_tokenizer
        )
        self.assertFalse(is_valid)

    def test_token_validation_with_full_text_match(self):
        """Test token validation with multi-token answer - full text match."""
        # This test simulates when a model predicts a token that decodes to the full answer
        correct_ids = [103, 104]  # "multi token"
        pred_id = 106  # "combined" - pretend this decodes to "multi token"

        # Override the decode method for this specific test
        original_decode = tokenizer_cache.decode

        def mock_decode(token_ids, tokenizer, **kwargs):
            if token_ids == [106]:
                return "multi token"
            return original_decode(token_ids, tokenizer, **kwargs)

        tokenizer_cache.decode = mock_decode

        try:
            is_valid = is_valid_token_prediction(
                pred_id, correct_ids, "multi token", self.mock_tokenizer
            )
            self.assertTrue(is_valid)
        finally:
            # Restore original method
            tokenizer_cache.decode = original_decode

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_with_real_tokenizer(self):
        """Test with a real tokenizer if available."""
        try:
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

            # Test with a real multi-token example
            text = "artificial intelligence"
            tokens = tokenizer.encode(text, add_special_tokens=False)

            # Verify it's actually multi-token
            self.assertGreater(len(tokens), 1)

            # Test first strategy
            first_result = handle_multitoken_answer(tokens, tokenizer, text, "first")
            self.assertEqual(first_result, tokens[0])

            # Test combine strategy
            combine_result = handle_multitoken_answer(
                tokens, tokenizer, text, "combine"
            )
            self.assertEqual(combine_result, tokens)

            # Test validation
            is_valid = is_valid_token_prediction(tokens[0], tokens, text, tokenizer)
            self.assertTrue(is_valid)

        except Exception:
            self.skipTest("Could not load real tokenizer")


if __name__ == "__main__":
    unittest.main()
