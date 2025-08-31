"""
Tests for output types and data structures.
"""

import unittest
from pathlib import Path
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from messages_completion.output_types import (
    TokenInfo, 
    CompletionResult, 
    BatchCompletionResult
)


class TestTokenInfo(unittest.TestCase):
    """Test TokenInfo data structure."""
    
    def test_valid_token_info(self):
        """Test creating valid token info."""
        token = TokenInfo(
            token_id=42,
            token_text="hello",
            logprob=-0.5,
            prob=0.6
        )
        
        self.assertEqual(token.token_id, 42)
        self.assertEqual(token.token_text, "hello")
        self.assertEqual(token.logprob, -0.5)
        self.assertEqual(token.prob, 0.6)
    
    def test_invalid_probability(self):
        """Test invalid probability values."""
        # Probability > 1
        with self.assertRaises(ValueError):
            TokenInfo(token_id=1, token_text="test", prob=1.5)
        
        # Negative probability
        with self.assertRaises(ValueError):
            TokenInfo(token_id=1, token_text="test", prob=-0.1)
    
    def test_optional_fields(self):
        """Test token info with optional fields."""
        token = TokenInfo(
            token_id=42,
            token_text="hello"
        )
        
        self.assertEqual(token.token_id, 42)
        self.assertEqual(token.token_text, "hello")
        self.assertIsNone(token.logprob)
        self.assertIsNone(token.prob)
    
    def test_top_logprobs(self):
        """Test token info with top logprobs."""
        top_logprobs = {"hello": -0.1, "hi": -0.5, "hey": -1.0}
        
        token = TokenInfo(
            token_id=42,
            token_text="hello",
            top_logprobs=top_logprobs
        )
        
        self.assertEqual(token.top_logprobs, top_logprobs)


class TestCompletionResult(unittest.TestCase):
    """Test CompletionResult data structure."""
    
    def test_valid_completion_result(self):
        """Test creating valid completion result."""
        result = CompletionResult(
            text="Hello world!",
            token_ids=[1, 2, 3, 4]
        )
        
        self.assertEqual(result.text, "Hello world!")
        self.assertEqual(result.token_ids, [1, 2, 3, 4])
        self.assertEqual(result.num_tokens, 4)
    
    def test_empty_token_ids(self):
        """Test completion result with empty token IDs."""
        with self.assertRaises(ValueError):
            CompletionResult(
                text="Hello",
                token_ids=[]
            )
    
    def test_token_info_length_mismatch(self):
        """Test token info length mismatch."""
        token_info = [
            TokenInfo(token_id=1, token_text="hello"),
            TokenInfo(token_id=2, token_text="world")
        ]
        
        # Mismatched lengths
        with self.assertRaises(ValueError):
            CompletionResult(
                text="Hello world!",
                token_ids=[1, 2, 3],  # 3 tokens
                token_info=token_info  # 2 token infos
            )
    
    def test_logprobs_length_mismatch(self):
        """Test logprobs length mismatch."""
        # Mismatched lengths
        with self.assertRaises(ValueError):
            CompletionResult(
                text="Hello world!",
                token_ids=[1, 2, 3],
                logprobs=[-0.1, -0.2]  # 2 logprobs for 3 tokens
            )
    
    def test_average_logprob(self):
        """Test average logprob calculation."""
        result = CompletionResult(
            text="Hello world!",
            token_ids=[1, 2, 3],
            logprobs=[-0.1, -0.2, -0.3]
        )
        
        expected_avg = (-0.1 + -0.2 + -0.3) / 3
        self.assertAlmostEqual(result.average_logprob, expected_avg)
    
    def test_average_logprob_none(self):
        """Test average logprob when logprobs is None."""
        result = CompletionResult(
            text="Hello world!",
            token_ids=[1, 2, 3]
        )
        
        self.assertIsNone(result.average_logprob)
    
    def test_get_token_at_position(self):
        """Test getting token at specific position."""
        token_info = [
            TokenInfo(token_id=1, token_text="hello"),
            TokenInfo(token_id=2, token_text="world")
        ]
        
        result = CompletionResult(
            text="Hello world!",
            token_ids=[1, 2],
            token_info=token_info
        )
        
        # Valid position
        token = result.get_token_at_position(0)
        self.assertEqual(token.token_text, "hello")
        
        # Invalid position
        token = result.get_token_at_position(5)
        self.assertIsNone(token)
        
        # No token info
        result_no_info = CompletionResult(
            text="Hello world!",
            token_ids=[1, 2]
        )
        token = result_no_info.get_token_at_position(0)
        self.assertIsNone(token)
    
    def test_completion_with_metadata(self):
        """Test completion result with metadata."""
        metadata = {"temperature": 0.8, "model_version": "1.0"}
        
        result = CompletionResult(
            text="Hello world!",
            token_ids=[1, 2, 3],
            model_name="test-model",
            sampling_method="top_p",
            generation_time=0.5,
            metadata=metadata
        )
        
        self.assertEqual(result.model_name, "test-model")
        self.assertEqual(result.sampling_method, "top_p")
        self.assertEqual(result.generation_time, 0.5)
        self.assertEqual(result.metadata, metadata)


class TestBatchCompletionResult(unittest.TestCase):
    """Test BatchCompletionResult data structure."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.completions = [
            CompletionResult(
                text="Hello!",
                token_ids=[1, 2]
            ),
            CompletionResult(
                text="World!",
                token_ids=[3, 4, 5]
            ),
            CompletionResult(
                text="Test!",
                token_ids=[6, 7, 8, 9]
            )
        ]
    
    def test_valid_batch_result(self):
        """Test creating valid batch result."""
        batch_result = BatchCompletionResult(
            completions=self.completions,
            batch_size=3
        )
        
        self.assertEqual(len(batch_result), 3)
        self.assertEqual(batch_result.batch_size, 3)
        self.assertEqual(batch_result.total_tokens, 9)  # 2 + 3 + 4
        self.assertEqual(batch_result.average_tokens_per_completion, 3.0)
    
    def test_batch_size_mismatch(self):
        """Test batch size mismatch."""
        with self.assertRaises(ValueError):
            BatchCompletionResult(
                completions=self.completions,
                batch_size=5  # Mismatch with actual length
            )
    
    def test_batch_iteration(self):
        """Test batch iteration and indexing."""
        batch_result = BatchCompletionResult(
            completions=self.completions,
            batch_size=3
        )
        
        # Test indexing
        self.assertEqual(batch_result[0], self.completions[0])
        self.assertEqual(batch_result[1], self.completions[1])
        
        # Test iteration
        iterated_completions = list(batch_result)
        self.assertEqual(iterated_completions, self.completions)
    
    def test_completion_texts_property(self):
        """Test completion texts property."""
        batch_result = BatchCompletionResult(
            completions=self.completions,
            batch_size=3
        )
        
        expected_texts = ["Hello!", "World!", "Test!"]
        self.assertEqual(batch_result.completion_texts, expected_texts)
    
    def test_all_token_ids_property(self):
        """Test all token IDs property."""
        batch_result = BatchCompletionResult(
            completions=self.completions,
            batch_size=3
        )
        
        expected_token_ids = [[1, 2], [3, 4, 5], [6, 7, 8, 9]]
        self.assertEqual(batch_result.all_token_ids, expected_token_ids)
    
    def test_get_average_logprob(self):
        """Test average logprob calculation across batch."""
        # Add logprobs to completions
        completions_with_logprobs = [
            CompletionResult(
                text="Hello!",
                token_ids=[1, 2],
                logprobs=[-0.1, -0.2]  # avg = -0.15
            ),
            CompletionResult(
                text="World!",
                token_ids=[3, 4, 5],
                logprobs=[-0.3, -0.4, -0.5]  # avg = -0.4
            )
        ]
        
        batch_result = BatchCompletionResult(
            completions=completions_with_logprobs,
            batch_size=2
        )
        
        # Average of -0.15 and -0.4 = -0.275
        expected_avg = (-0.15 + -0.4) / 2
        self.assertAlmostEqual(batch_result.get_average_logprob(), expected_avg)
    
    def test_get_average_logprob_none(self):
        """Test average logprob when no logprobs available."""
        batch_result = BatchCompletionResult(
            completions=self.completions,
            batch_size=3
        )
        
        self.assertIsNone(batch_result.get_average_logprob())
    
    def test_filter_by_length(self):
        """Test filtering completions by token length."""
        batch_result = BatchCompletionResult(
            completions=self.completions,
            batch_size=3
        )
        
        # Filter to keep only completions with 3+ tokens
        filtered = batch_result.filter_by_length(min_tokens=3)
        
        self.assertEqual(len(filtered), 2)  # Only 2nd and 3rd completions
        self.assertEqual(filtered[0].text, "World!")
        self.assertEqual(filtered[1].text, "Test!")
    
    def test_filter_by_max_length(self):
        """Test filtering completions by maximum token length."""
        batch_result = BatchCompletionResult(
            completions=self.completions,
            batch_size=3
        )
        
        # Filter to keep only completions with ≤3 tokens
        filtered = batch_result.filter_by_length(max_tokens=3)
        
        self.assertEqual(len(filtered), 2)  # Only 1st and 2nd completions
        self.assertEqual(filtered[0].text, "Hello!")
        self.assertEqual(filtered[1].text, "World!")
    
    def test_filter_by_length_range(self):
        """Test filtering completions by token length range."""
        batch_result = BatchCompletionResult(
            completions=self.completions,
            batch_size=3
        )
        
        # Filter to keep only completions with 2-3 tokens
        filtered = batch_result.filter_by_length(min_tokens=2, max_tokens=3)
        
        self.assertEqual(len(filtered), 2)  # Only 1st and 2nd completions
        self.assertEqual(filtered[0].text, "Hello!")
        self.assertEqual(filtered[1].text, "World!")
    
    def test_batch_metadata(self):
        """Test batch result with metadata."""
        metadata = {"experiment_id": "test_001"}
        
        batch_result = BatchCompletionResult(
            completions=self.completions,
            batch_size=3,
            total_generation_time=1.5,
            model_name="test-model",
            sampling_method="greedy",
            metadata=metadata
        )
        
        self.assertEqual(batch_result.total_generation_time, 1.5)
        self.assertEqual(batch_result.model_name, "test-model")
        self.assertEqual(batch_result.sampling_method, "greedy")
        self.assertEqual(batch_result.metadata, metadata)


if __name__ == "__main__":
    unittest.main()