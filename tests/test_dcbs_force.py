#!/usr/bin/env python3
"""
Unit tests for the filter_tokens functionality in the canonical DCBS implementation.
Tests to ensure that DCBS properly restricts sampling to specified tokens even when they have
extremely low probabilities.
"""

import os
import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

# Set fixed seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Add the parent directory to the path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from dcbs import DCBSSampler, SamplingContext


class TestDCBSTokenFiltering(unittest.TestCase):
    """Test case for DCBS token filtering functionality."""

    def setUp(self):
        """Set up the test environment."""
        self.sampler = DCBSSampler.create_default(k=3, top_n=10)

    def test_filter_tokens_basic(self):
        """Test that filter_tokens properly restricts sampling to specified tokens."""
        vocab_size = 10
        embed_dim = 5

        # Create mock embedding
        embedding = torch.nn.Embedding(vocab_size, embed_dim)
        torch.manual_seed(42)
        embedding.weight.data = torch.randn(vocab_size, embed_dim)

        # Create sampling context
        context = SamplingContext(
            embedding_layer=embedding, tokenizer=None, device=torch.device("cpu")
        )

        # Create logits with very low probability for filtered tokens
        logits = torch.ones(vocab_size)
        logits[3] = -10.0  # Very low probability for token 3
        logits[7] = -10.0  # Very low probability for token 7

        # Only allow sampling from tokens 3 and 7 (low probability tokens)
        filter_tokens = {3, 7}

        # Sample multiple times
        samples = []
        for _ in range(20):
            token_id = self.sampler.sample(logits, context, filter_tokens=filter_tokens)
            samples.append(token_id)

        # All samples should be in the filter set
        for token_id in samples:
            self.assertIn(
                token_id,
                filter_tokens,
                f"Token {token_id} not in filter_tokens {filter_tokens}",
            )

    def test_single_filter_token(self):
        """Test that DCBS handles single token filtering correctly."""
        vocab_size = 10
        embed_dim = 5

        # Create mock embedding
        embedding = torch.nn.Embedding(vocab_size, embed_dim)
        torch.manual_seed(42)
        embedding.weight.data = torch.randn(vocab_size, embed_dim)

        # Create sampling context
        context = SamplingContext(
            embedding_layer=embedding, tokenizer=None, device=torch.device("cpu")
        )

        # Create logits with very low probability for the answer token
        logits = torch.ones(vocab_size)
        logits[5] = -10.0  # Very low probability for token 5

        # Only allow sampling from token 5
        filter_tokens = {5}

        # Run multiple trials - should always return token 5
        for _ in range(10):
            token_id = self.sampler.sample(logits, context, filter_tokens=filter_tokens)
            self.assertEqual(token_id, 5, f"Expected token 5, got {token_id}")

    def test_filter_tokens_deterministic(self):
        """Test that DCBS filtering is deterministic when using greedy selection."""
        vocab_size = 10
        embed_dim = 5

        # Create mock embedding
        embedding = torch.nn.Embedding(vocab_size, embed_dim)
        torch.manual_seed(42)
        embedding.weight.data = torch.randn(vocab_size, embed_dim)

        # Create sampling context
        context = SamplingContext(
            embedding_layer=embedding, tokenizer=None, device=torch.device("cpu")
        )

        # Create logits where token 2 has higher probability than token 8
        logits = torch.ones(vocab_size) * -10.0
        logits[2] = 5.0  # High probability
        logits[8] = 1.0  # Lower probability

        # Filter to tokens 2 and 8
        filter_tokens = {2, 8}

        # Since DCBS is deterministic and greedy, should always pick token 2
        results = []
        for _ in range(10):
            token_id = self.sampler.sample(logits, context, filter_tokens=filter_tokens)
            results.append(token_id)

        # All results should be the same (deterministic)
        self.assertEqual(
            len(set(results)), 1, f"Expected deterministic results, got {set(results)}"
        )

        # Should pick the higher probability token (2)
        self.assertEqual(
            results[0], 2, f"Expected token 2 (higher prob), got {results[0]}"
        )


if __name__ == "__main__":
    unittest.main()
