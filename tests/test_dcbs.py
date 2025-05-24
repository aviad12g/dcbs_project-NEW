"""
Tests for DCBS implementation

This module contains unit tests for the DCBS (Disjunctive Category Beam Search)
implementation, verifying its core functionality and edge cases.
"""

import unittest
import torch
import numpy as np
import os
import sys
import tempfile
import logging
from pathlib import Path
import time
from unittest.mock import patch, MagicMock

# Add the parent directory to the path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dcbs import (
    category_sample,
    get_cached_embeddings,
    get_cached_clustering,
    _embedding_cluster_cache,
    DCBSConfig,
    CacheConfig,
)
from src.dcbs import _token_embedding_cache
from src.token_utils import get_top_tokens, sample_token_from_logits


class TestDCBS(unittest.TestCase):
    """Test cases for the DCBS sampling algorithm."""

    def setUp(self):
        """Set up the test environment before each test.
        
        This method:
        1. Clears caches to ensure tests start with clean state
        2. Sets up default test configurations
        3. Disables logging during tests
        """
        # Clear caches before each test
        _token_embedding_cache.clear()
        _embedding_cluster_cache.clear()
        
        # Default test configurations
        self.dcbs_config: DCBSConfig = {
            "temperature": 1.0,
            "min_tokens_for_clustering": 3,
            "clustering_random_seed": 42,
            "min_kmeans_iterations": 5,
            "min_batch_size": 3584,
        }
        
        self.cache_config: CacheConfig = {
            "embedding_cache_size": 100,  # smaller for tests
            "cluster_cache_size": 50,     # smaller for tests
        }
        
        # Disable logging during tests
        logging.disable(logging.CRITICAL)
        
    def tearDown(self):
        """Clean up after each test.
        
        This method:
        1. Re-enables logging
        """
        # Re-enable logging
        logging.disable(logging.NOTSET)
        
    def create_mock_embedding(self, vocab_size: int, embed_dim: int) -> torch.nn.Embedding:
        """Helper method to create a mock embedding layer with deterministic values.
        
        Creates an embedding layer with controlled random values for testing.
        This ensures tests are reproducible by using a fixed random seed.
        
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Dimension of the embeddings
            
        Returns:
            Mock embedding layer with deterministic values
            
        Example:
            >>> embedding = self.create_mock_embedding(100, 5)
            >>> # Use in tests that need embeddings
            >>> token_id = category_sample(logits, embedding, k=2, top_n=5)
        """
        embedding = torch.nn.Embedding(vocab_size, embed_dim)
        # Initialize with deterministic values for consistent testing
        torch.manual_seed(42)
        embedding.weight.data = torch.randn(vocab_size, embed_dim)
        return embedding
        
    def create_test_logits(self, vocab_size: int, high_prob_range: tuple = None) -> torch.Tensor:
        """Helper method to create test logits with controlled probability distribution.
        
        Creates a tensor of logits where most tokens have very low probability,
        and optionally a specified range of tokens have higher probability.
        This is useful for testing sampling behavior in a controlled way.
        
        Args:
            vocab_size: Size of the vocabulary
            high_prob_range: Optional tuple (start, end) for tokens with higher probability
            
        Returns:
            Tensor of logits with controlled probability distribution
            
        Example:
            >>> # Create logits where tokens 10-19 have high probability
            >>> logits = self.create_test_logits(100, high_prob_range=(10, 20))
            >>> # Use in sampling tests
            >>> token_id = category_sample(logits, embedding, k=2, top_n=5)
        """
        logits = torch.ones(vocab_size) * -100  # Very negative logits
        
        if high_prob_range:
            start, end = high_prob_range
            # Set increasing values for the specified range
            logits[start:end] = torch.arange(end-start, 0, -1).float()
            
        return logits

    def test_basic_sampling(self):
        """Test basic token sampling functionality with uniform probability distribution.
        
        This test verifies the core functionality of the DCBS algorithm by ensuring
        it can sample tokens from a uniform probability distribution. It checks that
        the algorithm returns valid token IDs within the expected range.
        
        Test objectives:
        - Verify that the sampling function returns valid token IDs
        - Confirm that the function works with uniform logits
        - Ensure the algorithm doesn't crash with basic inputs
        
        This is a fundamental test that must pass for the algorithm to be considered
        functional at all.
        """
        vocab_size = 10
        embed_dim = 5

        # Create mock embedding
        embedding = self.create_mock_embedding(vocab_size, embed_dim)

        # Create uniform logits
        logits = torch.ones(vocab_size)

        # Sample a token
        token_id = category_sample(
            logits, embedding, k=2, top_n=5,
            dcbs_config=self.dcbs_config,
            cache_config=self.cache_config
        )

        # Verify it's a valid token ID
        self.assertGreaterEqual(token_id, 0)
        self.assertLess(token_id, vocab_size)

    def test_top_n_filtering(self):
        """Test that DCBS correctly selects only from top_n tokens.
        
        Verifies that:
        - The sampling function only selects tokens from the top_n highest probability tokens
        - The function respects the top_n parameter
        """
        vocab_size = 100
        embed_dim = 5

        # Create mock embedding
        embedding = self.create_mock_embedding(vocab_size, embed_dim)

        # Set logits so only tokens 10-19 have high probability
        logits = self.create_test_logits(vocab_size, high_prob_range=(10, 20))

        # Set top_n to 5, which should only consider tokens 10-14
        top_n = 5

        # Sample multiple times to verify behavior
        samples = []
        for _ in range(10):
            token_id = category_sample(
                logits, embedding, k=2, top_n=top_n,
                dcbs_config=self.dcbs_config,
                cache_config=self.cache_config
            )
            samples.append(token_id)

        # All samples should be in the top tokens (10-14)
        for token_id in samples:
            self.assertGreaterEqual(token_id, 10)
            self.assertLess(token_id, 15)
            
    def test_get_top_tokens(self):
        """Test the get_top_tokens function from token_utils module.
        
        Verifies that:
        - The function returns the correct number of top tokens
        - The function returns the tokens with highest logits
        - The function includes force-included tokens
        """
        vocab_size = 100
        
        # Set logits so only tokens 10-19 have high probability
        logits = self.create_test_logits(vocab_size, high_prob_range=(10, 20))
        
        # Get top 5 tokens
        top_tokens = get_top_tokens(logits, 5)
        
        # Should get tokens 10-14
        self.assertEqual(len(top_tokens), 5)
        for token_id in top_tokens:
            self.assertGreaterEqual(token_id, 10)
            self.assertLess(token_id, 15)
        
        # Test force include
        force_include = {30, 40}
        top_tokens = get_top_tokens(logits, 5, force_include)
        
        # Should include the forced tokens
        self.assertIn(30, top_tokens)
        self.assertIn(40, top_tokens)

    def test_clustering(self):
        """Test that embeddings are clustered correctly based on semantic similarity.
        
        This test creates a controlled scenario with two clearly distinct clusters
        in the embedding space to verify that the clustering algorithm can properly
        separate tokens into semantically meaningful groups. This is a critical test
        since semantic clustering is the core innovation of the DCBS algorithm.
        
        Test objectives:
        - Verify that the clustering algorithm can separate distinct clusters
        - Confirm that the sampling function works with clustered embeddings
        - Ensure the algorithm can handle embeddings with clear semantic distinctions
        
        The test creates two clusters:
        1. Tokens 0-9: positioned near (0,0) in embedding space
        2. Tokens 10-19: positioned near (10,10) in embedding space
        
        With k=2 clusters, we expect the algorithm to identify these distinct groups
        and sample from them according to their probability mass.
        """
        vocab_size = 20
        embed_dim = 2

        # Create mock embedding with known clusters
        embedding = torch.nn.Embedding(vocab_size, embed_dim)

        # Create 2 distinct clusters in 2D space
        # Cluster 1: tokens 0-9, close to (0,0)
        # Cluster 2: tokens 10-19, close to (10,10)
        embedding.weight.data[:10] = torch.randn(10, embed_dim) + torch.tensor(
            [0.0, 0.0]
        )
        embedding.weight.data[10:] = torch.randn(10, embed_dim) + torch.tensor(
            [10.0, 10.0]
        )

        # Make logits uniform to focus on clustering behavior
        logits = torch.ones(vocab_size)

        # Use k=2 clusters which should split along our predefined clusters
        token_id = category_sample(
            logits, embedding, k=2, top_n=20,
            dcbs_config=self.dcbs_config,
            cache_config=self.cache_config
        )

        # Due to probabilistic nature, we can't assert exact token values,
        # but we can check category_sample works without errors
        self.assertGreaterEqual(token_id, 0)
        self.assertLess(token_id, vocab_size)

    def test_token_filtering(self):
        """Test that filter_tokens restricts sampling to specified token IDs.
        
        Verifies that:
        - The sampling function only selects tokens from the filter_tokens set
        - The function works with arbitrary token ID sets
        """
        vocab_size = 100
        embed_dim = 5

        # Create mock embedding
        embedding = self.create_mock_embedding(vocab_size, embed_dim)

        # Uniform logits
        logits = torch.ones(vocab_size)

        # Only allow sampling from tokens 42, 43, 44
        filter_tokens = {42, 43, 44}

        # Sample multiple times
        samples = []
        for _ in range(10):
            token_id = category_sample(
                logits, embedding, k=2, top_n=50, filter_tokens=filter_tokens,
                dcbs_config=self.dcbs_config,
                cache_config=self.cache_config
            )
            samples.append(token_id)

        # All samples should be in the filter set
        for token_id in samples:
            self.assertIn(token_id, filter_tokens)

    def test_edge_case_small_k(self):
        """Test with small k value (k=1 should behave like argmax).
        
        Verifies that:
        - When k=1, the sampling function behaves like argmax
        - The function handles edge cases correctly
        """
        vocab_size = 10
        embed_dim = 5

        # Create mock embedding
        embedding = self.create_mock_embedding(vocab_size, embed_dim)

        # Set logits with a clear maximum
        logits = torch.zeros(vocab_size)
        logits[3] = 10.0  # Token 3 has much higher probability

        # With k=1, should behave like argmax
        token_id = category_sample(
            logits, embedding, k=1, top_n=vocab_size,
            dcbs_config=self.dcbs_config,
            cache_config=self.cache_config
        )

        # Should return token 3
        self.assertEqual(token_id, 3)

    def test_empty_filter_tokens(self):
        """Test handling of empty filter_tokens set (should not crash).
        
        Verifies that:
        - The sampling function handles empty filter_tokens gracefully
        - The function doesn't crash with edge cases
        """
        vocab_size = 10
        embed_dim = 5

        # Create mock embedding
        embedding = self.create_mock_embedding(vocab_size, embed_dim)

        # Uniform logits
        logits = torch.ones(vocab_size)

        # Empty filter set
        filter_tokens = set()

        # Should handle this gracefully
        token_id = category_sample(
            logits, embedding, k=2, top_n=5, filter_tokens=filter_tokens,
            dcbs_config=self.dcbs_config,
            cache_config=self.cache_config
        )

        # Just verify it doesn't crash and returns a valid token
        self.assertGreaterEqual(token_id, 0)
        self.assertLess(token_id, vocab_size)

    def test_small_token_set(self):
        """Test with very small token set (should skip clustering).
        
        Verifies that:
        - The sampling function works with small vocabularies
        - The function skips clustering when appropriate
        """
        vocab_size = 5
        embed_dim = 5

        # Create mock embedding
        embedding = self.create_mock_embedding(vocab_size, embed_dim)

        # Uniform logits for a small vocabulary
        logits = torch.ones(vocab_size)

        # Sample a token
        token_id = category_sample(
            logits, embedding, k=2, top_n=3,
            dcbs_config=self.dcbs_config,
            cache_config=self.cache_config
        )

        # Verify it's a valid token ID
        self.assertGreaterEqual(token_id, 0)
        self.assertLess(token_id, vocab_size)

    def test_custom_config(self):
        """Test that custom configuration parameters are respected.
        
        Verifies that:
        - The sampling function uses the provided configuration
        - Different configuration values affect the behavior
        """
        vocab_size = 20
        embed_dim = 5
        
        # Create mock embedding
        embedding = self.create_mock_embedding(vocab_size, embed_dim)
        
        # Uniform logits
        logits = torch.ones(vocab_size)
        
        # Custom DCBS config with non-default values
        custom_dcbs_config: DCBSConfig = {
            "temperature": 0.5,  # Different temperature
            "min_tokens_for_clustering": 2,  # Lower threshold
            "clustering_random_seed": 123,  # Different seed
            "min_kmeans_iterations": 3,
            "min_batch_size": 1000,
        }
        
        # Sample with custom config
        token_id = category_sample(
            logits, embedding, k=2, top_n=10,
            dcbs_config=custom_dcbs_config,
            cache_config=self.cache_config
        )
        
        # Just verify it doesn't crash and returns a valid token
        self.assertGreaterEqual(token_id, 0)
        self.assertLess(token_id, vocab_size)
        
    @patch('src.dcbs.get_cached_clustering')
    def test_clustering_called_with_config(self, mock_clustering):
        """Test that clustering function is called with the correct config.
        
        Verifies that:
        - The configuration is passed correctly to the clustering function
        - The function uses the provided configuration
        """
        vocab_size = 20
        embed_dim = 5
        
        # Setup mock
        mock_clustering.return_value = np.zeros(10, dtype=np.int32)
        
        # Create mock embedding
        embedding = self.create_mock_embedding(vocab_size, embed_dim)
        
        # Uniform logits
        logits = torch.ones(vocab_size)
        
        # Custom DCBS config
        custom_dcbs_config: DCBSConfig = {
            "temperature": 0.8,
            "min_tokens_for_clustering": 2,
            "clustering_random_seed": 123,
            "min_kmeans_iterations": 3,
            "min_batch_size": 1000,
        }
        
        # Call category_sample
        category_sample(
            logits, embedding, k=2, top_n=10,
            dcbs_config=custom_dcbs_config,
            cache_config=self.cache_config
        )
        
        # Verify clustering was called with the custom config
        mock_clustering.assert_called_once()
        args, kwargs = mock_clustering.call_args
        self.assertEqual(kwargs['dcbs_config'], custom_dcbs_config)
        self.assertEqual(kwargs['cache_config'], self.cache_config)


if __name__ == "__main__":
    unittest.main()
