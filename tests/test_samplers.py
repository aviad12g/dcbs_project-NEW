"""
Comprehensive test suite for sampler classes.

Tests all sampling methods including edge cases, statistical properties,
and interface consistency.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from dcbs import (
    DCBSSampler,
    GreedySampler,
    RandomSampler,
    Sampler,
    SamplingContext,
    TopPSampler,
)


class TestSamplingContext:
    """Test the SamplingContext dataclass."""

    def test_context_creation(self):
        """Test that SamplingContext can be created with various parameters."""
        # Test empty context
        context = SamplingContext()
        assert context.embedding_layer is None
        assert context.tokenizer is None
        assert context.device is None

        # Test with mock objects
        mock_embedding = Mock()
        mock_tokenizer = Mock()
        mock_device = torch.device("cpu")

        context = SamplingContext(
            embedding_layer=mock_embedding, tokenizer=mock_tokenizer, device=mock_device
        )
        assert context.embedding_layer == mock_embedding
        assert context.tokenizer == mock_tokenizer
        assert context.device == mock_device


class TestGreedySampler:
    """Test the GreedySampler class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.sampler = GreedySampler()
        self.logits = torch.tensor([1.0, 3.0, 2.0, 0.5, 2.5])

    def test_greedy_sampling_no_filter(self):
        """Test greedy sampling without token filtering."""
        token_id = self.sampler.sample(self.logits)
        assert token_id == 1  # Index of maximum value (3.0)

    def test_greedy_sampling_with_filter(self):
        """Test greedy sampling with token filtering."""
        filter_tokens = {0, 2, 4}  # Available tokens
        token_id = self.sampler.sample(self.logits, filter_tokens=filter_tokens)
        assert token_id == 4  # Highest value among filtered tokens (2.5)

    def test_greedy_sampling_single_filter(self):
        """Test greedy sampling with single allowed token."""
        filter_tokens = {3}
        token_id = self.sampler.sample(self.logits, filter_tokens=filter_tokens)
        assert token_id == 3

    def test_greedy_sampling_empty_filter(self):
        """Test greedy sampling with empty filter (should use global argmax)."""
        filter_tokens = set()
        token_id = self.sampler.sample(self.logits, filter_tokens=filter_tokens)
        assert token_id == 1  # Global argmax

    def test_greedy_sampling_negative_logits(self):
        """Test greedy sampling with negative logits."""
        negative_logits = torch.tensor([-2.0, -1.0, -3.0, -0.5])
        token_id = self.sampler.sample(negative_logits)
        assert token_id == 3  # Index of maximum value (-0.5)

    def test_greedy_sampling_edge_cases(self):
        """Test edge cases for greedy sampling."""
        # Test with inf values
        inf_logits = torch.tensor([float("inf"), 1.0, 2.0])
        token_id = self.sampler.sample(inf_logits)
        assert token_id == 0

        # Test with -inf values (should still work)
        ninf_logits = torch.tensor([float("-inf"), 1.0, 2.0])
        token_id = self.sampler.sample(ninf_logits)
        assert token_id == 2


class TestTopPSampler:
    """Test the TopPSampler class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.sampler = TopPSampler(p=0.9)
        # Create logits where we can predict top-p behavior
        self.logits = torch.tensor(
            [3.0, 2.0, 1.0, 0.0, -1.0]
        )  # Sorted for easy calculation

    def test_top_p_initialization(self):
        """Test TopPSampler initialization."""
        assert self.sampler.p == 0.9

        # Test with different p values
        sampler_low = TopPSampler(p=0.1)
        assert sampler_low.p == 0.1

    def test_top_p_sampling_deterministic_check(self):
        """Test that top-p sampling is working (probabilistic, so we check multiple runs)."""
        # Use a fixed seed for reproducibility in testing
        torch.manual_seed(42)

        # Sample multiple times and check that we get reasonable results
        results = []
        for _ in range(100):
            token_id = self.sampler.sample(self.logits)
            results.append(token_id)

        # Should sample from top tokens more frequently
        unique_results = set(results)
        assert len(unique_results) > 1  # Should have some diversity
        assert all(r in range(len(self.logits)) for r in results)  # All valid indices

    def test_top_p_with_filter(self):
        """Test top-p sampling with token filtering."""
        torch.manual_seed(42)
        filter_tokens = {1, 3}  # Only allow specific tokens

        results = []
        for _ in range(50):
            token_id = self.sampler.sample(self.logits, filter_tokens=filter_tokens)
            results.append(token_id)

        # All results should be from filter set
        assert all(r in filter_tokens for r in results)

    def test_top_p_edge_cases(self):
        """Test edge cases for top-p sampling."""
        # Test with very high p (should include almost all tokens)
        high_p_sampler = TopPSampler(p=0.99)
        token_id = high_p_sampler.sample(self.logits)
        assert token_id in range(len(self.logits))

        # Test with very low p (should be more selective)
        low_p_sampler = TopPSampler(p=0.1)
        token_id = low_p_sampler.sample(self.logits)
        assert token_id in range(len(self.logits))


class TestRandomSampler:
    """Test the RandomSampler class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.sampler = RandomSampler()
        self.logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_random_sampling_no_filter(self):
        """Test random sampling without filtering."""
        # Test multiple samples to check randomness
        results = []
        for _ in range(100):
            token_id = self.sampler.sample(self.logits)
            results.append(token_id)

        # Should have some diversity and all valid indices
        unique_results = set(results)
        assert len(unique_results) > 1
        assert all(r in range(len(self.logits)) for r in results)

    def test_random_sampling_with_filter(self):
        """Test random sampling with token filtering."""
        filter_tokens = {1, 3, 4}

        results = []
        for _ in range(50):
            token_id = self.sampler.sample(self.logits, filter_tokens=filter_tokens)
            results.append(token_id)

        # All results should be from filter set
        assert all(r in filter_tokens for r in results)

    def test_random_sampling_single_filter(self):
        """Test random sampling with single allowed token."""
        filter_tokens = {2}
        token_id = self.sampler.sample(self.logits, filter_tokens=filter_tokens)
        assert token_id == 2

    def test_random_sampling_empty_filter(self):
        """Test random sampling with empty filter."""
        filter_tokens = set()
        token_id = self.sampler.sample(self.logits, filter_tokens=filter_tokens)
        assert token_id in range(len(self.logits))


class TestDCBSSampler:
    """Test the DCBSSampler class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.sampler = DCBSSampler(k=3, top_n=10)
        self.logits = torch.tensor([1.0, 3.0, 2.0, 0.5, 2.5, 1.5, 0.8, 2.2, 1.8, 0.3])

        # Create mock embedding layer
        self.mock_embedding = Mock()
        embedding_dim = 8
        vocab_size = len(self.logits)

        # Create realistic embedding weights
        torch.manual_seed(42)
        self.mock_embedding.weight = torch.randn(vocab_size, embedding_dim)
        self.mock_embedding.embedding_dim = embedding_dim

        # Mock the forward call
        def mock_forward(token_ids):
            if isinstance(token_ids, torch.Tensor):
                indices = token_ids.cpu().numpy()
            else:
                indices = np.array(token_ids)
            return self.mock_embedding.weight[indices]

        self.mock_embedding.side_effect = mock_forward

        # Create context
        self.context = SamplingContext(
            embedding_layer=self.mock_embedding, device=torch.device("cpu")
        )

    def test_dcbs_initialization(self):
        """Test DCBSSampler initialization."""
        assert self.sampler.k == 3
        assert self.sampler.top_n == 10

    def test_dcbs_requires_context(self):
        """Test that DCBS requires context with embedding layer."""
        with pytest.raises(ValueError, match="DCBS requires a SamplingContext"):
            self.sampler.sample(self.logits)

        with pytest.raises(ValueError, match="DCBS requires a SamplingContext"):
            context_no_embedding = SamplingContext()
            self.sampler.sample(self.logits, context=context_no_embedding)

    def test_dcbs_sampling_basic(self):
        """Test basic DCBS sampling functionality."""
        token_id = self.sampler.sample(self.logits, context=self.context)
        assert isinstance(token_id, int)
        assert 0 <= token_id < len(self.logits)

    def test_dcbs_with_filter(self):
        """Test DCBS sampling with token filtering."""
        filter_tokens = {1, 4, 7}
        token_id = self.sampler.sample(
            self.logits, filter_tokens=filter_tokens, context=self.context
        )
        assert token_id in filter_tokens

    def test_dcbs_single_token_filter(self):
        """Test DCBS with single allowed token."""
        filter_tokens = {5}
        token_id = self.sampler.sample(
            self.logits, filter_tokens=filter_tokens, context=self.context
        )
        assert token_id == 5

    def test_dcbs_insufficient_tokens_for_clustering(self):
        """Test DCBS behavior when too few tokens for clustering."""
        # Create a scenario with very few candidate tokens
        small_logits = torch.tensor([1.0, 2.0])
        small_context = SamplingContext(
            embedding_layer=self.mock_embedding, device=torch.device("cpu")
        )

        token_id = self.sampler.sample(small_logits, context=small_context)
        assert token_id in [0, 1]

    def test_dcbs_edge_cases(self):
        """Test DCBS edge cases."""
        # Test with inf logits
        inf_logits = torch.tensor([float("inf")] * 5 + [1.0, 2.0, 3.0])
        token_id = self.sampler.sample(inf_logits, context=self.context)
        assert isinstance(token_id, int)

        # Test with nan logits
        nan_logits = torch.tensor([float("nan")] * 3 + [1.0, 2.0, 3.0])
        token_id = self.sampler.sample(nan_logits, context=self.context)
        assert isinstance(token_id, int)

    def test_dcbs_deterministic_behavior(self):
        """Test that DCBS is deterministic with same input."""
        # DCBS should be deterministic, so same inputs should give same outputs
        results = []
        for _ in range(5):
            token_id = self.sampler.sample(self.logits, context=self.context)
            results.append(token_id)

        # All results should be the same (deterministic)
        assert len(set(results)) == 1

    def test_dcbs_caching(self):
        """Test that DCBS caching works correctly."""
        # First call should cache embeddings
        token_id1 = self.sampler.sample(self.logits, context=self.context)

        # Second call should use cached embeddings
        token_id2 = self.sampler.sample(self.logits, context=self.context)

        # Results should be the same (deterministic + caching)
        assert token_id1 == token_id2


class TestSamplerInterfaceConsistency:
    """Test that all samplers implement the interface consistently."""

    def setup_method(self):
        """Setup test fixtures."""
        self.logits = torch.tensor([1.0, 3.0, 2.0, 0.5, 2.5])
        self.filter_tokens = {1, 2, 4}

        # Setup context for DCBS
        mock_embedding = Mock()
        embedding_dim = 8
        vocab_size = len(self.logits)
        mock_embedding.weight = torch.randn(vocab_size, embedding_dim)
        mock_embedding.embedding_dim = embedding_dim

        def mock_forward(token_ids):
            if isinstance(token_ids, torch.Tensor):
                indices = token_ids.cpu().numpy()
            else:
                indices = np.array(token_ids)
            return mock_embedding.weight[indices]

        mock_embedding.side_effect = mock_forward

        self.context = SamplingContext(
            embedding_layer=mock_embedding, device=torch.device("cpu")
        )

        # Create all samplers
        self.samplers = {
            "greedy": GreedySampler(),
            "top_p": TopPSampler(p=0.9),
            "random": RandomSampler(),
            "dcbs": DCBSSampler(k=3, top_n=5),
        }

    def test_interface_consistency(self):
        """Test that all samplers follow the same interface."""
        for name, sampler in self.samplers.items():
            # Test basic sampling
            if name == "dcbs":
                token_id = sampler.sample(self.logits, context=self.context)
            else:
                token_id = sampler.sample(self.logits)

            assert isinstance(token_id, int)
            assert 0 <= token_id < len(self.logits)

            # Test with filter tokens
            if name == "dcbs":
                token_id = sampler.sample(
                    self.logits, filter_tokens=self.filter_tokens, context=self.context
                )
            else:
                token_id = sampler.sample(self.logits, filter_tokens=self.filter_tokens)

            assert token_id in self.filter_tokens

    def test_inheritance(self):
        """Test that all samplers inherit from Sampler."""
        for sampler in self.samplers.values():
            assert isinstance(sampler, Sampler)

    def test_method_signatures(self):
        """Test that all samplers have consistent method signatures."""
        for sampler in self.samplers.values():
            # Check that sample method exists
            assert hasattr(sampler, "sample")
            assert callable(getattr(sampler, "sample"))


class TestStatisticalProperties:
    """Test statistical properties of samplers."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create biased logits for testing
        self.biased_logits = torch.tensor(
            [5.0, 1.0, 2.0, 0.5]
        )  # Heavily biased toward index 0

    def test_greedy_determinism(self):
        """Test that greedy sampling is fully deterministic."""
        sampler = GreedySampler()
        results = [sampler.sample(self.biased_logits) for _ in range(100)]

        # All results should be the same
        assert len(set(results)) == 1
        assert results[0] == 0  # Should always pick highest probability

    def test_random_uniformity(self):
        """Test that random sampling has reasonable uniformity."""
        # Create uniform logits
        uniform_logits = torch.zeros(4)  # All equal probability
        sampler = RandomSampler()
        filter_tokens = {0, 1, 2, 3}

        results = [
            sampler.sample(uniform_logits, filter_tokens=filter_tokens)
            for _ in range(1000)
        ]

        # Check that we get reasonable distribution
        unique_results = set(results)
        assert len(unique_results) >= 3  # Should sample from multiple tokens

        # Check that each token appears with reasonable frequency (rough test)
        counts = {i: results.count(i) for i in range(4)}
        min_count = min(counts.values())
        max_count = max(counts.values())

        # Shouldn't be too skewed (allowing for random variation)
        assert max_count / min_count < 3.0  # Rough uniformity check

    def test_top_p_diversity(self):
        """Test that top-p sampling provides appropriate diversity."""
        torch.manual_seed(42)  # For reproducibility
        sampler = TopPSampler(p=0.9)

        results = [sampler.sample(self.biased_logits) for _ in range(200)]
        unique_results = set(results)

        # Should have some diversity but bias toward high-probability tokens
        assert len(unique_results) > 1
        assert 0 in results  # Should include the highest probability token

        # Most samples should be the highest probability token, but not all
        proportion_max = results.count(0) / len(results)
        assert 0.5 < proportion_max < 0.95  # Biased but not completely deterministic


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
