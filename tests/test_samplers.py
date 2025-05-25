"""
Comprehensive test suite for sampler classes.
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
        context = SamplingContext()
        assert context.embedding_layer is None
        assert context.tokenizer is None
        assert context.device is None

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
        self.sampler = GreedySampler()
        self.logits = torch.tensor([1.0, 3.0, 2.0, 0.5, 2.5])

    def test_greedy_sampling_no_filter(self):
        token_id = self.sampler.sample(self.logits)
        assert token_id == 1  # Index of maximum value (3.0)

    def test_greedy_sampling_with_filter(self):
        filter_tokens = {0, 2, 4}
        token_id = self.sampler.sample(self.logits, filter_tokens=filter_tokens)
        assert token_id == 4  # Highest value among filtered tokens (2.5)

    def test_greedy_sampling_single_filter(self):
        filter_tokens = {3}
        token_id = self.sampler.sample(self.logits, filter_tokens=filter_tokens)
        assert token_id == 3

    def test_greedy_sampling_empty_filter(self):
        filter_tokens = set()
        token_id = self.sampler.sample(self.logits, filter_tokens=filter_tokens)
        assert token_id == 1  # Global argmax

    def test_greedy_sampling_negative_logits(self):
        negative_logits = torch.tensor([-2.0, -1.0, -3.0, -0.5])
        token_id = self.sampler.sample(negative_logits)
        assert token_id == 3  # Index of maximum value (-0.5)

    def test_greedy_sampling_edge_cases(self):
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
        self.sampler = TopPSampler(p=0.9)
        self.logits = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0])

    def test_top_p_initialization(self):
        assert self.sampler.p == 0.9

        sampler_low = TopPSampler(p=0.1)
        assert sampler_low.p == 0.1

    def test_top_p_sampling_deterministic_check(self):
        torch.manual_seed(42)

        results = []
        for _ in range(100):
            token_id = self.sampler.sample(self.logits)
            results.append(token_id)

        unique_results = set(results)
        assert len(unique_results) > 1  # Should have some diversity
        assert all(r in range(len(self.logits)) for r in results)

    def test_top_p_with_filter(self):
        torch.manual_seed(42)
        filter_tokens = {1, 3}

        results = []
        for _ in range(50):
            token_id = self.sampler.sample(self.logits, filter_tokens=filter_tokens)
            results.append(token_id)

        assert all(r in filter_tokens for r in results)

    def test_top_p_edge_cases(self):
        high_p_sampler = TopPSampler(p=0.99)
        token_id = high_p_sampler.sample(self.logits)
        assert token_id in range(len(self.logits))

        low_p_sampler = TopPSampler(p=0.1)
        token_id = low_p_sampler.sample(self.logits)
        assert token_id in range(len(self.logits))


class TestRandomSampler:
    """Test the RandomSampler class."""

    def setup_method(self):
        self.sampler = RandomSampler()
        self.logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_random_sampling_no_filter(self):
        results = []
        for _ in range(100):
            token_id = self.sampler.sample(self.logits)
            results.append(token_id)

        unique_results = set(results)
        assert len(unique_results) > 1
        assert all(r in range(len(self.logits)) for r in results)

    def test_random_sampling_with_filter(self):
        filter_tokens = {1, 3, 4}

        results = []
        for _ in range(50):
            token_id = self.sampler.sample(self.logits, filter_tokens=filter_tokens)
            results.append(token_id)

        assert all(r in filter_tokens for r in results)

    def test_random_sampling_single_filter(self):
        filter_tokens = {2}
        token_id = self.sampler.sample(self.logits, filter_tokens=filter_tokens)
        assert token_id == 2

    def test_random_sampling_empty_filter(self):
        filter_tokens = set()
        token_id = self.sampler.sample(self.logits, filter_tokens=filter_tokens)
        assert token_id in range(len(self.logits))


class TestDCBSSampler:
    """Test the DCBSSampler class."""

    def setup_method(self):
        from dcbs.clustering import KMeansClusterer, TopNCandidateSelector
        
        clusterer = KMeansClusterer(k=3)
        candidate_selector = TopNCandidateSelector(top_n=10)
        self.sampler = DCBSSampler(clusterer, candidate_selector)
        self.logits = torch.tensor([1.0, 3.0, 2.0, 0.5, 2.5, 1.5, 0.8, 2.2, 1.8, 0.3])

        self.mock_embedding = Mock()
        embedding_dim = 8
        vocab_size = len(self.logits)

        torch.manual_seed(42)
        self.mock_embedding.weight = torch.randn(vocab_size, embedding_dim)
        self.mock_embedding.embedding_dim = embedding_dim

        def mock_forward(token_ids):
            if isinstance(token_ids, torch.Tensor):
                indices = token_ids.cpu().numpy()
            else:
                indices = np.array(token_ids)
            return self.mock_embedding.weight[indices]

        self.mock_embedding.side_effect = mock_forward

        self.context = SamplingContext(
            embedding_layer=self.mock_embedding, device=torch.device("cpu")
        )

    def test_dcbs_initialization(self):
        assert self.sampler.k == 3
        assert self.sampler.top_n == 10

    def test_dcbs_requires_context(self):
        with pytest.raises(ValueError, match="DCBS requires a SamplingContext"):
            self.sampler.sample(self.logits)

        with pytest.raises(ValueError, match="DCBS requires a SamplingContext"):
            context_no_embedding = SamplingContext()
            self.sampler.sample(self.logits, context=context_no_embedding)

    def test_dcbs_sampling_basic(self):
        token_id = self.sampler.sample(self.logits, context=self.context)
        assert isinstance(token_id, int)
        assert 0 <= token_id < len(self.logits)

    def test_dcbs_with_filter(self):
        filter_tokens = {1, 4, 7}
        token_id = self.sampler.sample(
            self.logits, filter_tokens=filter_tokens, context=self.context
        )
        assert token_id in filter_tokens

    def test_dcbs_single_token_filter(self):
        filter_tokens = {5}
        token_id = self.sampler.sample(
            self.logits, filter_tokens=filter_tokens, context=self.context
        )
        assert token_id == 5

    def test_dcbs_insufficient_tokens_for_clustering(self):
        small_logits = torch.tensor([1.0, 2.0])
        small_context = SamplingContext(
            embedding_layer=self.mock_embedding, device=torch.device("cpu")
        )

        token_id = self.sampler.sample(small_logits, context=small_context)
        assert token_id in [0, 1]

    def test_dcbs_edge_cases(self):
        inf_logits = torch.tensor([float("inf")] * 5 + [1.0, 2.0, 3.0])
        token_id = self.sampler.sample(inf_logits, context=self.context)
        assert isinstance(token_id, int)

        nan_logits = torch.tensor([float("nan")] * 3 + [1.0, 2.0, 3.0])
        token_id = self.sampler.sample(nan_logits, context=self.context)
        assert isinstance(token_id, int)

    def test_dcbs_deterministic_behavior(self):
        results = []
        for _ in range(5):
            token_id = self.sampler.sample(self.logits, context=self.context)
            results.append(token_id)

        assert len(set(results)) == 1

    def test_dcbs_caching(self):
        token_id1 = self.sampler.sample(self.logits, context=self.context)
        token_id2 = self.sampler.sample(self.logits, context=self.context)

        assert token_id1 == token_id2

    def test_dcbs_caching_configuration(self):
        sampler_cached = DCBSSampler.create_default(k=3, top_n=5, enable_caching=True)
        assert sampler_cached.enable_caching is True
        assert sampler_cached.cache_manager is not None

        sampler_no_cache = DCBSSampler.create_default(k=3, top_n=5, enable_caching=False)
        assert sampler_no_cache.enable_caching is False
        assert sampler_no_cache.cache_manager is None

        sampler_no_cache_2 = DCBSSampler.create_no_cache(k=3, top_n=5)
        assert sampler_no_cache_2.enable_caching is False
        assert sampler_no_cache_2.cache_manager is None

    def test_dcbs_cache_disabled_functionality(self):
        sampler = DCBSSampler.create_no_cache(k=3, top_n=5)
        
        token_id = sampler.sample(self.logits, context=self.context)
        assert isinstance(token_id, int)
        assert 0 <= token_id < len(self.logits)

        stats = sampler.get_cache_stats()
        assert stats["caching_enabled"] is False
        assert "message" in stats

        sampler.clear_caches()

    def test_dcbs_cache_enabled_vs_disabled_consistency(self):
        sampler_cached = DCBSSampler.create_default(k=3, top_n=5, enable_caching=True)
        sampler_no_cache = DCBSSampler.create_no_cache(k=3, top_n=5)

        token_cached = sampler_cached.sample(self.logits, context=self.context)
        token_no_cache = sampler_no_cache.sample(self.logits, context=self.context)

        assert token_cached == token_no_cache, "Cached and non-cached results should be identical"


class TestSamplerInterfaceConsistency:
    """Test that all samplers implement the interface consistently."""

    def setup_method(self):
        self.logits = torch.tensor([1.0, 3.0, 2.0, 0.5, 2.5])
        self.filter_tokens = {1, 2, 4}

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

        self.samplers = {
            "greedy": GreedySampler(),
            "top_p": TopPSampler(p=0.9),
            "random": RandomSampler(),
            "dcbs": DCBSSampler(k=3, top_n=5),
        }

    def test_interface_consistency(self):
        for name, sampler in self.samplers.items():
            if name == "dcbs":
                token_id = sampler.sample(self.logits, context=self.context)
            else:
                token_id = sampler.sample(self.logits)

            assert isinstance(token_id, int)
            assert 0 <= token_id < len(self.logits)

            if name == "dcbs":
                token_id = sampler.sample(
                    self.logits, filter_tokens=self.filter_tokens, context=self.context
                )
            else:
                token_id = sampler.sample(self.logits, filter_tokens=self.filter_tokens)

            assert token_id in self.filter_tokens

    def test_inheritance(self):
        for sampler in self.samplers.values():
            assert isinstance(sampler, Sampler)

    def test_method_signatures(self):
        for sampler in self.samplers.values():
            assert hasattr(sampler, "sample")
            assert callable(getattr(sampler, "sample"))


class TestStatisticalProperties:
    """Test statistical properties of samplers."""

    def setup_method(self):
        self.biased_logits = torch.tensor([5.0, 1.0, 2.0, 0.5])

    def test_greedy_determinism(self):
        sampler = GreedySampler()
        results = [sampler.sample(self.biased_logits) for _ in range(100)]

        assert len(set(results)) == 1
        assert results[0] == 0

    def test_random_uniformity(self):
        uniform_logits = torch.zeros(4)
        sampler = RandomSampler()
        filter_tokens = {0, 1, 2, 3}

        results = [
            sampler.sample(uniform_logits, filter_tokens=filter_tokens)
            for _ in range(1000)
        ]

        unique_results = set(results)
        assert len(unique_results) >= 3

        counts = {i: results.count(i) for i in range(4)}
        min_count = min(counts.values())
        max_count = max(counts.values())

        assert max_count / min_count < 3.0

    def test_top_p_diversity(self):
        torch.manual_seed(42)
        sampler = TopPSampler(p=0.9)

        results = [sampler.sample(self.biased_logits) for _ in range(200)]
        unique_results = set(results)

        assert len(unique_results) > 1
        assert 0 in results

        proportion_max = results.count(0) / len(results)
        assert 0.5 < proportion_max < 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
