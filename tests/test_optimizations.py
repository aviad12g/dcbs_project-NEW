import sys
from pathlib import Path

import torch
import pytest

# Import modules directly from the src directory to avoid loading the entire
# project package (which requires heavy optional dependencies).
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dcbs.optimizations import (
    OptimizationConfig,
    MemoryEfficientDCBS,
)
from dcbs.cache_manager import DCBSCacheManager, CacheConfig
from dcbs.samplers.base import SamplingContext


def create_test_context():
    embedding = torch.nn.Embedding(6, 2)
    weights = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [20.0, 20.0],
        ],
        dtype=torch.float32,
    )
    embedding.weight.data = weights
    return SamplingContext(embedding_layer=embedding, device=torch.device("cpu"))


logits_cluster = torch.tensor([-1.0, -1.1, -1.2, 4.0, 3.9, 0.5])
logits_small = torch.tensor([0.0, 0.5, 0.8, -1.0, -1.0, 1.0])


def test_dcbs_sampler_sample_batch_equivalent_to_legacy_behavior():
    # Replace legacy BatchDCBSProcessor tests with DCBS sampler batch path
    from dcbs.samplers.dcbs_sampler import DCBSSampler
    from dcbs.clustering import KMeansClusterer, TopNCandidateSelector
    from dcbs.category_sampling import CategorySampler, GreedyCategorySelector, GreedyTokenSelector

    context = create_test_context()
    sampler = DCBSSampler(
        clusterer=KMeansClusterer(k=2),
        candidate_selector=TopNCandidateSelector(top_n=5),
        category_sampler=CategorySampler(GreedyCategorySelector(), GreedyTokenSelector()),
        context=context,
        enable_caching=False,
    )

    logits_batch = torch.stack([logits_cluster, logits_small])
    filter_tokens_batch = [None, None]

    results = sampler.sample_batch(logits_batch, filter_tokens_batch)
    assert isinstance(results, list) and len(results) == 2


def test_dcbs_sampler_parallel_pool_works():
    from dcbs.samplers.dcbs_sampler import DCBSSampler
    from dcbs.clustering import KMeansClusterer, TopNCandidateSelector
    from dcbs.category_sampling import CategorySampler, GreedyCategorySelector, GreedyTokenSelector

    context = create_test_context()
    sampler = DCBSSampler(
        clusterer=KMeansClusterer(k=2),
        candidate_selector=TopNCandidateSelector(top_n=5),
        category_sampler=CategorySampler(GreedyCategorySelector(), GreedyTokenSelector()),
        context=context,
        enable_caching=False,
    )

    logits_batch = torch.stack([logits_cluster] * 6)
    filter_tokens_batch = [None] * 6

    results = sampler.sample_batch(logits_batch, filter_tokens_batch)
    assert len(results) == 6


def test_memory_efficient_dcbs():
    context = create_test_context()
    mdcbs = MemoryEfficientDCBS()

    result = mdcbs.sample_with_memory_limit(
        logits_cluster, None, context, k=2, top_n=5
    )

    assert result == 3
