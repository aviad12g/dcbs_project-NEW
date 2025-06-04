import sys
from pathlib import Path

import torch
import pytest

# Import modules directly from the src directory to avoid loading the entire
# project package (which requires heavy optional dependencies).
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dcbs.optimizations import (
    OptimizationConfig,
    BatchDCBSProcessor,
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


def test_sequential_batch_sample():
    context = create_test_context()
    config = OptimizationConfig(
        enable_parallel_processing=False,
        use_gpu_clustering=False,
        use_mixed_precision=False,
    )
    processor = BatchDCBSProcessor(config, DCBSCacheManager(CacheConfig()))

    logits_batch = torch.stack([logits_cluster, logits_small])
    filter_tokens_batch = [None, None]

    results = processor._sequential_batch_sample(
        logits_batch, filter_tokens_batch, context, k=2, top_n=5
    )

    assert results == [3, 5]


def test_parallel_batch_sample():
    context = create_test_context()
    config = OptimizationConfig(
        enable_parallel_processing=True,
        max_workers=2,
        use_gpu_clustering=False,
        use_mixed_precision=False,
    )
    processor = BatchDCBSProcessor(config, DCBSCacheManager(CacheConfig()))

    logits_batch = torch.stack([logits_cluster] * 6)
    filter_tokens_batch = [None] * 6

    results = processor._parallel_batch_sample(
        logits_batch, filter_tokens_batch, context, k=2, top_n=5
    )

    assert len(results) == 6
    assert set(results) == {3}


def test_memory_efficient_dcbs():
    context = create_test_context()
    mdcbs = MemoryEfficientDCBS()

    result = mdcbs.sample_with_memory_limit(
        logits_cluster, None, context, k=2, top_n=5
    )

    assert result == 3
