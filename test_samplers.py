#!/usr/bin/env python3
"""
Test script to verify the new sampler classes work correctly.
"""

import numpy as np
import torch

from dcbs import DCBSSampler, GreedySampler, RandomSampler, TopPSampler


def test_samplers():
    """Test all sampler classes with example data."""
    print("Testing DCBS Sampler Classes")
    print("=" * 40)

    # Create example logits (vocabulary size 100)
    torch.manual_seed(42)
    logits = torch.randn(100)

    # Create example filter tokens (simulate answer options)
    filter_tokens = {25, 67}  # Two answer options

    # Create a mock embedding layer for DCBS
    embedding_dim = 64
    vocab_size = 100
    embedding = torch.nn.Embedding(vocab_size, embedding_dim)

    # Initialize samplers
    greedy = GreedySampler()
    top_p = TopPSampler(p=0.9)
    dcbs = DCBSSampler(k=4, top_n=20)
    random_sampler = RandomSampler()

    print(f"Input logits shape: {logits.shape}")
    print(f"Filter tokens: {filter_tokens}")
    print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    print()

    # Test each sampler
    samplers = [
        ("Greedy", greedy),
        ("Top-p", top_p),
        ("DCBS", dcbs),
        ("Random", random_sampler),
    ]

    results = {}

    for name, sampler in samplers:
        print(f"Testing {name} Sampler:")

        try:
            # Test without filtering
            if isinstance(sampler, DCBSSampler):
                token_no_filter = sampler.sample(logits, embedding=embedding)
            else:
                token_no_filter = sampler.sample(logits)

            # Test with filtering
            if isinstance(sampler, DCBSSampler):
                token_with_filter = sampler.sample(
                    logits, filter_tokens=filter_tokens, embedding=embedding
                )
            else:
                token_with_filter = sampler.sample(logits, filter_tokens=filter_tokens)

            print(f"  Without filtering: token {token_no_filter}")
            print(f"  With filtering: token {token_with_filter}")
            print(f"  Filter check: {token_with_filter in filter_tokens}")

            results[name] = {
                "no_filter": token_no_filter,
                "with_filter": token_with_filter,
                "valid": token_with_filter in filter_tokens,
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {"error": str(e)}

        print()

    # Verify deterministic behavior
    print("Testing Deterministic Behavior:")
    print("-" * 30)

    for name, sampler in [("Greedy", greedy), ("DCBS", dcbs)]:
        print(f"{name} Sampler:")

        # Run multiple times to check consistency
        samples = []
        for i in range(3):
            if isinstance(sampler, DCBSSampler):
                token = sampler.sample(
                    logits, filter_tokens=filter_tokens, embedding=embedding
                )
            else:
                token = sampler.sample(logits, filter_tokens=filter_tokens)
            samples.append(token)

        print(f"  Multiple runs: {samples}")
        print(f"  Deterministic: {len(set(samples)) == 1}")
        print()

    # Test edge cases
    print("Testing Edge Cases:")
    print("-" * 20)

    # Single token filter
    single_filter = {25}
    token = greedy.sample(logits, filter_tokens=single_filter)
    print(f"Single token filter: {token} (should be 25)")

    # Empty filter (should work without filtering)
    try:
        empty_filter = set()
        token = greedy.sample(logits, filter_tokens=empty_filter)
        print(f"Empty filter: token {token}")
    except Exception as e:
        print(f"Empty filter error: {e}")

    print("\nAll tests completed!")
    return results


if __name__ == "__main__":
    test_samplers()
