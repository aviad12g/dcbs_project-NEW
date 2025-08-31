"""
Test batch invariance with the new clean API.
"""

import copy
from messages_completion import CompletionConfig, MessageCompleter

EXAMPLE_CONVOS = [
    [{"role":"system","content":"You are terse."},
     {"role":"user","content":"2+2?"}],
    [{"role":"system","content":"You are terse."},
     {"role":"user","content":"Capital of France?"}],
]

def test_batch_invariance():
    """Test that batch processing produces identical results to sequential processing."""
    # Create deterministic greedy configuration
    config = CompletionConfig(
        model_name="gpt2",
        max_new_tokens=8,
        sampling_method="greedy",
        deterministic=True
    )
    
    completer = MessageCompleter(config)
    
    # Sequential processing (one at a time)
    seq_ids = []
    for convo in EXAMPLE_CONVOS:
        result = completer.complete([copy.deepcopy(convo)])
        seq_ids.append(result.token_ids)
    
    # Batch processing (all at once)
    batch_result = completer.complete(copy.deepcopy(EXAMPLE_CONVOS))
    batch_ids = [c.token_ids for c in batch_result.completions]
    
    # Results must be identical
    assert seq_ids == batch_ids, f"Sequential {seq_ids} != Batch {batch_ids}"

def test_dcbs_batch_invariance():
    """Test batch invariance with DCBS sampling."""
    try:
        # Create deterministic DCBS configuration
        config = CompletionConfig(
            model_name="gpt2",
            max_new_tokens=6,
            sampling_method="dcbs",
            sampling_params={
                "k": 4,
                "top_n": 20,
                "clustering_method": "dbscan"
            },
            deterministic=True
        )
        
        completer = MessageCompleter(config)
        
        # Sequential processing
        seq_ids = []
        for convo in EXAMPLE_CONVOS:
            result = completer.complete([copy.deepcopy(convo)])
            seq_ids.append(result.token_ids)
        
        # Batch processing
        batch_result = completer.complete(copy.deepcopy(EXAMPLE_CONVOS))
        batch_ids = [c.token_ids for c in batch_result.completions]
        
        # Results must be identical for deterministic DCBS
        assert seq_ids == batch_ids, f"DCBS Sequential {seq_ids} != Batch {batch_ids}"
        
    except ImportError:
        # DCBS not available, skip test
        print("DCBS not available, skipping DCBS batch invariance test")

if __name__ == "__main__":
    test_batch_invariance()
    test_dcbs_batch_invariance()
    print("All batch invariance tests passed!")