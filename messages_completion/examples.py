"""
Example usage of the messages completion module.

Demonstrates GPU-based completion with batch invariance.
"""

import copy
from messages_completion import MessageCompleter, HuggingFaceModelInterface


def basic_gpu_example():
    """Basic GPU completion example with greedy mode."""
    print("=== Basic GPU Completion Example ===")
    
    try:
        # Initialize model (requires GPU)
        model = HuggingFaceModelInterface("meta-llama/Meta-Llama-3-8B-Instruct")
        model.set_seed(42)  # For deterministic results
        
        # Create completer
        comp = MessageCompleter(model, max_new_tokens=16)
        
        # Single conversation
        convos = [[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one word."}
        ]]
        
        result = comp.complete(convos, use_batching=False, return_logprobs=False)
        print(f"Input: {convos[0][-1]['content']}")
        print(f"Completion: {result.text}")
        print(f"Tokens: {len(result.token_ids)}")
        
    except Exception as e:
        print(f"Example failed: {e}")
        print("This example requires a GPU and the Meta-Llama model.")


def batch_invariance_example():
    """Demonstrate batch invariance - key feature for deterministic completion."""
    print("\n=== Batch Invariance Example ===")
    
    try:
        # Initialize model
        model = HuggingFaceModelInterface("meta-llama/Meta-Llama-3-8B-Instruct")
        model.set_seed(42)
        comp = MessageCompleter(model, max_new_tokens=8)
        
        # Test conversations
        convos = [
            [{"role": "system", "content": "You are terse."},
             {"role": "user", "content": "2+2?"}],
            [{"role": "system", "content": "You are terse."},
             {"role": "user", "content": "Capital of France?"}]
        ]
        
        # Sequential processing (N calls with batch=1)
        print("Sequential processing...")
        seq_results = []
        for convo in convos:
            out = comp.complete([copy.deepcopy(convo)], use_batching=False, return_logprobs=False)
            seq_results.append(out.token_ids)
            print(f"  '{convo[-1]['content']}' -> {out.text}")
        
        # Batch processing (1 call with batch=N)
        print("\nBatch processing...")
        batch_result = comp.complete(copy.deepcopy(convos), use_batching=True, return_logprobs=False)
        batch_results = [c.token_ids for c in batch_result.completions]
        
        for i, completion in enumerate(batch_result.completions):
            print(f"  '{convos[i][-1]['content']}' -> {completion.text}")
        
        # Verify batch invariance
        invariant = seq_results == batch_results
        print(f"\nBatch invariance verified: {invariant}")
        
        if not invariant:
            print("Sequential and batch results differ!")
            for i, (seq, batch) in enumerate(zip(seq_results, batch_results)):
                if seq != batch:
                    print(f"  Conversation {i}: seq={seq} vs batch={batch}")
        
    except Exception as e:
        print(f"Batch invariance example failed: {e}")


def logprobs_example():
    """Example with log probabilities."""
    print("\n=== Log Probabilities Example ===")
    
    try:
        model = HuggingFaceModelInterface("meta-llama/Meta-Llama-3-8B-Instruct")
        model.set_seed(42)
        comp = MessageCompleter(model, max_new_tokens=6)
        
        convos = [[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Count to 3."}
        ]]
        
        result = comp.complete(convos, use_batching=False, return_logprobs=True)
        
        print(f"Input: {convos[0][-1]['content']}")
        print(f"Completion: {result.text}")
        print(f"Token IDs: {result.token_ids}")
        
        if result.logprobs:
            print("Log probabilities:")
            for i, (token_id, logprob) in enumerate(zip(result.token_ids, result.logprobs)):
                print(f"  Token {i}: ID={token_id}, logprob={logprob:.3f}")
            
            avg_logprob = sum(result.logprobs) / len(result.logprobs)
            print(f"Average logprob: {avg_logprob:.3f}")
        
    except Exception as e:
        print(f"Logprobs example failed: {e}")


def main():
    """Run all examples."""
    print("Messages Completion Module - GPU Examples")
    print("=" * 50)
    
    try:
        basic_gpu_example()
        batch_invariance_example()
        logprobs_example()
        
    except Exception as e:
        print(f"Examples failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("GPU examples completed!")
    print("\nNote: These examples require:")
    print("- CUDA-capable GPU")
    print("- Sufficient GPU memory for Meta-Llama-3-8B-Instruct")
    print("- HuggingFace transformers library")


if __name__ == "__main__":
    main()