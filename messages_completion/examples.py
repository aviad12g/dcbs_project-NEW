"""
Example usage of the messages completion module.

Demonstrates the clean two-class API with configuration-based completion.
"""

from messages_completion import CompletionConfig, MessageCompleter


def basic_example():
    """Basic completion example with greedy sampling."""
    print("=== Basic Greedy Completion ===")
    
    try:
        # Create configuration
        config = CompletionConfig(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            max_new_tokens=16,
            sampling_method="greedy",
            deterministic=True
        )
        
        # Create completer
        completer = MessageCompleter(config)
        
        # Single conversation
        conversations = [[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one word."}
        ]]
        
        result = completer.complete(conversations)
        print(f"Input: {conversations[0][-1]['content']}")
        print(f"Completion: {result.text}")
        print(f"Method: {result.sampling_method}")
        
    except Exception as e:
        print(f"Example failed: {e}")


def dcbs_example():
    """DCBS sampling example with explicit configuration."""
    print("\n=== DCBS Completion ===")
    
    try:
        # Create explicit DCBS configuration
        config = CompletionConfig(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            max_new_tokens=20,
            sampling_method="dcbs",
            sampling_params={
                # Core DCBS parameters
                "k": 8,
                "top_n": 50,
                "clustering_method": "dbscan",
                
                # Deterministic DCBS parameters
                "weighted": False,  # Deterministic mode
                "levels": 1,
                "tie_break": "min_id",
                "seed": 42,
                
                # Optional frozen clusters
                "assignments_path": None
            },
            return_logprobs=True,
            deterministic=True
        )
        
        # Create completer
        completer = MessageCompleter(config)
        
        # Test conversation
        conversations = [[
            {"role": "system", "content": "You are creative."},
            {"role": "user", "content": "Write a short poem about AI."}
        ]]
        
        result = completer.complete(conversations)
        print(f"Input: {conversations[0][-1]['content']}")
        print(f"DCBS Completion: {result.text}")
        print(f"Method: {result.sampling_method}")
        
        if result.logprobs:
            avg_logprob = sum(result.logprobs) / len(result.logprobs)
            print(f"Average logprob: {avg_logprob:.3f}")
        
    except ImportError as e:
        print(f"DCBS not available: {e}")
        print("Install DCBS dependencies to use DCBS sampling.")
    
    except Exception as e:
        print(f"DCBS example failed: {e}")


def batch_example():
    """Batch completion example."""
    print("\n=== Batch Completion ===")
    
    try:
        # Create configuration for batch processing
        config = CompletionConfig(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            max_new_tokens=12,
            sampling_method="greedy",
            batch_size=4,
            deterministic=True
        )
        
        # Create completer
        completer = MessageCompleter(config)
        
        # Multiple conversations
        conversations = [
            [{"role": "system", "content": "You are terse."},
             {"role": "user", "content": "2+2?"}],
            [{"role": "system", "content": "You are terse."},
             {"role": "user", "content": "Capital of France?"}],
            [{"role": "system", "content": "You are terse."},
             {"role": "user", "content": "Largest planet?"}]
        ]
        
        # Batch completion
        result = completer.complete(conversations)
        
        print(f"Batch size: {result.batch_size}")
        print(f"Method: {result.sampling_method}")
        
        for i, completion in enumerate(result.completions):
            question = conversations[i][-1]['content']
            print(f"  Q: {question}")
            print(f"  A: {completion.text}")
        
    except Exception as e:
        print(f"Batch example failed: {e}")


def top_p_example():
    """Top-p sampling example."""
    print("\n=== Top-p Completion ===")
    
    try:
        # Create top-p configuration
        config = CompletionConfig(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            max_new_tokens=25,
            sampling_method="top_p",
            sampling_params={
                "p": 0.9,
                "temperature": 0.8
            },
            deterministic=False  # Top-p is non-deterministic
        )
        
        # Create completer
        completer = MessageCompleter(config)
        
        # Creative conversation
        conversations = [[
            {"role": "system", "content": "You are a creative storyteller."},
            {"role": "user", "content": "Start a story about a robot."}
        ]]
        
        result = completer.complete(conversations)
        print(f"Input: {conversations[0][-1]['content']}")
        print(f"Top-p Completion: {result.text}")
        print(f"Method: {result.sampling_method}")
        print(f"Deterministic: {config.is_deterministic}")
        
    except Exception as e:
        print(f"Top-p example failed: {e}")


def config_comparison():
    """Compare different configurations."""
    print("\n=== Configuration Comparison ===")
    
    configs = [
        ("Greedy", CompletionConfig(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            max_new_tokens=15,
            sampling_method="greedy"
        )),
        ("Top-p", CompletionConfig(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            max_new_tokens=15,
            sampling_method="top_p",
            sampling_params={"p": 0.9, "temperature": 0.7}
        )),
        ("DCBS", CompletionConfig(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            max_new_tokens=15,
            sampling_method="dcbs",
            sampling_params={"k": 6, "top_n": 30}
        ))
    ]
    
    conversation = [[
        {"role": "user", "content": "Explain quantum computing briefly."}
    ]]
    
    for name, config in configs:
        try:
            completer = MessageCompleter(config)
            result = completer.complete(conversation)
            print(f"{name:8}: {result.text}")
        except Exception as e:
            print(f"{name:8}: Failed - {e}")


def main():
    """Run all examples."""
    print("Messages Completion Module - Clean API Examples")
    print("=" * 60)
    
    try:
        basic_example()
        dcbs_example()
        batch_example()
        top_p_example()
        config_comparison()
        
    except Exception as e:
        print(f"Examples failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("\nAPI Summary:")
    print("- CompletionConfig: Configure model, sampling method, parameters")
    print("- MessageCompleter: Complete conversations based on config")
    print("- Supports: greedy, top_p, dcbs sampling methods")
    print("- Deterministic when possible, batch processing built-in")


if __name__ == "__main__":
    main()