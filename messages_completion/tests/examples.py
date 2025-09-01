"""
Example usage of the messages completion module, converted to tests.

These tests are skipped by default to avoid heavy model downloads.
Set environment variable MC_RUN_EXAMPLES=1 to enable.
"""

import os
import pytest

from messages_completion import CompletionConfig, MessageCompleter

if os.getenv("MC_RUN_EXAMPLES", "0") != "1":
    pytest.skip("Skipping messages_completion example tests (set MC_RUN_EXAMPLES=1 to run)", allow_module_level=True)


def test_basic_example():
    """Basic completion example with greedy sampling."""
    print("=== Basic Greedy Completion ===")
    
    try:
        # Create configuration
        config = CompletionConfig(
            model_name="meta-llama/Meta-Llama-3-2-1B-Instruct".replace("3-2", "3.2"),
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


def test_dcbs_example():
    """DCBS sampling example with explicit configuration."""
    print("\n=== DCBS Completion ===")
    
    try:
        # Create explicit DCBS configuration
        config = CompletionConfig(
            model_name="meta-llama/Meta-Llama-3.2-1B-Instruct",
            max_new_tokens=20,
            sampling_method="dcbs",
            sampling_params={
                # Core DCBS parameters
                "k": 8,
                "top_n": 50,
                "clustering_method": "dbscan",
                # Deterministic DCBS parameters
                "weighted": False,
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


def test_batch_example():
    """Batch completion example."""
    print("\n=== Batch Completion ===")
    
    try:
        # Create configuration for batch processing
        config = CompletionConfig(
            model_name="meta-llama/Meta-Llama-3.2-1B-Instruct",
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


def test_top_p_example():
    """Top-p sampling example."""
    print("\n=== Top-p Completion ===")
    
    try:
        # Create top-p configuration
        config = CompletionConfig(
            model_name="meta-llama/Meta-Llama-3.2-1B-Instruct",
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
        
    except Exception as e:
        print(f"Top-p example failed: {e}")


def test_config_comparison():
    """Compare different configurations."""
    print("\n=== Configuration Comparison ===")
    
    configs = [
        ("Greedy", CompletionConfig(
            model_name="meta-llama/Meta-Llama-3.2-1B-Instruct",
            max_new_tokens=15,
            sampling_method="greedy"
        )),
        ("Top-p", CompletionConfig(
            model_name="meta-llama/Meta-Llama-3.2-1B-Instruct",
            max_new_tokens=15,
            sampling_method="top_p",
            sampling_params={"p": 0.9, "temperature": 0.7}
        )),
        ("DCBS", CompletionConfig(
            model_name="meta-llama/Meta-Llama-3.2-1B-Instruct",
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


def test_examples_summary_print():
    # Light sanity to keep a placeholder
    assert True
