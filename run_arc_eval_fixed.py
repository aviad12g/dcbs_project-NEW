"""
Fixed ARC evaluation script with proper timing measurement.

This version measures wall-clock inference time including model forward pass,
not just sampling post-processing time.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dcbs import (
    CacheConfig,
    DCBSSampler,
    GreedySampler,
    RandomSampler,
    SamplingContext,
    TopPSampler,
)


def load_model(model_name_or_path):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name_or_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )

    if device.type != "cuda":
        model = model.to(device)

    model.eval()
    return model, tokenizer, device


def create_prompt(question, options, use_cot=True):
    """Create prompt for ARC question."""
    prompt = f"Question: {question}\n\n"

    for i, option in enumerate(options):
        prompt += f"{chr(65 + i)}. {option}\n"

    if use_cot:
        prompt += "\nLet me think about this step by step.\n"

    prompt += "\nAnswer: "
    return prompt


def get_option_tokens(options, tokenizer):
    """Get token IDs for answer options (A, B, C, D)."""
    option_tokens = {}

    for i, option_text in enumerate(options):
        letter = chr(65 + i)  # A, B, C, D

        # Try different tokenizations
        token_ids = []
        for variant in [letter, f" {letter}", f"{letter}.", f" {letter}."]:
            tokens = tokenizer.encode(variant, add_special_tokens=False)
            if len(tokens) == 1:
                token_ids.append(tokens[0])
            elif len(tokens) > 1:
                token_ids.append(tokens[-1])

        if token_ids:
            option_tokens[letter] = token_ids[0]
        else:
            option_tokens[letter] = 65 + i  # Fallback

    return option_tokens


def evaluate_question_with_proper_timing(question_data, model, tokenizer, device, samplers, use_cot=True):
    """Evaluate a single question with PROPER wall-clock timing."""
    question = question_data["question"]
    options = question_data["options"]
    correct_answer = question_data["answer_key"]

    # Create prompt
    prompt = create_prompt(question, options, use_cot)
    option_tokens = get_option_tokens(options, tokenizer)

    # Tokenize input (ONCE - outside timing)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    results = {}
    
    # Create context once
    context = SamplingContext(
        embedding_layer=model.get_input_embeddings(),
        tokenizer=tokenizer,
        device=device,
    )

    # Test each sampler with PROPER timing
    for name, sampler in samplers.items():
        # START TIMING HERE - includes full inference + sampling
        start_time = time.time()

        try:
            # Model inference (THIS is what takes 552ms!)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1, :]

            # Filter to valid answer tokens
            valid_tokens = set(option_tokens.values())

            # Sampling (this is just 1-6ms)
            if name == "dcbs":
                selected_token = sampler.sample(
                    logits, context, filter_tokens=valid_tokens
                )
            else:
                selected_token = sampler.sample(logits, filter_tokens=valid_tokens)

            # END TIMING HERE - full wall-clock time
            elapsed_ms = (time.time() - start_time) * 1000

            # Find which option this corresponds to
            predicted_letter = None
            for letter, token_id in option_tokens.items():
                if token_id == selected_token:
                    predicted_letter = letter
                    break

            if predicted_letter is None:
                decoded = tokenizer.decode([selected_token]).strip()
                if decoded in option_tokens:
                    predicted_letter = decoded
                else:
                    predicted_letter = "A"  # Default fallback

            is_correct = predicted_letter == correct_answer

            results[name] = {
                "predicted": predicted_letter,
                "correct": is_correct,
                "time_ms": elapsed_ms,
            }

        except Exception as e:
            print(f"Error with {name} sampler: {e}")
            results[name] = {"predicted": "A", "correct": False, "time_ms": 0.0}

    return results


def main():
    parser = argparse.ArgumentParser(description="Fixed DCBS evaluation with proper timing")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",  # Default to instruct model
        help="Model name or path",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/arc_easy_full.json",
        help="Path to ARC-Easy data",
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="results/arc_fixed_evaluation.json", 
        help="Output file"
    )
    parser.add_argument("--limit", type=int, help="Limit number of questions")
    parser.add_argument(
        "--no_cot", action="store_true", help="Disable chain of thought"
    )
    parser.add_argument("--k", type=int, default=8, help="Number of clusters for DCBS")
    parser.add_argument("--top_n", type=int, default=50, help="Top N for DCBS")
    parser.add_argument(
        "--disable_cache", action="store_true", help="Disable DCBS caching for timing comparison"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}")
    if not os.path.exists(args.data):
        print(f"Data file not found: {args.data}")
        return 1

    with open(args.data) as f:
        questions = json.load(f)

    if args.limit:
        questions = questions[: args.limit]
        print(f"Limited to {args.limit} questions")

    print(f"Loaded {len(questions)} questions")

    # Load model
    model, tokenizer, device = load_model(args.model)

    # Create samplers with proper cache configuration
    if args.disable_cache:
        cache_config = CacheConfig(embedding_cache_size=0, enable_metrics=False)
        print("🚫 Cache disabled for timing comparison")
    else:
        cache_config = CacheConfig(embedding_cache_size=1000, enable_metrics=True)
        print("✅ Cache enabled")

    samplers = {
        "greedy": GreedySampler(),
        "top_p": TopPSampler(p=0.9),
        "dcbs": DCBSSampler.create_default(
            k=args.k, top_n=args.top_n, cache_config=cache_config.__dict__
        ),
        "random": RandomSampler(),
    }

    print(f"Testing {len(samplers)} sampling methods")
    print("⚠️  NOTE: This version measures FULL inference time (including model forward pass)")

    # Run evaluation
    all_results = []
    method_stats = {
        name: {"correct": 0, "total": 0, "times": []} for name in samplers.keys()
    }

    print("Starting evaluation...")
    start_time = time.time()

    for i, question in enumerate(questions):
        if i % 5 == 0:
            print(f"Progress: {i}/{len(questions)}")

        try:
            results = evaluate_question_with_proper_timing(
                question, model, tokenizer, device, samplers, use_cot=not args.no_cot
            )

            # Store individual result
            result_entry = {
                "question_id": question["id"],
                "question": question["question"],
                "correct_answer": question["answer_key"],
                "results": results,
            }
            all_results.append(result_entry)

            # Update stats
            for method, result in results.items():
                method_stats[method]["total"] += 1
                if result["correct"]:
                    method_stats[method]["correct"] += 1
                method_stats[method]["times"].append(result["time_ms"])

        except Exception as e:
            print(f"Error on question {i}: {e}")
            continue

    total_time = time.time() - start_time
    print(f"Evaluation completed in {total_time:.2f} seconds")

    # Calculate final statistics
    final_stats = {}
    for method, stats in method_stats.items():
        if stats["total"] > 0:
            accuracy = stats["correct"] / stats["total"]
            avg_time = np.mean(stats["times"]) if stats["times"] else 0

            final_stats[method] = {
                "accuracy": accuracy,
                "correct": stats["correct"],
                "total": stats["total"],
                "avg_time_ms": avg_time,
            }

    # Print results
    print("\n🎯 **FIXED RESULTS (with proper timing):**")
    print("-" * 60)
    for method, stats in final_stats.items():
        print(
            f"{method:10s}: {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']}) "
            f"avg: {stats['avg_time_ms']:.0f}ms"
        )

    # Save results
    output_data = {
        "config": {
            "model": args.model,
            "data_file": args.data,
            "num_questions": len(questions),
            "use_cot": not args.no_cot,
            "dcbs_k": args.k,
            "dcbs_top_n": args.top_n,
            "cache_disabled": args.disable_cache,
            "timing_method": "full_inference_plus_sampling",
        },
        "statistics": final_stats,
        "individual_results": all_results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {args.output}")

    # Show cache stats
    try:
        if not args.disable_cache and hasattr(samplers["dcbs"], "get_cache_stats"):
            cache_stats = samplers["dcbs"].get_cache_stats()
            if cache_stats.get("embedding_cache", {}).get("metrics"):
                hit_rate = cache_stats["embedding_cache"]["metrics"].hit_rate
                print(f"DCBS cache hit rate: {hit_rate:.2%}")
    except:
        pass

    return 0


if __name__ == "__main__":
    exit(main()) 