"""
Chat-based evaluation framework for language models.

This module provides a comprehensive framework for evaluating language models
using chat templates and conversational prompts.
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Remove sys.path.append line for clean imports

try:
    import numpy as np
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    from dcbs import (
        DCBSSampler,
        GreedySampler,
        RandomSampler,
        SamplingContext,
        TopPSampler,
    )

    # Import local modules
    from src.errors import eval_logger as logger
    from src.errors import setup_logging
    from src.token_utils import get_answer_token_ids, is_valid_token_prediction
    from src.visualization import generate_all_visualizations

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required packages are installed:")
    print("pip install torch transformers numpy tqdm")
    exit(1)


class ChatModel:
    """Wrapper for HuggingFace chat model with sampling support."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

        # Ensure chat template is available
        if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
            # Set a default chat template for Llama models
            tokenizer.chat_template = (
                "{% if messages[0]['role'] == 'system' %}"
                "{% set loop_messages = messages[1:] %}"
                "{% set system_message = messages[0]['content'] %}"
                "{% else %}"
                "{% set loop_messages = messages %}"
                "{% set system_message = false %}"
                "{% endif %}"
                "{% for message in loop_messages %}"
                "{% if loop.index0 == 0 and system_message %}"
                "{{ '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n' + system_message + '<|eot_id|>' }}"
                "{% endif %}"
                "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
                "{% if loop.last and add_generation_prompt %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
                "{% endif %}"
                "{% endfor %}"
            )

    def generate_response(
        self, messages: List[Dict[str, str]], sampler, max_new_tokens: int = 50
    ) -> str:
        """Generate a response using the specified sampler.

        Args:
            messages: List of chat messages
            sampler: Sampler instance to use for token selection
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            Generated response text
        """
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        # Generate using the sampler
        generated_tokens = []
        current_input = inputs.input_ids

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get model output
                outputs = self.model(current_input)
                logits = outputs.logits[:, -1, :].squeeze(0)  # Get last token logits

                # Sample next token using the specified sampler
                if isinstance(sampler, DCBSSampler):
                    next_token = sampler.sample(
                        logits, embedding=self.model.get_input_embeddings()
                    )
                else:
                    next_token = sampler.sample(logits)

                # Check for end of generation
                if next_token == self.tokenizer.eos_token_id:
                    break

                generated_tokens.append(next_token)

                # Update input for next iteration
                next_token_tensor = torch.tensor([[next_token]], device=self.device)
                current_input = torch.cat([current_input, next_token_tensor], dim=1)

        # Decode the generated tokens
        if generated_tokens:
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            response = ""

        return response.strip()

    def get_final_answer_logits(self, messages: List[Dict[str, str]]) -> torch.Tensor:
        """Get logits for final answer prediction.

        Args:
            messages: List of chat messages ending with final answer prompt

        Returns:
            Logits tensor for next token prediction
        """
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :].squeeze(0)

        return logits


def create_cot_messages(sentence: str, options: List[str]) -> List[Dict[str, str]]:
    """Create messages for chain-of-thought reasoning.

    Args:
        sentence: The problem sentence
        options: List of answer options

    Returns:
        List of chat messages for CoT reasoning
    """
    system_message = (
        "You are a helpful assistant that thinks step by step. "
        "When given a problem, first reason through it carefully, "
        "then provide your final answer."
    )

    user_prompt = f"{sentence}\n\nOptions:\n"
    for i, option in enumerate(options, 1):
        user_prompt += f"{i}. {option}\n"
    user_prompt += (
        "\nPlease think through this step by step and explain your reasoning."
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]


def create_final_answer_messages(
    cot_reasoning: str, sentence: str, options: List[str]
) -> List[Dict[str, str]]:
    """Create messages for final answer extraction.

    Args:
        cot_reasoning: The chain-of-thought reasoning
        sentence: The problem sentence
        options: List of answer options

    Returns:
        List of chat messages for final answer
    """
    system_message = (
        "You are a helpful assistant. Based on the reasoning provided, "
        "give the final answer. Respond with just the answer option."
    )

    user_prompt = f"Problem: {sentence}\n\nOptions:\n"
    for i, option in enumerate(options, 1):
        user_prompt += f"{i}. {option}\n"

    user_prompt += (
        f"\nReasoning: {cot_reasoning}\n\nBased on this reasoning, the answer is:"
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]


def process_example_with_cot(example: Dict, chat_model: ChatModel, sampler) -> Dict:
    """Process an example using chain-of-thought reasoning.

    Args:
        example: Example data from benchmark
        chat_model: ChatModel instance
        sampler: Sampler to use for generation

    Returns:
        Processed example with results
    """
    prompt_id = example.get("id", "unknown_id")

    # Validate example
    if not ("sentence" in example and "option1" in example and "option2" in example):
        raise DataError(f"Example {prompt_id} missing required fields")

    sentence = example["sentence"]
    options = [example["option1"], example["option2"]]
    correct_option = example.get("correct_option", "1")
    correct_answer = example[f"option{correct_option}"]

    result = {
        "prompt_id": prompt_id,
        "sentence": sentence,
        "options": options,
        "correct_answer": correct_answer,
        "correct_option": correct_option,
    }

    try:
        # Step 1: Generate chain-of-thought reasoning
        cot_messages = create_cot_messages(sentence, options)
        cot_reasoning = chat_model.generate_response(
            cot_messages, sampler, max_new_tokens=100
        )
        result["cot_reasoning"] = cot_reasoning

        # Step 2: Get final answer using CoT reasoning
        final_messages = create_final_answer_messages(cot_reasoning, sentence, options)
        final_logits = chat_model.get_final_answer_logits(final_messages)

        # Get token IDs for answer options
        answer_ids = {}
        for option in options:
            # Try with leading space first, then without
            token_ids_with_space = get_answer_token_ids(
                f" {option}", chat_model.tokenizer, add_leading_space=False
            )
            token_ids_no_space = get_answer_token_ids(
                option, chat_model.tokenizer, add_leading_space=False
            )

            # Use the single-token version if available
            if len(token_ids_with_space) == 1:
                answer_ids[option] = token_ids_with_space[0]
            elif len(token_ids_no_space) == 1:
                answer_ids[option] = token_ids_no_space[0]
            else:
                # Fall back to first token
                answer_ids[option] = (
                    token_ids_with_space[0]
                    if token_ids_with_space
                    else token_ids_no_space[0]
                )

        result["answer_ids"] = answer_ids
        result["filter_tokens"] = set(answer_ids.values())
        result["correct_id"] = answer_ids[correct_answer]
        result["logits"] = final_logits

        # Calculate answer probabilities
        all_probs = torch.softmax(final_logits, dim=0)
        answer_probs = {
            option: all_probs[token_id].item()
            for option, token_id in answer_ids.items()
        }
        result["answer_probs"] = answer_probs

        return result

    except Exception as e:
        raise EvaluationError(f"Error processing example {prompt_id}: {str(e)}")


def evaluate_with_sampler(
    result: Dict, chat_model: ChatModel, sampler, sampler_name: str
) -> Dict:
    """Evaluate a single example with a specific sampler.

    Args:
        result: Processed example data
        chat_model: ChatModel instance
        sampler: Sampler to use
        sampler_name: Name of the sampler for logging

    Returns:
        Evaluation results
    """
    start_time = time.time()

    logits = result["logits"]
    filter_tokens = result["filter_tokens"]
    correct_id = result["correct_id"]
    correct_answer = result["correct_answer"]

    # Sample using the specified sampler
    if isinstance(sampler, DCBSSampler):
        pred_id = sampler.sample(
            logits,
            filter_tokens=filter_tokens,
            embedding=chat_model.model.get_input_embeddings(),
        )
    else:
        pred_id = sampler.sample(logits, filter_tokens=filter_tokens)

    elapsed_ms = (time.time() - start_time) * 1000

    # Check if prediction is correct
    correct = is_valid_token_prediction(
        pred_id, correct_id, correct_answer, chat_model.tokenizer
    )

    return {
        "sampler": sampler_name,
        "pred_id": pred_id,
        "correct": correct,
        "elapsed_ms": elapsed_ms,
    }


def main(args):
    """Main evaluation function."""
    try:
        setup_logging(args.log_level if hasattr(args, "log_level") else "INFO")
        logger.info(f"Starting chat-based evaluation with CoT")

        # Load model and tokenizer
        logger.info(f"Loading model: {args.model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            token=os.environ.get("HF_HUB_TOKEN"),
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, token=os.environ.get("HF_HUB_TOKEN")
        )

        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        chat_model = ChatModel(model, tokenizer)
        logger.info(f"Model loaded on device: {chat_model.device}")

        # Load benchmark data
        logger.info(f"Loading benchmark: {args.benchmark}")
        with open(args.benchmark, "r") as f:
            benchmark_data = json.load(f)

        if args.limit:
            benchmark_data = benchmark_data[: args.limit]
            logger.info(f"Limited to {args.limit} examples")

        # Initialize samplers
        samplers = {
            "greedy": GreedySampler(),
            "top-p": TopPSampler(p=args.top_p),
            "dcbs": DCBSSampler(k=args.k, top_n=args.top_n),
            "random": RandomSampler(),
        }

        # Results storage
        all_results = []
        method_stats = {
            name: {"correct": 0, "total": 0, "avg_time": 0.0}
            for name in samplers.keys()
        }

        # Process examples
        for i, example in enumerate(benchmark_data):
            if (i + 1) % 10 == 0:
                logger.info(f"Processing example {i + 1}/{len(benchmark_data)}")

            try:
                # Process with chain-of-thought for each sampler
                for sampler_name, sampler in samplers.items():
                    logger.debug(
                        f"Processing example {example.get('id', i)} with {sampler_name}"
                    )

                    # Process example with CoT
                    processed_result = process_example_with_cot(
                        example, chat_model, sampler
                    )

                    # Evaluate with sampler
                    eval_result = evaluate_with_sampler(
                        processed_result, chat_model, sampler, sampler_name
                    )

                    # Combine results
                    final_result = {**processed_result, **eval_result}
                    all_results.append(final_result)

                    # Update statistics
                    stats = method_stats[sampler_name]
                    stats["total"] += 1
                    if eval_result["correct"]:
                        stats["correct"] += 1
                    stats["avg_time"] += eval_result["elapsed_ms"]

            except Exception as e:
                log_exception(e, logger)
                continue

        # Calculate final statistics
        for sampler_name, stats in method_stats.items():
            if stats["total"] > 0:
                accuracy = stats["correct"] / stats["total"] * 100
                avg_time = stats["avg_time"] / stats["total"]
                stats["accuracy"] = accuracy
                stats["avg_time"] = avg_time

                logger.info(
                    f"{sampler_name}: {accuracy:.2f}% accuracy ({stats['correct']}/{stats['total']}), "
                    f"avg time: {avg_time:.2f}ms"
                )

        # Save results
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

        output_data = {
            "config": {
                "model": args.model,
                "benchmark": args.benchmark,
                "top_p": args.top_p,
                "k": args.k,
                "top_n": args.top_n,
                "limit": args.limit,
            },
            "statistics": method_stats,
            "results": all_results,
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Results saved to {args.output}")
        return 0

    except Exception as e:
        log_exception(e, logger)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chat-based evaluation with chain-of-thought"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument(
        "--benchmark", type=str, required=True, help="Benchmark JSON file"
    )
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p value for nucleus sampling"
    )
    parser.add_argument("--k", type=int, default=8, help="Number of clusters for DCBS")
    parser.add_argument("--top-n", type=int, default=50, help="Top-n tokens for DCBS")
    parser.add_argument("--limit", type=int, help="Limit number of examples")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()
    exit_code = main(args)
    exit(exit_code)
