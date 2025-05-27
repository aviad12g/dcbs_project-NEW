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
from src.errors import setup_logging, DataError, EvaluationError, log_exception
from src.evaluation_core.template_manager import ChatTemplateManager
from src.token_utils import get_answer_token_ids, is_valid_token_prediction
from src.visualization import generate_all_visualizations


class ChatModel:
    """Wrapper for HuggingFace chat model with sampling support."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

        # Ensure chat template is available using the template manager
        ChatTemplateManager.setup_chat_template(self.tokenizer, self.model.config._name_or_path)
        
        # Validate that the chat template is working correctly
        if not ChatTemplateManager.validate_template(self.tokenizer, self.model.config._name_or_path):
            logger.warning(
                "Chat template validation failed. Using fallback template."
            )

    def generate_response(
        self, messages: List[Dict[str, str]], sampler, max_new_tokens: int = 50
    ) -> str:
        """Generate a response using the specified sampler with token-by-token generation.

        This ensures each sampler produces genuinely different reasoning that reflects
        the sampler's characteristics for scientifically accurate comparison.

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

        # Generate using the sampler with token-by-token approach
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
                        logits, context=SamplingContext(
                            embedding_layer=self.model.get_input_embeddings(),
                            tokenizer=self.tokenizer,
                            device=self.device
                        )
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

    # Validate example - handle both old and new data formats
    if "sentence" in example and "option1" in example and "option2" in example:
        # Old format
        sentence = example["sentence"]
        options = [example["option1"], example["option2"]]
        correct_option = example.get("correct_option", "1")
        correct_answer = example[f"option{correct_option}"]
    elif "question" in example and "options" in example:
        # New format
        sentence = example["question"]
        options = example["options"]
        correct_option = example.get("correct_option", "1")
        # Convert 1-based index to 0-based for options array
        correct_idx = int(correct_option) - 1
        if 0 <= correct_idx < len(options):
            correct_answer = options[correct_idx]
        else:
            raise DataError(f"Example {prompt_id} has invalid correct_option: {correct_option}")
    else:
        raise DataError(f"Example {prompt_id} missing required fields")

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

        # Get token IDs for answer options with improved strategy
        answer_ids = {}
        for option in options:
            # Try with leading space first, then without
            token_ids_with_space = get_answer_token_ids(
                f" {option}", chat_model.tokenizer, add_leading_space=False
            )
            token_ids_no_space = get_answer_token_ids(
                option, chat_model.tokenizer, add_leading_space=False
            )

            # Smart token selection strategy
            if len(token_ids_with_space) == 1:
                answer_ids[option] = token_ids_with_space[0]
            elif len(token_ids_no_space) == 1:
                answer_ids[option] = token_ids_no_space[0]
            elif len(token_ids_with_space) >= 2:
                # For multi-token options, prefer the first token (more distinctive)
                # unless it's a common word, then use second token
                first_token = token_ids_with_space[0]
                decoded_first = chat_model.tokenizer.decode([first_token]).strip()
                
                # If first token is very short or common, use second token
                if len(decoded_first) <= 2 or decoded_first.lower() in [' the', ' a', ' an', ' it', ' they']:
                    answer_ids[option] = token_ids_with_space[1] if len(token_ids_with_space) > 1 else first_token
                else:
                    answer_ids[option] = first_token
            elif len(token_ids_no_space) >= 2:
                # Same logic for no-space tokens
                first_token = token_ids_no_space[0]
                decoded_first = chat_model.tokenizer.decode([first_token]).strip()
                
                if len(decoded_first) <= 2 or decoded_first.lower() in ['the', 'a', 'an', 'it', 'they']:
                    answer_ids[option] = token_ids_no_space[1] if len(token_ids_no_space) > 1 else first_token
                else:
                    answer_ids[option] = first_token
            else:
                # Fallback
                answer_ids[option] = (
                    token_ids_with_space[0]
                    if token_ids_with_space
                    else token_ids_no_space[0]
                )

        # Check for duplicate token IDs and fix them
        token_counts = {}
        for option, token_id in answer_ids.items():
            if token_id in token_counts:
                token_counts[token_id].append(option)
            else:
                token_counts[token_id] = [option]
        
        # If duplicates exist, try alternative tokenization
        for token_id, options_list in token_counts.items():
            if len(options_list) > 1:
                logger.warning(f"Duplicate token {token_id} for options: {options_list}")
                # Try to fix by using different parts of the tokens
                for i, option in enumerate(options_list):
                    tokens_space = get_answer_token_ids(f" {option}", chat_model.tokenizer, add_leading_space=False)
                    tokens_no_space = get_answer_token_ids(option, chat_model.tokenizer, add_leading_space=False)
                    
                    # Try using middle token if available
                    if len(tokens_space) >= 3:
                        answer_ids[option] = tokens_space[1]  # Use second token
                    elif len(tokens_space) >= 2:
                        answer_ids[option] = tokens_space[-1]  # Use last token
                    elif len(tokens_no_space) >= 2:
                        answer_ids[option] = tokens_no_space[-1]  # Use last token
                    # If still the same, just keep original but log warning
                    
        # Verify uniqueness
        final_tokens = set(answer_ids.values())
        if len(final_tokens) < len(options):
            logger.error(f"Still have duplicate tokens after fix: {len(options)} options -> {len(final_tokens)} unique tokens")

        result["answer_ids"] = answer_ids
        result["filter_tokens"] = list(answer_ids.values())  # Convert set to list for JSON serialization
        result["correct_id"] = answer_ids[correct_answer]
        # Note: logits tensor removed from results for JSON serialization

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
    result: Dict, chat_model: ChatModel, sampler, sampler_name: str, final_logits: torch.Tensor
) -> Dict:
    """Evaluate a single example with a specific sampler.

    Args:
        result: Processed example data
        chat_model: ChatModel instance
        sampler: Sampler to use
        sampler_name: Name of the sampler for logging
        final_logits: Logits tensor for final answer prediction

    Returns:
        Evaluation results
    """
    start_time = time.time()

    filter_tokens = set(result["filter_tokens"])  # Convert back to set for filtering
    correct_id = result["correct_id"]
    correct_answer = result["correct_answer"]

    # Sample using the specified sampler
    if isinstance(sampler, DCBSSampler):
        pred_id = sampler.sample(
            final_logits,
            filter_tokens=filter_tokens,
            context=SamplingContext(
                embedding_layer=chat_model.model.get_input_embeddings(),
                tokenizer=chat_model.tokenizer,
                device=chat_model.device
            )
        )
    else:
        pred_id = sampler.sample(final_logits, filter_tokens=filter_tokens)

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
            "top_p": TopPSampler(p=args.top_p),
            "dcbs": DCBSSampler.create_default(k=args.k, top_n=args.top_n),
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
                # Generate CoT reasoning ONCE using Greedy (most reliable)
                base_processed_result = process_example_with_cot(
                    example, chat_model, samplers["greedy"]
                )
                
                # Get logits ONCE for all samplers using the same CoT reasoning
                final_messages = create_final_answer_messages(
                    base_processed_result["cot_reasoning"], 
                    base_processed_result["sentence"], 
                    base_processed_result["options"]
                )
                final_logits = chat_model.get_final_answer_logits(final_messages)

                # Now evaluate each sampler using the SAME CoT and logits
                for sampler_name, sampler in samplers.items():
                    logger.debug(
                        f"Processing example {example.get('id', i)} with {sampler_name}"
                    )

                    # Use the SAME base result and logits for all samplers
                    eval_result = evaluate_with_sampler(
                        base_processed_result, chat_model, sampler, sampler_name, final_logits
                    )

                    # Combine results - use base_processed_result for consistency
                    final_result = {**base_processed_result, **eval_result}
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
    parser = argparse.ArgumentParser(description="Chat-based evaluation with CoT")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--benchmark", required=True, help="Benchmark JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--limit", type=int, help="Limit number of examples")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p value")
    parser.add_argument("--k", type=int, default=3, help="DCBS k value")
    parser.add_argument("--top_n", type=int, default=50, help="DCBS top_n value")
    parser.add_argument("--log_level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    exit(main(args))
