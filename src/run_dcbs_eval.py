#!/usr/bin/env python3
"""
Evaluation script comparing DCBS against other sampling methods.

Methods: DCBS, Greedy, Top-p, Random
Reference: Smith, J. et al. (2023). "DCBS: Semantically Diverse Sampling for LMs."
"""

import argparse
import csv
import json
import os
import random
import time
from typing import Any, Dict, List, NamedTuple, Optional, Set, TypedDict, Union

import psutil
import torch
import yaml

from dcbs import DCBSSampler, GreedySampler, RandomSampler, SamplingContext, TopPSampler
from load_model import load_model_and_tokenizer
from src.errors import (
    ConfigurationError,
    DataError,
    EvaluationError,
    MemoryProfiler,
)
from src.errors import eval_logger as logger
from src.errors import (
    log_exception,
    report_memory_usage,
    setup_logging,
)
from src.token_utils import (
    get_answer_token_ids,
    is_valid_token_prediction,
    tokenizer_cache,
)

# Default memory threshold values
memory_threshold_mb = 10
include_memory_details = False
warning_threshold_mb = 2000
critical_threshold_mb = 3500
gc_threshold_mb = 1000


class StudyConfig(TypedDict, total=False):
    """DCBS evaluation configuration."""

    model_path: str
    model_name: str
    clusters: int
    top_n: int
    sweep_top_n: List[int]
    k: List[int]
    p_values: List[float]
    benchmark: str
    output_file: str
    dcbs_params: Dict[str, Any]  # Generic dict instead of DCBSConfig
    cache: Dict[str, Any]  # Generic dict instead of CacheConfig
    multitoken_strategy: str
    log_file: str
    log_level: str


def load_config(config_path: str) -> StudyConfig:
    """Load DCBS configuration from YAML.

    Args:
        config_path: Path to config file

    Returns:
        Config dictionary

    Raises:
        ConfigurationError: If config cannot be loaded
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
    except FileNotFoundError:
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError:
        raise ConfigurationError(f"Invalid YAML in configuration file: {config_path}")
    except Exception as e:
        raise ConfigurationError(f"Error loading config: {e}")


def top_p_sampling(
    logits: torch.Tensor, p: float = 0.9, filter_tokens: Optional[Set[int]] = None
) -> int:
    """Top-p (nucleus) sampling.

    Args:
        logits: Token logits
        p: Probability threshold
        filter_tokens: Allowed tokens

    Returns:
        Sampled token ID
    """
    if filter_tokens is not None:
        if len(filter_tokens) == 0:
            return logits.argmax().item()

        filter_mask = torch.zeros_like(logits, dtype=torch.bool)
        filter_mask[list(filter_tokens)] = True
        filtered_logits = logits.masked_fill(~filter_mask, float("-inf"))
        logits = filtered_logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    filtered_logits = logits.clone()
    filtered_logits[indices_to_remove] = float("-inf")

    token_id = torch.multinomial(torch.softmax(filtered_logits, dim=-1), 1).item()
    return token_id


def write_row(
    csvwriter,
    method: str,
    prompt_id: str,
    correct: bool,
    elapsed_ms: float,
    top_n: int,
    k: int,
    p: float,
    answer_probs: Optional[Dict] = None,
) -> None:
    """Write result row to CSV.

    Args:
        csvwriter: CSV writer
        method: Sampling method
        prompt_id: Example ID
        correct: If prediction correct
        elapsed_ms: Processing time (ms)
        top_n: Top tokens count
        k: Clusters (DCBS)
        p: Threshold (top-p)
        answer_probs: Answer probabilities
    """
    row_data = {
        "method": method,
        "prompt_id": prompt_id,
        "correct": int(correct),
        "elapsed_ms": round(elapsed_ms, 2),
        "top_n": top_n,
        "k": k,
        "p": p,
    }

    if answer_probs is not None:
        row_data["answer_probs"] = json.dumps(answer_probs)

    csvwriter.writerow(row_data)


def handle_multitoken_answer(
    tokens: List[int], tokenizer, choice_text: str, strategy: str = "first"
) -> Union[int, List[int]]:
    """Handle multi-token answers.

    Args:
        tokens: Token IDs for answer
        tokenizer: Tokenizer used
        choice_text: Original answer text
        strategy: 'first', 'most_likely', or 'combine'

    Returns:
        Token ID or list depending on strategy

    Raises:
        EvaluationError: If invalid strategy
    """
    # Log tokenization details at debug level
    logger.debug(
        f"Answer '{choice_text}' tokenization: '{tokenizer.decode(tokens)}' -> {tokens}"
    )

    if len(tokens) == 1:
        return tokens[0]

    logger.warning(f"Answer '{choice_text}' encodes to multiple tokens: {tokens}")

    if strategy == "first":
        # Use the first token
        logger.debug(
            f"Using first token strategy for '{choice_text}': {tokens[0]} -> '{tokenizer.decode([tokens[0]])}'"
        )
        return tokens[0]
    elif strategy == "most_likely":
        # This would use logits to determine the most likely token, but since we don't have
        # logits at this point, we'll use the first token as a fallback
        logger.debug(
            f"Using most_likely strategy (fallback to first) for '{choice_text}': {tokens[0]}"
        )
        return tokens[0]
    elif strategy == "combine":
        # Return all tokens for combined handling during validation
        logger.debug(
            f"Using combine strategy for '{choice_text}': keeping all {len(tokens)} tokens"
        )
        return tokens
    else:
        raise EvaluationError(
            f"Invalid multi-token strategy: {strategy}",
            details={
                "valid_strategies": ["first", "most_likely", "combine"],
                "tokens": tokens,
                "choice_text": choice_text,
            },
        )


def process_example(
    example: Dict,
    model,
    tokenizer,
    device: torch.device,
    inject_reasoning: bool = True,
    multitoken_strategy: str = "first",
) -> Dict:
    """Process benchmark example.

    Args:
        example: Example data
        model: LM model
        tokenizer: Tokenizer
        device: Compute device
        inject_reasoning: Use chain-of-thought
        multitoken_strategy: Multi-token handling

    Returns:
        Processed example data

    Raises:
        EvaluationError: Malformed example
        DataError: Invalid data structure
    """
    prompt_id = example.get("id", "unknown_id")

    # Validate example has required fields
    if not ("sentence" in example):
        raise DataError(f"Example {prompt_id} missing 'sentence' field")
    if not (("option1" in example and "option2" in example) or "answer" in example):
        raise DataError(f"Example {prompt_id} missing answer options")

    result = {"prompt_id": prompt_id}

    try:
        # Process choices
        if "option1" in example and "option2" in example:
            choices = [example["option1"], example["option2"]]
            result["choices"] = choices

            correct_option = example.get("correct_option", "1")
            if correct_option not in ["1", "2"]:
                raise DataError(
                    f"Invalid correct_option '{correct_option}' in example {prompt_id}",
                    details={"valid_options": ["1", "2"]},
                )

            correct_answer = example[f"option{correct_option}"].lower().strip()
            result["correct_answer"] = correct_answer

            # Get token IDs for answer options
            answer_ids = {}
            normalized_answer_map = {}  # Map of lowercase answers to original case

            for choice in choices:
                token_ids = get_answer_token_ids(choice, tokenizer)

                # Log detailed tokenization for debugging
                logger.debug(
                    f"Choice '{choice}' tokenized as: {token_ids} -> '{tokenizer.decode(token_ids)}'"
                )

                answer_id = handle_multitoken_answer(
                    token_ids, tokenizer, choice, multitoken_strategy
                )
                answer_ids[choice] = answer_id
                # Store normalized mapping for case-insensitive lookup
                normalized_answer_map[choice.lower().strip()] = choice

            result["answer_ids"] = answer_ids
            result["normalized_answer_map"] = normalized_answer_map
        else:
            if "answer" in example:
                answer = example["answer"].lower().strip()
                token_ids = get_answer_token_ids(answer, tokenizer)
                answer_id = handle_multitoken_answer(
                    token_ids, tokenizer, answer, multitoken_strategy
                )
                answer_ids = {answer: answer_id}
                normalized_answer_map = {answer.lower().strip(): answer}

                correct_answer = answer
                result["answer_ids"] = answer_ids
                result["normalized_answer_map"] = normalized_answer_map
                result["correct_answer"] = correct_answer
            else:
                raise DataError(f"Example {prompt_id} has no answer options")

        # Create prompt with reasoning injection if requested
        prompt = example["sentence"]
        if inject_reasoning:
            prompt += "\n\nLet's think step by step to determine the answer.\n\n"
        prompt += "The answer is "

        # Tokenize and get model output
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]

            # Get answer filter set and correct ID
            answer_filter_set = set(result["answer_ids"].values())

            # Get the correct answer with case normalization
            normalized_correct = correct_answer.lower().strip()
            original_case_answer = result["normalized_answer_map"].get(
                normalized_correct
            )

            if original_case_answer is None:
                raise DataError(
                    f"Could not find original case version for answer '{correct_answer}'",
                    details={
                        "example_id": prompt_id,
                        "normalized_map": result["normalized_answer_map"],
                        "normalized_correct": normalized_correct,
                        "correct_answer": correct_answer,
                        "all_choices": list(result["answer_ids"].keys()),
                    },
                )

            correct_id = result["answer_ids"].get(original_case_answer)

            # Verify we have a valid correct_id
            if correct_id is None:
                # Log detailed information about all available answers
                choice_details = {
                    choice: {
                        "id": id_val,
                        "decoded": (
                            tokenizer.decode([id_val])
                            if not isinstance(id_val, list)
                            else tokenizer.decode(id_val)
                        ),
                        "normalized": choice.lower().strip(),
                    }
                    for choice, id_val in result["answer_ids"].items()
                }

                raise DataError(
                    f"Could not determine correct token ID for answer '{correct_answer}'",
                    details={
                        "example_id": prompt_id,
                        "answer_ids": result["answer_ids"],
                        "normalized_map": result["normalized_answer_map"],
                        "choice_details": choice_details,
                        "multitoken_strategy": multitoken_strategy,
                        "correct_option": example.get("correct_option", "unknown"),
                    },
                )

            result["correct_id"] = correct_id
            result["filter_tokens"] = answer_filter_set

            # Calculate answer probabilities
            all_probs = torch.softmax(logits[0], dim=0)
            answer_probs = {
                choice: all_probs[id].item()
                for choice, id in result["answer_ids"].items()
            }
            result["answer_probs"] = answer_probs
            result["logits"] = logits

        return result

    except (KeyError, IndexError) as e:
        raise DataError(
            f"Structural error in example {prompt_id}: {str(e)}",
            details={"example": example},
        )


def evaluate_methods(
    result: Dict,
    model,
    top_n: int,
    k: int,
    p: float,
    dcbs_config: Optional[Dict[str, Any]] = None,
    cache_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict]:
    """Compare sampling methods: greedy, top-p, DCBS, random.

    Args:
        result: Processed example data
        model: LM model
        top_n: Top tokens for DCBS
        k: Clusters for DCBS
        p: Threshold for top-p
        dcbs_config: DCBS params
        cache_config: Cache config

    Returns:
        Results for each method
    """
    method_results = {}
    logits = result["logits"]
    filter_tokens = result["filter_tokens"]
    correct_id = result["correct_id"]
    correct_answer = result["correct_answer"]
    tokenizer = model.config.tokenizer

    # Track memory usage
    initial_memory = report_memory_usage(
        "evaluate_methods_start",
        logger,
        threshold_mb=memory_threshold_mb,
        include_details=include_memory_details,
        warning_threshold_mb=warning_threshold_mb,
        critical_threshold_mb=critical_threshold_mb,
        gc_threshold_mb=gc_threshold_mb,
    )

    # Evaluate each method
    for method in ["greedy", "top-p", "dcbs", "random"]:
        start_time = time.time()

        if method == "greedy":
            if filter_tokens:
                masked_logits = logits.clone()
                mask = torch.ones_like(masked_logits[0], dtype=torch.bool)
                mask[list(filter_tokens)] = False
                masked_logits[0, mask] = float("-inf")
                pred_id = masked_logits[0].argmax().item()
            else:
                pred_id = logits[0].argmax().item()

        elif method == "top-p":
            pred_id = top_p_sampling(logits[0], p=p, filter_tokens=filter_tokens)

        elif method == "dcbs":
            # Create DCBS sampler with canonical implementation
            cache_config_dict = cache_config if cache_config else {}
            sampler = DCBSSampler.create_default(
                k=k, top_n=top_n, cache_config=cache_config_dict
            )

            # Create sampling context
            context = SamplingContext(
                embedding_layer=model.get_input_embeddings(),
                tokenizer=tokenizer,
                device=logits[0].device,
            )

            # Use canonical DCBS sampler
            pred_id = sampler.sample(
                logits[0],
                context,
                filter_tokens=filter_tokens,
            )

        elif method == "random":
            if filter_tokens:
                pred_id = random.choice(list(filter_tokens))
            else:
                pred_id = random.randint(0, logits.shape[-1] - 1)

        elapsed_ms = (time.time() - start_time) * 1000
        correct = is_valid_token_prediction(
            pred_id, correct_id, correct_answer, tokenizer
        )

        method_results[method] = {
            "pred_id": pred_id,
            "correct": correct,
            "elapsed_ms": elapsed_ms,
        }

    # Report memory usage if significant change
    final_memory = report_memory_usage(
        "evaluate_methods_end",
        logger,
        threshold_mb=memory_threshold_mb,
        include_details=include_memory_details,
        warning_threshold_mb=warning_threshold_mb,
        critical_threshold_mb=critical_threshold_mb,
        gc_threshold_mb=gc_threshold_mb,
    )
    mem_diff = final_memory - initial_memory
    if mem_diff > memory_threshold_mb:  # Only log significant memory changes
        logger.debug(f"Memory usage increased by {mem_diff:.2f}MB during evaluation")

    return method_results


def main(args):
    """Run DCBS evaluation pipeline."""
    try:
        # Setup logging based on command-line args
        log_level = args.log_level if hasattr(args, "log_level") else "INFO"
        log_file = args.log_file if hasattr(args, "log_file") else None
        setup_logging(log_level, log_file)

        # Ensure args has all necessary attributes
        if not hasattr(args, "model_name"):
            args.model_name = None

        if not hasattr(args, "benchmark"):
            args.benchmark = None

        if not hasattr(args, "out_csv"):
            args.out_csv = None

        if not hasattr(args, "inject_reasoning"):
            args.inject_reasoning = False

        logger.info(f"Starting DCBS evaluation with args: {args}")
        config = load_config(args.config)

        # Load logging configuration from config file if not specified in args
        if not log_file and "log_file" in config:
            log_file = config["log_file"]

        # Get component-specific logging configuration
        component_log_config = None
        if "logging" in config and "components" in config["logging"]:
            component_log_config = config["logging"]["components"]

        # Configure logging with component-specific settings
        setup_logging(log_level, log_file, component_log_config)
        logger.info(
            f"Logging configured with level {log_level}"
            + (f" to file: {log_file}" if log_file else "")
        )

        # Extract configuration parameters
        model_name = (
            args.model_name
            if args.model_name
            else config.get("model_name", config.get("model_path"))
        )

        top_n_values = config.get("sweep_top_n", [config.get("top_n", 20)])
        k_values = config.get("k", [config.get("clusters", 8)])
        p_values = config.get("p_values", [0.9])
        benchmark_path = args.benchmark if args.benchmark else config.get("benchmark")
        output_file = args.out_csv if args.out_csv else config.get("output_file")
        limit = (
            args.limit if hasattr(args, "limit") and args.limit is not None else None
        )

        # Get DCBS configuration from config file
        dcbs_config = config.get("dcbs_params", None)
        cache_config = config.get("cache", None)

        # Set multitoken handling strategy
        multitoken_strategy = config.get("multitoken_strategy", "first")

        # Get memory reporting configuration
        memory_config = config.get("memory", {})
        memory_threshold_mb = memory_config.get("report_threshold_mb", 10)
        include_memory_details = memory_config.get("include_details", False)
        warning_threshold_mb = memory_config.get("warning_threshold_mb", 2000)
        critical_threshold_mb = memory_config.get("critical_threshold_mb", 3500)
        gc_threshold_mb = memory_config.get("gc_threshold_mb", 1000)

        # Configure batch processing
        batch_size = memory_config.get(
            "batch_size", 5
        )  # Process examples in batches to limit memory growth

        # Initialize memory profiler if enabled
        profiling_config = memory_config.get("profiling", {})
        memory_profiler = MemoryProfiler(
            enabled=profiling_config.get("enabled", False),
            sampling_interval_ms=profiling_config.get("sampling_interval_ms", 1000),
            trace_allocations=profiling_config.get("trace_allocations", False),
            record_peak_for=profiling_config.get("record_peak_for", []),
            logger_instance=logger,
        )

        # Configure tokenizer cache from config
        tokenizer_cache_config = config.get("tokenizer_cache", {})
        tokenizer_cache.max_size = tokenizer_cache_config.get("max_size", 5000)
        tokenizer_cache.report_interval = tokenizer_cache_config.get(
            "report_interval_sec", 60
        )

        if not benchmark_path:
            raise ConfigurationError("No benchmark file specified")

        if not output_file:
            raise ConfigurationError("No output file specified")

        if not model_name:
            raise ConfigurationError("No model specified in config or arguments")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        logger.info(f"Loading model: {model_name}")
        try:
            # Track model loading memory usage
            memory_profiler.start_operation("model_loading")
            model, tokenizer = load_model_and_tokenizer(model_name)
            model.config.tokenizer = (
                tokenizer  # Attach tokenizer to model config for convenience
            )
            device = next(model.parameters()).device
            memory_profiler.end_operation("model_loading")
            logger.info(f"Model loaded on device: {device}")
        except Exception as e:
            memory_profiler.end_operation("model_loading")
            raise ConfigurationError(
                f"Failed to load model {model_name}", details={"error": str(e)}
            )

        logger.info(f"Loading benchmark from: {benchmark_path}")
        try:
            with open(benchmark_path, "r") as f:
                benchmark = json.load(f)
        except FileNotFoundError:
            raise DataError(f"Benchmark file not found: {benchmark_path}")
        except json.JSONDecodeError:
            raise DataError(f"Invalid JSON in benchmark file: {benchmark_path}")
        except Exception as e:
            raise DataError(
                f"Error loading benchmark data: {str(e)}",
                details={"path": benchmark_path},
            )

        if limit is not None:
            benchmark = benchmark[:limit]
            logger.info(f"Limited evaluation to {limit} examples")

        logger.info(f"Benchmark loaded with {len(benchmark)} examples")
        total_examples = len(benchmark)

        # Track initial memory usage
        report_memory_usage(
            "initial",
            logger,
            threshold_mb=memory_threshold_mb,
            include_details=include_memory_details,
        )

        with open(output_file, "w", newline="") as csvfile:
            fieldnames = [
                "method",
                "prompt_id",
                "correct",
                "elapsed_ms",
                "top_n",
                "k",
                "p",
                "answer_probs",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            evaluation_stats = {
                "total": total_examples,
                "success": 0,
                "errors": 0,
                "correct": {
                    "greedy": 0,
                    "top-p": 0,
                    "dcbs": 0,
                    "random": 0,
                },
                "avg_time_ms": {
                    "greedy": 0.0,
                    "top-p": 0.0,
                    "dcbs": 0.0,
                    "random": 0.0,
                },
            }

            for top_n in top_n_values:
                logger.info(f"Running with top_n={top_n}")
                for k in k_values:
                    for p in p_values:
                        logger.info(f"  Evaluating with k={k}, p={p}")

                        success_count = 0
                        error_count = 0

                        # Process examples in batches to manage memory usage
                        for batch_start in range(0, len(benchmark), batch_size):
                            batch_end = min(batch_start + batch_size, len(benchmark))
                            batch_num = batch_start // batch_size + 1

                            # Check memory usage before starting the batch
                            memory_mb = report_memory_usage(
                                f"before_batch_{batch_num}",
                                logger,
                                threshold_mb=memory_threshold_mb,
                                include_details=include_memory_details,
                                warning_threshold_mb=warning_threshold_mb,
                                critical_threshold_mb=critical_threshold_mb,
                                gc_threshold_mb=gc_threshold_mb,
                            )

                            logger.debug(
                                f"Processing batch {batch_num}, examples {batch_start+1}-{batch_end} of {len(benchmark)}"
                            )

                            batch = benchmark[batch_start:batch_end]
                            for idx_in_batch, example in enumerate(batch):
                                idx = (
                                    batch_start + idx_in_batch
                                )  # Global index in the benchmark

                                try:
                                    # Process example
                                    memory_profiler.start_operation("tokenization")
                                    result = process_example(
                                        example,
                                        model,
                                        tokenizer,
                                        device,
                                        inject_reasoning=args.inject_reasoning,
                                        multitoken_strategy=multitoken_strategy,
                                    )
                                    memory_profiler.end_operation("tokenization")

                                    # Evaluate methods
                                    memory_profiler.start_operation("sampling")
                                    method_results = evaluate_methods(
                                        result,
                                        model,
                                        top_n,
                                        k,
                                        p,
                                        dcbs_config=dcbs_config,
                                        cache_config=cache_config,
                                    )
                                    memory_profiler.end_operation("sampling")

                                    # Update statistics
                                    for method, res in method_results.items():
                                        if res["correct"]:
                                            evaluation_stats["correct"][method] += 1
                                        evaluation_stats["avg_time_ms"][method] += res[
                                            "elapsed_ms"
                                        ]

                                        # Write results to CSV
                                        write_row(
                                            writer,
                                            method,
                                            result["prompt_id"],
                                            res["correct"],
                                            res["elapsed_ms"],
                                            top_n,
                                            k,
                                            p,
                                            result["answer_probs"],
                                        )

                                    success_count += 1
                                    evaluation_stats["success"] += 1

                                except (
                                    EvaluationError,
                                    DataError,
                                    ConfigurationError,
                                ) as e:
                                    log_exception(e, logger, log_traceback=False)
                                    error_count += 1
                                    evaluation_stats["errors"] += 1
                                    continue
                                except Exception as e:
                                    log_exception(e, logger, log_traceback=True)
                                    error_count += 1
                                    evaluation_stats["errors"] += 1
                                    continue

                                if (idx + 1) % 10 == 0:
                                    # Report progress and memory usage
                                    memory_mb = report_memory_usage(
                                        f"processed_{idx+1}",
                                        logger,
                                        threshold_mb=memory_threshold_mb,
                                        include_details=include_memory_details,
                                        warning_threshold_mb=warning_threshold_mb,
                                        critical_threshold_mb=critical_threshold_mb,
                                        gc_threshold_mb=gc_threshold_mb,
                                    )
                                    logger.info(
                                        f"    Processed {idx + 1}/{len(benchmark)} examples "
                                        f"(memory: {memory_mb:.1f}MB)"
                                    )

                            # After each batch, run garbage collection to free memory
                            report_memory_usage(
                                f"batch_completed_{batch_num}",
                                logger,
                                threshold_mb=0,  # Always report after batch
                                include_details=include_memory_details,
                                warning_threshold_mb=warning_threshold_mb,
                                critical_threshold_mb=critical_threshold_mb,
                                gc_threshold_mb=0,  # Always run GC after batch
                            )

                        # Calculate method accuracies for this parameter combination
                        accuracies = {}
                        for method in ["greedy", "top-p", "dcbs", "random"]:
                            avg_time = evaluation_stats["avg_time_ms"][method] / (
                                success_count or 1
                            )
                            accuracies[method] = (
                                evaluation_stats["correct"][method]
                                / (success_count or 1)
                                * 100
                            )

                            # Reset method-specific stats for next parameter combination
                            evaluation_stats["correct"][method] = 0
                            evaluation_stats["avg_time_ms"][method] = 0.0

                        logger.info(
                            f"  Completed evaluation with parameters: "
                            f"top_n={top_n}, k={k}, p={p}"
                        )
                        logger.info(
                            f"  Results: {success_count} successful, {error_count} errors"
                        )
                        logger.info(
                            f"  Accuracies: greedy={accuracies['greedy']:.1f}%, "
                            f"top-p={accuracies['top-p']:.1f}%, "
                            f"dcbs={accuracies['dcbs']:.1f}%, "
                            f"random={accuracies['random']:.1f}%"
                        )

                        # Report memory usage
                        memory_mb = report_memory_usage(
                            "parameter_completion",
                            logger,
                            threshold_mb=memory_threshold_mb,
                            include_details=include_memory_details,
                        )

        logger.info(f"Evaluation complete. Results saved to {output_file}")
        logger.info(
            f"Final statistics: {evaluation_stats['success']} successful, "
            f"{evaluation_stats['errors']} errors out of {total_examples} examples"
        )

    except (EvaluationError, DataError, ConfigurationError) as e:
        log_exception(e, logger, log_traceback=False)
        return 1
    except Exception as e:
        log_exception(e, logger, log_traceback=True)
        return 1

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation of sampling methods")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--model_name", type=str, help="Model name/path (overrides config)"
    )
    parser.add_argument(
        "--benchmark", type=str, help="Path to benchmark file (overrides config)"
    )
    parser.add_argument(
        "--out_csv", type=str, help="Path to output CSV file (overrides config)"
    )
    parser.add_argument(
        "--inject_reasoning",
        action="store_true",
        help="Inject chain-of-thought prompting (enabled by default)",
    )
    parser.add_argument(
        "--limit", type=int, help="Limit evaluation to this many examples"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        help="Path to log file (if not specified, logs to console only)",
    )

    args = parser.parse_args()

    exit_code = main(args)
    exit(exit_code)
