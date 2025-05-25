"""
Core evaluation logic shared across different evaluation scripts.

This module provides common functionality for model loading, evaluation,
and result processing to eliminate code duplication.
"""

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dcbs import DCBSSampler, GreedySampler, RandomSampler, SamplingContext, TopPSampler
from src.errors import (
    DataError,
    EvaluationError,
)
from src.errors import eval_logger as logger
from src.errors import (
    setup_logging,
)
from src.token_utils import get_answer_token_ids, is_valid_token_prediction


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""

    model_name: str
    benchmark_path: str
    output_dir: str
    limit: Optional[int] = None
    top_p: float = 0.9
    k: int = 8
    top_n: int = 50
    include_cot: bool = True
    log_level: str = "INFO"
    load_in_4bit: bool = False
    enable_caching: bool = True  # NEW: Control DCBS caching


@dataclass
class EvaluationResult:
    """Results from evaluating a single example with a sampler."""

    example_id: str
    method: str
    correct: bool
    elapsed_ms: float
    pred_id: int
    predicted_answer: str
    cot_reasoning: Optional[str] = None
    answer_probs: Optional[Dict[str, float]] = None


class ModelManager:
    """Handles model loading and management separately from evaluation logic."""

    def __init__(self, model_name: str, load_in_4bit: bool = False):
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        self.device = None
        self.context = None

    def load_model(self) -> Tuple[object, object, SamplingContext]:
        logger.info(f"Loading model: {self.model_name}")

        # Set dtype based on CUDA support
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

        model_kwargs = {
            "token": os.environ.get("HF_HUB_TOKEN"),
            "device_map": "auto",
            "torch_dtype": dtype,
        }

        # Add 4-bit quantization if requested
        if self.load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs
        )

        # CRITICAL: Set model to evaluation mode for inference
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, token=os.environ.get("HF_HUB_TOKEN")
        )

        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = next(self.model.parameters()).device

        # Create sampling context
        self.context = SamplingContext(
            embedding_layer=self.model.get_input_embeddings(),
            tokenizer=self.tokenizer,
            device=self.device,
        )

        logger.info(f"Model loaded on device: {self.device}")
        logger.info(f"Model dtype: {next(self.model.parameters()).dtype}")
        return self.model, self.tokenizer, self.context


class ChatTemplateManager:
    """Manages chat templates for different model families."""

    @classmethod
    def setup_chat_template(cls, tokenizer, model_name: str) -> None:
        # Check if model already has a template
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            logger.info("Using existing chat template")
            return

        # Use the chat_templates module for proper template handling
        try:
            from chat_templates import get_template_for_model

            tokenizer.chat_template = get_template_for_model(
                tokenizer.name_or_path or model_name
            )
            logger.info(f"Applied chat template for model: {model_name}")
        except ImportError:
            # Fallback to simple Llama 3 template if chat_templates module not available
            if "llama" in model_name.lower() and "3" in model_name:
                template = (
                    "{% if messages[0]['role'] == 'system' %}"
                    "{% set loop_messages = messages[1:] %}"
                    "{% set system_message = messages[0]['content'] %}"
                    "{% else %}"
                    "{% set loop_messages = messages %}"
                    "{% set system_message = false %}"
                    "{% endif %}"
                    "{% for message in loop_messages %}"
                    "{% if loop.index0 == 0 and system_message %}"
                    "{{ '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n' + system_message + '<|eot_id|>' }}"
                    "{% endif %}"
                    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
                    "{% if loop.last and add_generation_prompt %}"
                    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
                    "{% endif %}"
                    "{% endfor %}"
                )
                tokenizer.chat_template = template
                logger.info("Applied fallback Llama 3 chat template")
            else:
                # Generic fallback
                template = (
                    "{% for message in messages %}"
                    "{{ message['role'].title() + ': ' + message['content'] + '\\n' }}"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}"
                    "{{ 'Assistant: ' }}"
                    "{% endif %}"
                )
                tokenizer.chat_template = template
                logger.info("Applied generic fallback chat template")

    @staticmethod
    def validate_template(tokenizer, model_name: str) -> bool:
        try:
            test_messages = [{"role": "user", "content": "Test message"}]
            result = tokenizer.apply_chat_template(
                test_messages, tokenize=False, add_generation_prompt=True
            )
            return len(result) > 0
        except Exception as e:
            logger.warning(f"Chat template validation failed: {e}")
            return False


class SamplerFactory:
    """Factory for creating and managing sampler instances."""

    @staticmethod
    def create_samplers(config: EvaluationConfig) -> Dict[str, object]:
        return {
            "greedy": GreedySampler(),
            "top-p": TopPSampler(p=config.top_p),
            "dcbs": DCBSSampler.create_default(
                k=config.k, top_n=config.top_n, enable_caching=config.enable_caching
            ),
            "random": RandomSampler(),
        }


class ExampleProcessor:
    """Processes individual examples for evaluation."""

    def __init__(self, model, tokenizer, context: SamplingContext, sampler=None):
        self.model = model
        self.tokenizer = tokenizer
        self.context = context
        self.device = context.device
        self.sampler = sampler  # For chain-of-thought generation

    def create_prompt(
        self, sentence: str, options: List[str], include_cot: bool = True
    ) -> str:
        """Create a chat-formatted prompt for the problem."""
        if include_cot:
            system_msg = "You are a helpful assistant. Think step by step and then give your final answer."

            # Build options string dynamically
            options_str = ""
            for i, option in enumerate(options):
                label = chr(ord("A") + i)  # A, B, C, D, etc.
                options_str += f"{label}. {option}\n"

            user_msg = f"{sentence}\n\nOptions:\n{options_str}\nLet's think step by step to determine the answer.\n\nThe answer is"
        else:
            system_msg = "You are a helpful assistant."

            # Build options string dynamically
            options_str = ""
            for i, option in enumerate(options):
                label = chr(ord("A") + i)  # A, B, C, D, etc.
                options_str += f"{label}. {option}\n"

            user_msg = f"{sentence}\n\nOptions:\n{options_str}\nThe answer is"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        # Use chat template if available, otherwise fall back to simple formatting
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant: "

    def get_answer_token_ids(self, options: List[str]) -> Dict[str, int]:
        """Get token IDs for answer options with robust tokenization."""
        answer_ids = {}

        for option in options:
            # Try different tokenization approaches
            candidates = [
                option,  # Raw option
                f" {option}",  # With leading space
                f"{option}",  # Clean option
            ]

            token_id = None
            for candidate in candidates:
                tokens = self.tokenizer.encode(candidate, add_special_tokens=False)
                if len(tokens) == 1:
                    token_id = tokens[0]
                    break

            if token_id is None:
                # Fall back to first token
                tokens = self.tokenizer.encode(f" {option}", add_special_tokens=False)
                token_id = tokens[0] if tokens else 0

            answer_ids[option] = token_id

        return answer_ids

    def process_example(self, example: Dict, include_cot: bool = True) -> Dict:
        """Process a single example and get logits, with actual CoT generation if enabled."""

        # Handle both Winogrande format and ARC Easy format
        if "sentence" in example:
            # Winogrande format
            sentence = example["sentence"]
            options = [example["option1"], example["option2"]]
            correct_option = example.get("correct_option", "1")
            correct_answer = example[f"option{correct_option}"]
        elif "question" in example:
            # ARC Easy format
            sentence = example["question"]
            options = example["options"]
            correct_option = example.get("correct_option", "1")
            # Convert 1-based index to 0-based and get the correct answer
            correct_idx = int(correct_option) - 1
            correct_answer = options[correct_idx]
        else:
            raise ValueError(
                "Unknown example format - expected 'sentence' or 'question' field"
            )

        if include_cot:
            # First, generate chain-of-thought reasoning
            cot_prompt = self.create_cot_prompt(sentence, options)
            cot_reasoning = self.generate_reasoning(cot_prompt)

            # Then create final answer prompt with the generated reasoning
            final_prompt = self.create_final_answer_prompt(
                sentence, options, cot_reasoning
            )
        else:
            final_prompt = self.create_prompt(sentence, options, include_cot=False)
            cot_reasoning = None

        # Tokenize and get model output for final answer
        inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :].squeeze(0)

        # Get token IDs for answer options
        answer_ids = self.get_answer_token_ids(options)

        # Calculate answer probabilities
        all_probs = torch.softmax(logits, dim=0)
        answer_probs = {
            option: all_probs[token_id].item()
            for option, token_id in answer_ids.items()
        }

        return {
            "id": example.get("id", "unknown"),
            "sentence": sentence,
            "options": options,
            "correct_answer": correct_answer,
            "correct_option": correct_option,
            "answer_ids": answer_ids,
            "filter_tokens": set(answer_ids.values()),
            "correct_id": answer_ids[correct_answer],
            "logits": logits,
            "answer_probs": answer_probs,
            "cot_reasoning": cot_reasoning,
        }

    def create_cot_prompt(self, sentence: str, options: List[str]) -> str:
        """Create a prompt for chain-of-thought reasoning generation."""
        system_msg = (
            "You are a helpful assistant. Think step by step about the problem."
        )

        # Build options string dynamically
        options_str = ""
        for i, option in enumerate(options):
            label = chr(ord("A") + i)  # A, B, C, D, etc.
            options_str += f"{label}. {option}\n"

        user_msg = f"{sentence}\n\nOptions:\n{options_str}\nLet's think step by step about which option is correct:"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant: "

    def create_final_answer_prompt(
        self, sentence: str, options: List[str], reasoning: str
    ) -> str:
        """Create final answer prompt with generated reasoning."""
        system_msg = "You are a helpful assistant. Based on your reasoning, give the final answer."

        # Build options string dynamically
        options_str = ""
        for i, option in enumerate(options):
            label = chr(ord("A") + i)  # A, B, C, D, etc.
            options_str += f"{label}. {option}\n"

        user_msg = f"{sentence}\n\nOptions:\n{options_str}\nMy reasoning: {reasoning}\n\nTherefore, the answer is"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant: "

    def generate_reasoning(self, prompt: str, max_length: int = 200) -> str:
        """Generate chain-of-thought reasoning using the assigned sampling method."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        if self.sampler is None:
            # Fallback to greedy generation
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_length,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        else:
            # Use the same sampling method being evaluated
            generated_tokens = []
            current_input = inputs.input_ids

            with torch.no_grad():
                for _ in range(max_length):
                    outputs = self.model(current_input)
                    logits = outputs.logits[:, -1, :].squeeze(0)

                    # Use the assigned sampler with appropriate interface
                    if hasattr(self.sampler, "clusterer"):  # Check if it's DCBS
                        next_token = self.sampler.sample(logits, self.context)
                    else:
                        next_token = self.sampler.sample(logits, context=self.context)

                    generated_tokens.append(next_token)

                    # Stop if EOS token
                    if next_token == self.tokenizer.eos_token_id:
                        break

                    # Append token and continue
                    current_input = torch.cat(
                        [
                            current_input,
                            torch.tensor([[next_token]], device=self.device),
                        ],
                        dim=1,
                    )

        # Extract generated reasoning
        if self.sampler is None:
            reasoning_tokens = outputs[0][inputs.input_ids.shape[1] :]
        else:
            reasoning_tokens = torch.tensor(generated_tokens, device=self.device)

        reasoning = self.tokenizer.decode(
            reasoning_tokens, skip_special_tokens=True
        ).strip()

        # Truncate at first occurrence of answer patterns to avoid giving away the answer
        stop_patterns = [
            "the answer is",
            "answer:",
            "therefore",
            "so the answer",
            "option",
        ]
        for pattern in stop_patterns:
            if pattern in reasoning.lower():
                reasoning = reasoning[: reasoning.lower().find(pattern)].strip()
                break

        return reasoning


class EvaluationRunner:
    """Main evaluation runner that coordinates the evaluation process."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model_manager = ModelManager(config.model_name, config.load_in_4bit)
        self.samplers = SamplerFactory.create_samplers(config)
        self.stats = defaultdict(lambda: {"correct": 0, "total": 0, "times": []})

    def run_evaluation(self, benchmark_data: List[Dict]) -> Dict:
        """Run evaluation on all methods."""
        # Load model
        model, tokenizer, context = self.model_manager.load_model()

        # Setup chat template
        ChatTemplateManager.setup_chat_template(tokenizer, self.config.model_name)

        # Validate template
        if not ChatTemplateManager.validate_template(tokenizer, self.config.model_name):
            logger.warning("Chat template validation failed, results may be unreliable")

        # Limit data if requested
        if self.config.limit:
            benchmark_data = benchmark_data[: self.config.limit]
            logger.info(f"Limited evaluation to {self.config.limit} examples")

        logger.info(f"Starting evaluation on {len(benchmark_data)} examples")

        all_results = []

        for i, example in enumerate(benchmark_data):
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i + 1}/{len(benchmark_data)} examples")

            try:
                # Evaluate with each sampler
                for method_name, sampler in self.samplers.items():
                    # Create processor with the specific sampler for CoT generation
                    processor = ExampleProcessor(model, tokenizer, context, sampler)

                    # Process the example
                    processed = processor.process_example(
                        example, self.config.include_cot
                    )

                    result = self._evaluate_with_sampler(
                        processed, sampler, method_name, context, tokenizer
                    )
                    all_results.append(result)

                    # Update statistics
                    stats = self.stats[method_name]
                    stats["total"] += 1
                    if result.correct:
                        stats["correct"] += 1
                    stats["times"].append(result.elapsed_ms)

            except Exception as e:
                logger.error(f"Error processing example {example.get('id', i)}: {e}")
                continue

        return self._compile_results(all_results, len(benchmark_data))

    def _evaluate_with_sampler(
        self,
        processed_example: Dict,
        sampler,
        sampler_name: str,
        context: SamplingContext,
        tokenizer,
    ) -> EvaluationResult:
        """Evaluate a processed example with a specific sampler."""
        start_time = time.time()

        logits = processed_example["logits"]
        filter_tokens = processed_example["filter_tokens"]
        correct_id = processed_example["correct_id"]
        correct_answer = processed_example["correct_answer"]

        # Sample using the specified sampler
        if sampler_name == "dcbs":
            # DCBS requires context as a required parameter
            pred_id = sampler.sample(logits, context, filter_tokens=filter_tokens)
        else:
            # Other samplers use the standard interface
            pred_id = sampler.sample(
                logits, filter_tokens=filter_tokens, context=context
            )

        elapsed_ms = (time.time() - start_time) * 1000

        # Check if prediction is correct
        correct = is_valid_token_prediction(
            pred_id, correct_id, correct_answer, tokenizer
        )

        return EvaluationResult(
            example_id=processed_example["id"],
            method=sampler_name,
            correct=correct,
            elapsed_ms=elapsed_ms,
            pred_id=pred_id,
            predicted_answer=tokenizer.decode([pred_id]).strip(),
            answer_probs=processed_example["answer_probs"],
        )

    def _compile_results(
        self, all_results: List[EvaluationResult], total_examples: int
    ) -> Dict:
        """Compile final results with statistics."""
        # Calculate final statistics
        final_stats = {}
        for method, stats in self.stats.items():
            if stats["total"] > 0:
                accuracy = (stats["correct"] / stats["total"]) * 100
                avg_time = np.mean(stats["times"]) if stats["times"] else 0
                std_time = np.std(stats["times"]) if stats["times"] else 0

                final_stats[method] = {
                    "accuracy": accuracy,
                    "correct": stats["correct"],
                    "total": stats["total"],
                    "avg_time_ms": avg_time,
                    "std_time_ms": std_time,
                    "confidence_interval": self._calculate_confidence_interval(
                        stats["correct"], stats["total"]
                    ),
                }
                logger.info(
                    f"{method}: {accuracy:.2f}% ({stats['correct']}/{stats['total']})"
                )

        return {
            "statistics": final_stats,
            "detailed_results": [
                {
                    "example_id": r.example_id,
                    "method": r.method,
                    "correct": r.correct,
                    "elapsed_ms": r.elapsed_ms,
                    "predicted_answer": r.predicted_answer,
                }
                for r in all_results
            ],
            "config": {
                "model": self.config.model_name,
                "total_examples": total_examples,
                "methods": list(self.samplers.keys()),
                "include_cot": self.config.include_cot,
            },
        }

    @staticmethod
    def _calculate_confidence_interval(
        correct: int, total: int, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate binomial confidence interval for accuracy."""
        if total == 0:
            return (0.0, 0.0)

        p = correct / total
        z = 1.96  # 95% confidence - could use scipy.stats.norm.ppf but this is faster

        # Wilson score interval - more accurate than normal approximation for small samples
        # (personal preference after getting burned by normal approx edge cases)
        n = total
        z_squared = z * z

        center = (p + z_squared / (2 * n)) / (1 + z_squared / n)
        margin = (
            z * np.sqrt((p * (1 - p) + z_squared / (4 * n)) / n) / (1 + z_squared / n)
        )

        return (max(0, center - margin) * 100, min(1, center + margin) * 100)


def load_benchmark_data(benchmark_path: str) -> List[Dict]:
    """Load benchmark data with validation."""
    logger.info(f"Loading benchmark: {benchmark_path}")

    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    try:
        with open(benchmark_path, "r") as f:
            data = json.load(f)

        # Validate data structure
        if not isinstance(data, list):
            raise DataError("Benchmark data must be a list of examples")

        if len(data) == 0:
            raise DataError("Benchmark data is empty")

        # Validate first example has required fields (support both formats)
        first_example = data[0]

        # Check for Winogrande format
        winogrande_fields = ["sentence", "option1", "option2"]
        has_winogrande = all(field in first_example for field in winogrande_fields)

        # Check for ARC Easy format
        arc_fields = ["question", "options"]
        has_arc = all(field in first_example for field in arc_fields)

        if not (has_winogrande or has_arc):
            raise DataError(
                f"Benchmark examples must contain either Winogrande fields {winogrande_fields} or ARC Easy fields {arc_fields}"
            )

        # Log which format was detected
        if has_winogrande:
            logger.info(f"Detected Winogrande format dataset")
        elif has_arc:
            logger.info(f"Detected ARC Easy format dataset")

        logger.info(f"Loaded {len(data)} examples")
        return data

    except json.JSONDecodeError as e:
        raise DataError(f"Invalid JSON in benchmark file: {e}")
    except Exception as e:
        raise DataError(f"Error loading benchmark data: {e}")
