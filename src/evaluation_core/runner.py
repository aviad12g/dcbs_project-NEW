"""
Main evaluation runner that coordinates the evaluation process.

This module provides the EvaluationRunner class which orchestrates
the entire evaluation pipeline.
"""

import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from dcbs import SamplingContext
from src.errors import eval_logger as logger
from src.token_utils import is_valid_token_prediction

from .config import EvaluationConfig
from .result import EvaluationResult
from .model_manager import ModelManager
from .template_manager import ChatTemplateManager
from .sampler_factory import SamplerFactory
from .example_processor import ExampleProcessor

# Progress logging interval - configurable constant
LOG_PROGRESS_INTERVAL = 50


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

        for example_index, example in enumerate(benchmark_data):
            if (example_index + 1) % LOG_PROGRESS_INTERVAL == 0:
                logger.info(f"Progress: {example_index + 1}/{len(benchmark_data)} examples")
            
            # Log every 10 examples for more granular tracking
            if (example_index + 1) % 10 == 0:
                logger.info(f"Processing example {example_index + 1}/{len(benchmark_data)}")

            try:
                # Evaluate with each sampler
                for method_name, sampler in self.samplers.items():
                    logger.debug(f"Example {example_index + 1}: Starting {method_name} evaluation")
                    
                    # Create processor with the specific sampler for CoT generation
                    processor = ExampleProcessor(model, tokenizer, context, sampler)

                    # Process the example
                    logger.debug(f"Example {example_index + 1}: Processing with {method_name}")
                    processed = processor.process_example(
                        example, self.config.include_cot
                    )
                    
                    logger.debug(f"Example {example_index + 1}: Sampling with {method_name}")
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
                    
                    logger.debug(f"Example {example_index + 1}: Completed {method_name} in {result.elapsed_ms:.1f}ms")

            except Exception as e:
                logger.error(f"Error processing example {example.get('id', example_index)}: {e}")
                continue

        return self._aggregate_results(all_results, len(benchmark_data))

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
            pred_id = sampler.sample(logits, filter_tokens=filter_tokens, context=context)
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

    def _aggregate_results(
        self, all_results: List[EvaluationResult], total_examples: int
    ) -> Dict:
        """Aggregate final results with statistics."""
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