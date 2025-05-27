"""
Evaluation runner that coordinates the complete evaluation process.

This module provides the main EvaluationRunner class that orchestrates
model loading, example processing, and result aggregation.
"""

import time
from typing import Dict, List, Tuple

import numpy as np

from src.errors import eval_logger as logger
from src.evaluation_core.config import EvaluationConfig
from src.evaluation_core.model_manager import ModelManager
from src.evaluation_core.example_processor import ExampleProcessor
from src.evaluation_core.sampler_factory import SamplerFactory


def calculate_confidence_interval(correct: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate binomial confidence interval for accuracy."""
    if total == 0:
        return (0.0, 0.0)

    p = correct / total
    z = 1.96  # 95% confidence

    # Wilson score interval - more accurate than normal approximation
    n = total
    z_squared = z * z

    center = (p + z_squared / (2 * n)) / (1 + z_squared / n)
    margin = (
        z * np.sqrt((p * (1 - p) + z_squared / (4 * n)) / n) / (1 + z_squared / n)
    )

    return (max(0, center - margin) * 100, min(1, center + margin) * 100)


class EvaluationRunner:
    """Main evaluation runner that coordinates the evaluation process."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model_manager = ModelManager(config.model_name, config.load_in_4bit)
        self.samplers = SamplerFactory.create_samplers(config)

    def run_evaluation(self, benchmark_data: List[Dict]) -> Dict:
        """Run evaluation on all methods."""
        # Load model
        try:
            model, tokenizer, context = self.model_manager.load_model()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Try using improved model manager as fallback
            try:
                from .improved_model_manager import ImprovedModelManager
                improved_manager = ImprovedModelManager(self.config.model_name, self.config.load_in_4bit)
                model, tokenizer, context = improved_manager.load_model()
                logger.info("Successfully loaded model using improved manager")
            except Exception as e2:
                logger.error(f"Both model managers failed: {e2}")
                raise e

        # Limit data if requested
        if self.config.limit:
            benchmark_data = benchmark_data[: self.config.limit]
            logger.info(f"Limited evaluation to {self.config.limit} examples")

        logger.info(f"Starting evaluation on {len(benchmark_data)} examples")

        all_results = []
        method_stats = {
            name: {"correct": 0, "total": 0, "times": []}
            for name in self.samplers.keys()
        }

        # Process each example
        for i, example in enumerate(benchmark_data):
            if (i + 1) % 10 == 0:
                logger.info(f"Processing example {i + 1}/{len(benchmark_data)}")

            try:
                # Process each method for this example
                for sampler_name, sampler in self.samplers.items():
                    logger.debug(
                        f"Processing example {example.get('id', i)} with {sampler_name}"
                    )

                    # Create processor for this sampler
                    processor = ExampleProcessor(model, tokenizer, context, sampler)

                    start_time = time.time()

                    # Process example
                    result = processor.process_example(
                        example, include_cot=self.config.include_cot
                    )

                    elapsed_ms = (time.time() - start_time) * 1000

                    # Check correctness using the sampler
                    logits = result["logits"]
                    filter_tokens = result["filter_tokens"]
                    correct_id = result["correct_id"]

                    # Sample using the specified sampler
                    if hasattr(sampler, 'sample'):
                        if 'DCBSSampler' in str(type(sampler)):
                            pred_id = sampler.sample(logits, filter_tokens=filter_tokens, context=context)
                        else:
                            pred_id = sampler.sample(logits, filter_tokens=filter_tokens)
                    else:
                        pred_id = correct_id  # Fallback

                    correct = pred_id == correct_id

                    # Combine all results
                    final_result = {
                        **result,
                        "sampler": sampler_name,
                        "pred_id": pred_id,
                        "correct": correct,
                        "elapsed_ms": elapsed_ms,
                    }

                    # Remove logits for JSON serialization
                    final_result.pop("logits", None)
                    all_results.append(final_result)

                    # Update statistics
                    stats = method_stats[sampler_name]
                    stats["total"] += 1
                    if correct:
                        stats["correct"] += 1
                    stats["times"].append(elapsed_ms)

            except Exception as e:
                logger.error(f"Error processing example {i}: {e}")
                continue

        # Calculate final statistics
        statistics = {}
        for sampler_name, stats in method_stats.items():
            if stats["total"] > 0:
                accuracy = stats["correct"] / stats["total"] * 100
                avg_time = sum(stats["times"]) / len(stats["times"]) if stats["times"] else 0.0

                # Calculate confidence interval
                confidence_interval = calculate_confidence_interval(
                    stats["correct"], stats["total"]
                )

                statistics[sampler_name] = {
                    "accuracy": accuracy,
                    "correct": stats["correct"],
                    "total": stats["total"],
                    "avg_time_ms": avg_time,
                    "confidence_interval": confidence_interval,
                }

        # Create final results
        results = {
            "statistics": statistics,
            "config": {
                "model": self.config.model_name,
                "total_examples": len(benchmark_data),
                "methods": list(self.samplers.keys()),
                "include_cot": self.config.include_cot,
                "enable_caching": self.config.enable_caching,
            },
            "detailed_results": all_results,
            "evaluation_completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        logger.info("Evaluation completed successfully!")
        return results 