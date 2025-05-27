"""
Improved evaluation runner with fixed conversation flow and parameter handling.

This module provides the main evaluation logic using the improved components
that implement proper conversation flow and remove ChatTemplateManager dependency.
"""

import time
from typing import Dict, List

from src.errors import eval_logger as logger
from src.evaluation_core.config import EvaluationConfig
from src.evaluation_core.improved_model_manager import ImprovedModelManager
from src.evaluation_core.improved_example_processor import ImprovedExampleProcessor
from src.evaluation_core.sampler_factory import SamplerFactory
from src.statistical_analysis import calculate_statistics


class ImprovedEvaluationRunner:
    """Improved evaluation runner with proper conversation flow."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model_manager = ImprovedModelManager(config.model_name, config.load_in_4bit)
        self.samplers = SamplerFactory.create_samplers(config)
        
    def run_evaluation(self, benchmark_data: List[Dict]) -> Dict:
        """Run evaluation using improved conversation flow."""
        # Load model
        model, tokenizer, context = self.model_manager.load_model()
        
        # Create improved processor
        processor = ImprovedExampleProcessor(model, tokenizer, context)
        
        # Limit data if requested
        if self.config.limit:
            benchmark_data = benchmark_data[: self.config.limit]
            logger.info(f"Limited evaluation to {self.config.limit} examples")

        logger.info(f"Starting evaluation on {len(benchmark_data)} examples")
        logger.info(f"Using samplers: {list(self.samplers.keys())}")

        all_results = []
        method_stats = {
            name: {"correct": 0, "total": 0, "times": []}
            for name in self.samplers.keys()
        }

        # Process examples
        for i, example in enumerate(benchmark_data):
            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"Processing example {i + 1}/{len(benchmark_data)}")

            try:
                # Process example once to get reasoning and logits
                # Use a consistent sampler for reasoning generation (greedy for reproducibility)
                reasoning_sampler = self.samplers.get("greedy", list(self.samplers.values())[0])
                
                processed_result = processor.process_example(
                    example, 
                    reasoning_sampler, 
                    include_cot=self.config.include_cot
                )

                # Now evaluate with each sampler using the same reasoning/logits
                for sampler_name, sampler in self.samplers.items():
                    eval_result = processor.evaluate_with_sampler(
                        processed_result, sampler, sampler_name
                    )

                    # Combine results
                    final_result = {**processed_result, **eval_result}
                    # Remove logits for JSON serialization
                    final_result.pop("logits", None)
                    all_results.append(final_result)

                    # Update statistics
                    stats = method_stats[sampler_name]
                    stats["total"] += 1
                    if eval_result["correct"]:
                        stats["correct"] += 1
                    stats["times"].append(eval_result["elapsed_ms"])

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
                confidence_interval = calculate_statistics(
                    stats["correct"], stats["total"]
                )["confidence_interval"]

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