"""
Evaluation runner with fixed conversation flow and parameter handling.

This module provides the main evaluation logic using components
that implement proper conversation flow and remove ChatTemplateManager dependency.
"""

import time
import traceback
from typing import Dict, List

from src.errors import eval_logger as logger
from src.evaluation_core.config import EvaluationConfig
from src.evaluation_core.model_manager import ModelManager
from src.evaluation_core.example_processor import ExampleProcessor
from src.evaluation_core.sampler_factory import SamplerFactory


def calculate_confidence_interval(correct: int, total: int) -> tuple:
    """Calculate binomial confidence interval for accuracy."""
    import numpy as np
    from scipy import stats
    
    if total == 0:
        return (0.0, 0.0)
    
    # Wilson score interval (more accurate than normal approximation)
    p = correct / total
    z = 1.96  # 95% confidence
    
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    
    return (max(0, center - margin) * 100, min(100, center + margin) * 100)


class EvaluationRunner:
    """Evaluation runner with proper conversation flow."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model_manager = ModelManager(config.model_name, config.load_in_4bit)
        
    def run_evaluation(self, benchmark_data: List[Dict]) -> Dict:
        """Run evaluation using proper conversation flow."""
        # Load model
        model, tokenizer, context = self.model_manager.load_model()
        
        # Create samplers with context and clustering parameters
        self.samplers = SamplerFactory.create_samplers(
            self.config, 
            context,
            clustering_method=self.config.clustering_method,
            dbscan_eps=self.config.dbscan_eps,
            dbscan_min_samples=self.config.dbscan_min_samples,
            hierarchical_linkage=self.config.hierarchical_linkage
        )
        
        # Create processor
        processor = ExampleProcessor(model, tokenizer, context)
        
        # Limit data if requested
        if self.config.limit:
            benchmark_data = benchmark_data[: self.config.limit]
            logger.info(f"Limited evaluation to {self.config.limit} examples")

        logger.info(f"Starting evaluation on {len(benchmark_data)} examples")
        logger.info(f"Using samplers: {list(self.samplers.keys())}")
        if hasattr(self.config, 'clustering_method'):
            logger.info(f"DCBS clustering method: {self.config.clustering_method}")

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
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
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
                "clustering_method": getattr(self.config, 'clustering_method', 'dbscan'),
            },
            "detailed_results": all_results,
            "evaluation_completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        logger.info("Evaluation completed successfully!")
        return results 