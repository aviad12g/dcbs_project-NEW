"""
Evaluation runner with proper checkpointing and GPU optimization.

This module handles the main evaluation loop with support for:
- Proper checkpointing and resumption
- Automatic GPU batch size optimization
- Multi-GPU support
- Signal handling for graceful interruption
"""

import signal
import sys
import time
import traceback
from typing import Dict, List, Optional

from src.errors import eval_logger as logger
from src.evaluation_core.checkpoint import CheckpointManager, CheckpointState
from src.evaluation_core.config import EvaluationConfig
from src.evaluation_core.example_processor import ExampleProcessor
from src.evaluation_core.gpu_optimizer import get_gpu_optimizer
from src.evaluation_core.model_manager import ModelManager
from src.evaluation_core.sampler_factory import SamplerFactory


def calculate_confidence_interval(correct: int, total: int) -> tuple:
    """Calculate confidence interval for accuracy."""
    if total == 0:
        return (0.0, 0.0)
    
    # Wilson score interval for 95% confidence
    import math
    z = 1.96  # 95% confidence
    p = correct / total
    n = total
    
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_standard_deviation = math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    
    lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
    
    return (max(0.0, lower_bound * 100), min(100.0, upper_bound * 100))


class EvaluationRunner:
    """Main evaluation runner with checkpointing and GPU optimization."""
    
    def __init__(self, config: EvaluationConfig, requested_samplers: list = None, 
                 run_id: str = None, enable_checkpointing: bool = True):
        """Initialize evaluation runner with enhanced capabilities."""
        self.config = config
        self.requested_samplers = requested_samplers
        self.run_id = run_id or f"eval_{int(time.time())}"
        self.enable_checkpointing = enable_checkpointing
        
        # Initialize managers
        self.model_manager = ModelManager(config)
        self.checkpoint_manager = CheckpointManager() if enable_checkpointing else None
        self.gpu_optimizer = get_gpu_optimizer()
        
        # State tracking
        self.current_state = None
        self.samplers = {}
        
        # Setup signal handling for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle interruption signals gracefully."""
        logger.info("Received interruption signal, saving checkpoint...")
        if self.current_state and self.checkpoint_manager:
            self.checkpoint_manager.save_checkpoint(self.current_state)
        sys.exit(0)
        
    def try_resume_from_checkpoint(self) -> Optional[CheckpointState]:
        """Try to resume from existing checkpoint."""
        if not self.checkpoint_manager:
            return None
            
        checkpoint = self.checkpoint_manager.load_checkpoint(self.run_id)
        if checkpoint:
            logger.info(f"Resuming from checkpoint: {checkpoint.completed_examples}/{checkpoint.total_examples} examples completed")
            return checkpoint
        return None

    def run_evaluation(self, benchmark_data: List[Dict]) -> Dict:
        """Run evaluation with proper checkpointing and GPU optimization."""
        # Load model first 
        model, tokenizer, context = self.model_manager.load_model()
        
        # Try to resume from checkpoint FIRST to get optimized config
        checkpoint = self.try_resume_from_checkpoint()
        
        # If we have a checkpoint with optimized config, restore it
        if checkpoint and hasattr(checkpoint, 'config') and checkpoint.config:
            logger.info("Restoring optimized configuration from checkpoint")
            # Restore key optimization parameters from checkpoint
            checkpoint_config = checkpoint.config
            if 'batch_size' in checkpoint_config:
                self.config.batch_size = checkpoint_config['batch_size']
                logger.info(f"Restored optimized batch size: {self.config.batch_size}")
            if 'target_memory_utilization' in checkpoint_config:
                # Update GPU optimizer with restored settings
                self.gpu_optimizer.target_memory_utilization = checkpoint_config['target_memory_utilization']
        else:
            # Optimize batch size for GPU utilization (only if no checkpoint)
            if self.gpu_optimizer.available_gpus:
                optimal_batch_size = self.gpu_optimizer.get_optimal_batch_size(
                    model, tokenizer, sample_input_length=512
                )
                logger.info(f"Auto-detected optimal batch size: {optimal_batch_size}")
                # Update config with optimal batch size
                if hasattr(self.config, 'batch_size'):
                    original_batch_size = self.config.batch_size
                    self.config.batch_size = optimal_batch_size
                    logger.info(f"Updated batch size from {original_batch_size} to {optimal_batch_size}")
        
        # Create samplers with context and clustering parameters
        self.samplers = SamplerFactory.create_samplers(
            self.config,
            context,
            clustering_method=self.config.clustering_method,
            dbscan_eps=self.config.dbscan_eps,
            dbscan_min_samples=self.config.dbscan_min_samples,
            hierarchical_linkage=self.config.hierarchical_linkage,
            debug_mode=self.config.debug_mode,
            enable_cluster_history=self.config.enable_cluster_history,
            requested_samplers=self.requested_samplers,
        )
        
        # Initialize state variables
        start_idx = 0
        all_results = []
        method_stats = {
            name: {"correct": 0, "total": 0, "times": []}
            for name in self.samplers.keys()
        }
        prediction_map = {}
        
        # Limit data if requested
        if self.config.limit:
            benchmark_data = benchmark_data[: self.config.limit]
            logger.info(f"Limited evaluation to {self.config.limit} examples")

        # Apply checkpoint data if available
        if checkpoint:
            start_idx = checkpoint.completed_examples  # Use completed_examples, not current_example_idx
            all_results = checkpoint.results
            
            # Reconstruct method_stats from checkpoint results
            # Group results by example to count correctly
            processed_examples = set()
            for result in all_results:
                example_id = result.get("id")
                sampler_name = result.get("sampler")
                
                if sampler_name in method_stats:
                    # Only count each example once per sampler
                    if (example_id, sampler_name) not in processed_examples:
                        method_stats[sampler_name]["total"] += 1
                        if result.get("correct"):
                            method_stats[sampler_name]["correct"] += 1
                        method_stats[sampler_name]["times"].append(result.get("elapsed_ms", 0))
                        processed_examples.add((example_id, sampler_name))
                    
                    # Rebuild prediction map
                    if example_id not in prediction_map:
                        prediction_map[example_id] = {}
                    prediction_map[example_id][sampler_name] = result

        logger.info(f"Starting evaluation on {len(benchmark_data)} examples")
        logger.info(f"Using samplers: {list(self.samplers.keys())}")
        if hasattr(self.config, 'clustering_method'):
            logger.info(f"DCBS clustering method: {self.config.clustering_method}")

        if start_idx > 0:
            logger.info(f"Resuming from example {start_idx + 1}")

        # Create processor
        processor = ExampleProcessor(model, tokenizer, context)

        # Process examples starting from checkpoint position
        examples_since_checkpoint = 0
        utilization_history = []  # Track GPU utilization for dynamic adjustment
        
        for i, example in enumerate(benchmark_data):
            # Skip already processed examples
            if i < start_idx:
                continue
                
            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"Processing example {i + 1}/{len(benchmark_data)}")
                
                # Log GPU utilization periodically
                if self.gpu_optimizer.available_gpus and (i + 1) % 50 == 0:
                    usage_info = self.gpu_optimizer.monitor_gpu_usage()
                    for gpu_id, info in usage_info.items():
                        logger.info(f"GPU {gpu_id}: {info['utilization_percent']:.1f}% memory, "
                                  f"{info['allocated_gb']:.1f}GB/{info['total_gb']:.1f}GB used")

            try:
                # Process example once to get reasoning and logits
                # Use a consistent sampler for reasoning generation (greedy for reproducibility)
                reasoning_sampler = self.samplers.get("greedy", list(self.samplers.values())[0])
                
                processed_result = processor.process_example(
                    example, 
                    reasoning_sampler, 
                    include_cot=self.config.include_cot
                )

                # Track results for this example
                example_results = []

                # Now evaluate with each sampler using the same reasoning/logits
                for sampler_name, sampler in self.samplers.items():
                    eval_result = processor.evaluate_with_sampler(
                        processed_result, sampler, sampler_name
                    )

                    # Combine results
                    final_result = {**processed_result, **eval_result}
                    # Remove logits for JSON serialization
                    final_result.pop("logits", None)
                    
                    # Convert any sets to lists for JSON serialization
                    final_result = self._make_json_serializable(final_result)
                    example_results.append(final_result)

                    # Track predictions for difference analysis
                    ex_id = final_result.get("id")
                    if ex_id not in prediction_map:
                        prediction_map[ex_id] = {}
                    prediction_map[ex_id][sampler_name] = final_result

                    # Update statistics
                    stats = method_stats[sampler_name]
                    stats["total"] += 1
                    if eval_result["correct"]:
                        stats["correct"] += 1
                    stats["times"].append(eval_result["elapsed_ms"])

                # Add all results for this example at once (atomic operation)
                all_results.extend(example_results)

                # Update checkpoint state after processing ALL samplers for this example
                examples_since_checkpoint += 1
                completed_examples = i + 1  # Use actual example index + 1
                
                # Always update current state for signal handler
                config_dict = self.config.__dict__.copy()
                config_dict['target_memory_utilization'] = self.gpu_optimizer.target_memory_utilization
                config_dict['safety_margin'] = self.gpu_optimizer.safety_margin
                
                self.current_state = CheckpointState(
                    run_id=self.run_id,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    total_examples=len(benchmark_data),
                    completed_examples=completed_examples,
                    current_example_idx=i + 1,  # Next example to process
                    sampler_states={name: {} for name in self.samplers.keys()},
                    results=all_results,
                    config=config_dict
                )
                
                # Monitor GPU utilization and adjust batch size if needed
                if self.gpu_optimizer.available_gpus and (i + 1) % 20 == 0:  # Check every 20 examples
                    usage_info = self.gpu_optimizer.monitor_gpu_usage()
                    current_utilization = max(gpu_info.get('utilization_percent', 0) for gpu_info in usage_info.values())
                    utilization_history.append(current_utilization)
                    
                    # Keep only recent history (last 100 measurements)
                    if len(utilization_history) > 100:
                        utilization_history = utilization_history[-100:]
                    
                    # Attempt dynamic batch size adjustment if we have enough history
                    if len(utilization_history) >= 10:
                        current_batch_size = getattr(self.config, 'batch_size', 1)
                        new_batch_size = self.gpu_optimizer.adjust_batch_size_if_needed(
                            current_batch_size, utilization_history[-10:], model, tokenizer
                        )
                        if new_batch_size != current_batch_size:
                            self.config.batch_size = new_batch_size
                            logger.info(f"Dynamically adjusted batch size to {new_batch_size}")
                
                # Save checkpoint periodically or if we're getting close to memory limits
                should_save = (
                    self.checkpoint_manager and 
                    self.checkpoint_manager.should_save_checkpoint(examples_since_checkpoint)
                )
                
                # Also save if we detect high memory usage (emergency checkpoint)
                if self.gpu_optimizer.available_gpus and not should_save:
                    usage_info = self.gpu_optimizer.monitor_gpu_usage()
                    max_utilization = max(gpu_info.get('utilization_percent', 0) for gpu_info in usage_info.values())
                    if max_utilization > 90:  # Emergency checkpoint at 90% GPU memory
                        should_save = True
                        logger.warning(f"High GPU memory usage ({max_utilization:.1f}%), saving emergency checkpoint")
                
                if should_save:
                    # Create config dict with current optimization settings
                    config_dict = self.config.__dict__.copy()
                    config_dict['target_memory_utilization'] = self.gpu_optimizer.target_memory_utilization
                    config_dict['safety_margin'] = self.gpu_optimizer.safety_margin
                    
                    self.current_state = CheckpointState(
                        run_id=self.run_id,
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        total_examples=len(benchmark_data),
                        completed_examples=completed_examples,
                        current_example_idx=i + 1,  # Next example to process
                        sampler_states={name: {} for name in self.samplers.keys()},
                        results=all_results,
                        config=config_dict
                    )
                    self.checkpoint_manager.save_checkpoint(self.current_state)
                    examples_since_checkpoint = 0

            except Exception as e:
                logger.error(f"Error processing example {i}: {e}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                
                # Save emergency checkpoint on error
                if self.checkpoint_manager:
                    logger.info("Saving emergency checkpoint due to error...")
                    # Create config dict with current optimization settings
                    config_dict = self.config.__dict__.copy()
                    config_dict['target_memory_utilization'] = self.gpu_optimizer.target_memory_utilization
                    config_dict['safety_margin'] = self.gpu_optimizer.safety_margin
                    
                    emergency_state = CheckpointState(
                        run_id=self.run_id,
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        total_examples=len(benchmark_data),
                        completed_examples=i,  # Don't count the failed example
                        current_example_idx=i,  # Retry this example
                        sampler_states={name: {} for name in self.samplers.keys()},
                        results=all_results,
                        config=config_dict
                    )
                    self.checkpoint_manager.save_checkpoint(emergency_state)
                
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
        # Identify cases where DCBS and Greedy differ
        differences = []
        for ex_id, preds in prediction_map.items():
            if "dcbs" in preds and "greedy" in preds:
                if preds["dcbs"].get("pred_id") != preds["greedy"].get("pred_id"):
                    differences.append({
                        "id": ex_id,
                        "sentence": preds["dcbs"].get("sentence"),
                        "dcbs_answer": preds["dcbs"].get("predicted_answer"),
                        "greedy_answer": preds["greedy"].get("predicted_answer"),
                        "cluster_info": preds["dcbs"].get("cluster_info"),
                        "answer_probs": preds["dcbs"].get("answer_probs"),
                    })

        results = {
            "statistics": statistics,
            "config": {
                "model": self.config.model_name,
                "total_examples": len(benchmark_data),
                "methods": list(self.samplers.keys()),
                "include_cot": self.config.include_cot,
                "enable_caching": self.config.enable_caching,
                "clustering_method": getattr(self.config, 'clustering_method', 'dbscan'),
                "batch_size": getattr(self.config, 'batch_size', 'auto'),
                "gpu_info": [str(gpu) for gpu in self.gpu_optimizer.available_gpus],
            },
            "detailed_results": all_results,
            "prediction_differences": differences,
            "evaluation_completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Clean up checkpoint on successful completion
        if self.checkpoint_manager:
            self.checkpoint_manager.cleanup_checkpoint(self.run_id)

        logger.info("Evaluation completed successfully!")
        return results
    
    def _make_json_serializable(self, obj):
        """Convert sets and other non-serializable objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj 