#!/usr/bin/env python3
"""
Enhanced comparison script with checkpointing, GPU optimization, and resumption.

This script compares DCBS with baseline sampling methods while supporting:
- Automatic checkpoint saving and resumption
- GPU optimization for batch sizing
- Multi-GPU support
- Graceful interruption handling
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from src.cli_parser import ArgumentParserSetup
from src.config_builder import ConfigBuilder
from src.errors import eval_logger as logger
from src.evaluation_core.checkpoint import CheckpointManager
from src.evaluation_core.gpu_optimizer import get_gpu_optimizer
from src.evaluation_core.runner import EvaluationRunner
from src.evaluation_core.utils import load_benchmark_data
from src.visualization import generate_all_visualizations


def list_available_checkpoints():
    """List available checkpoints for resumption."""
    checkpoint_manager = CheckpointManager()
    checkpoints = checkpoint_manager.list_checkpoints()
    
    if not checkpoints:
        print("No checkpoints available for resumption.")
        return
    
    print("\nAvailable checkpoints:")
    print("-" * 50)
    
    for run_id in checkpoints:
        checkpoint = checkpoint_manager.load_checkpoint(run_id)
        if checkpoint:
            progress = f"{checkpoint.completed_examples}/{checkpoint.total_examples}"
            print(f"Run ID: {run_id}")
            print(f"  Progress: {progress} examples ({(checkpoint.completed_examples/checkpoint.total_examples)*100:.1f}%)")
            print(f"  Timestamp: {checkpoint.timestamp}")
            print()


def main():
    """Main evaluation function with enhanced capabilities."""
    # Parse arguments
    args = ArgumentParserSetup.parse_args()
    
    # Handle special commands
    if hasattr(args, 'list_checkpoints') and args.list_checkpoints:
        list_available_checkpoints()
        return 0
    
    # Load configuration
    yaml_config_path = "configs/dcbs_config.yaml"
    config = ConfigBuilder.from_yaml_and_args(yaml_config_path, args)
    
    # Initialize GPU optimizer early
    gpu_optimizer = get_gpu_optimizer()
    
    # Generate run ID
    run_id = args.run_id if hasattr(args, 'run_id') and args.run_id else None
    if not run_id:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"eval_{timestamp}"
    
    # Check for existing checkpoint
    checkpoint_manager = CheckpointManager()
    existing_checkpoint = checkpoint_manager.load_checkpoint(run_id)
    
    if existing_checkpoint:
        print(f"\nFound existing checkpoint for run ID: {run_id}")
        print(f"Progress: {existing_checkpoint.completed_examples}/{existing_checkpoint.total_examples} examples")
        
        resume = input("Resume from checkpoint? [Y/n]: ").strip().lower()
        if resume in ['n', 'no']:
            # Generate new run ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"eval_{timestamp}_new"
            print(f"Starting new evaluation with run ID: {run_id}")
        else:
            print("Resuming from checkpoint...")
    
    try:
        # Load benchmark data
        logger.info(f"Loading benchmark: {config.benchmark_path}")
        benchmark_data = load_benchmark_data(config.benchmark_path)
        logger.info(f"Loaded {len(benchmark_data)} examples from dataset {config.benchmark_path}")
        
        # Create and run evaluation
        samplers_to_use = args.samplers if hasattr(args, 'samplers') and args.samplers else None
        logger.info(f"Requested samplers: {samplers_to_use}")
        
        runner = EvaluationRunner(
            config, 
            requested_samplers=samplers_to_use,
            run_id=run_id,
            enable_checkpointing=True
        )
        
        results = runner.run_evaluation(benchmark_data)
        
        # Save results with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results/evaluation_results_{timestamp}.json"
        
        Path("results").mkdir(exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"JSON results saved to: {results_file}")
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        generate_all_visualizations(results, "results")
        
        # Display summary
        statistics = results["statistics"]
        config_info = results["config"]
        
        print("\n" + "=" * 70)
        print("DCBS EVALUATION RESULTS")
        print("=" * 70)
        print(f"Model: {config_info['model']}")
        print(f"Total Examples: {config_info['total_examples']}")
        print(f"Methods: {', '.join(config_info['methods'])}")
        print(f"CoT: {'Enabled' if config_info.get('include_cot', True) else 'Disabled'}")
        print(f"Caching: {'Enabled' if config_info.get('enable_caching', True) else 'Disabled'}")
        
        # GPU Information
        if gpu_optimizer.available_gpus:
            print(f"GPUs: {len(gpu_optimizer.available_gpus)} detected")
        
        print("-" * 70)
        print(f"{'Method':<12} {'Accuracy':<12} {'95% CI':<20} {'N':<8} {'Time (ms)':<12}")
        print("-" * 70)
        
        for method, stats in statistics.items():
            ci = stats.get("confidence_interval", (0, 0))
            accuracy = f"{stats['accuracy']:.2f}%"
            ci_str = f"({ci[0]:.1f}, {ci[1]:.1f})"
            n_str = f"{stats['correct']}/{stats['total']}"
            time_str = f"{stats['avg_time_ms']:.2f}"
            
            print(f"{method.upper():<12} {accuracy:<12} {ci_str:<20} {n_str:<8} {time_str:<12}")
        
        # Show prediction differences
        differences = results.get("prediction_differences", [])
        if differences:
            print(f"{len(differences)} differing predictions between DCBS and Greedy.")
        else:
            print("No differing predictions between DCBS and Greedy.")
        
        print("-" * 70)
        
        # Baseline comparison
        num_options = 4  # Assuming 4-option multiple choice
        random_baseline = 100.0 / num_options
        print(f"Random baseline for {num_options}-option: {random_baseline:.1f}%")
        print("=" * 70)
        
        logger.info("Evaluation completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user. Checkpoint should be saved automatically.")
        return 1
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 