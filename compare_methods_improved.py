#!/usr/bin/env python3
"""
Improved DCBS Evaluation Framework

This script provides the improved evaluation framework implementing the proper
conversation flow and addressing all code review feedback:

1. Never let LLM complete 'user' messages, only 'assistant' messages
2. Proper two-step conversation flow with KV caching
3. No ChatTemplateManager dependency (uses default tokenizer.chat_template)
4. Fixed parameter handling issues
5. Increased token limits to avoid truncation
6. Enhanced logging and debugging
"""

import csv
import datetime
import json
import os
import sys
from typing import Dict, List

from src.errors import eval_logger as logger
from src.errors import setup_logging
from src.evaluation_core import (
    EvaluationConfig,
    load_benchmark_data,
)
from src.evaluation_core.improved_runner import ImprovedEvaluationRunner
from src.visualization import generate_all_visualizations
from src.cli_parser import ArgumentParserSetup
from src.config_builder import ConfigBuilder


class ImprovedParameterSweepRunner:
    """Handles parameter sweep evaluations using improved components."""

    def __init__(self, base_config: EvaluationConfig):
        self.base_config = base_config

    def run_sweep(self, benchmark_data: List[dict], args) -> Dict:
        """Run parameter sweep evaluation using improved runner."""
        # Determine parameter ranges
        k_values = args.sweep_k if args.sweep_k else [self.base_config.k]
        top_n_values = args.sweep_top_n if args.sweep_top_n else [self.base_config.top_n]
        top_p_values = args.sweep_top_p if args.sweep_top_p else [self.base_config.top_p]
        
        all_results = []
        
        logger.info(f"Running parameter sweep: k={k_values}, top_n={top_n_values}, top_p={top_p_values}")
        
        for k in k_values:
            for top_n in top_n_values:
                for top_p in top_p_values:
                    logger.info(f"Evaluating with k={k}, top_n={top_n}, top_p={top_p}")
                    
                    sweep_config = self._create_sweep_config(k, top_n, top_p)
                    results = self._run_single_sweep(sweep_config, benchmark_data, args)
                    
                    results["sweep_params"] = {"k": k, "top_n": top_n, "top_p": top_p}
                    all_results.append(results)
        
        return {"sweep_results": all_results, "base_config": self.base_config.__dict__}

    def _create_sweep_config(self, k: int, top_n: int, top_p: float) -> EvaluationConfig:
        """Create a configuration for a single sweep iteration."""
        return EvaluationConfig(
            model_name=self.base_config.model_name,
            benchmark_path=self.base_config.benchmark_path,
            output_dir=self.base_config.output_dir,
            limit=self.base_config.limit,
            top_p=top_p,
            k=k,
            top_n=top_n,
            include_cot=self.base_config.include_cot,
            log_level=self.base_config.log_level,
            load_in_4bit=self.base_config.load_in_4bit,
            enable_caching=self.base_config.enable_caching,
        )

    def _run_single_sweep(self, config: EvaluationConfig, benchmark_data: List[dict], args) -> Dict:
        """Run evaluation for a single parameter configuration."""
        runner = ImprovedEvaluationRunner(config)
        
        if args.samplers:
            available_samplers = {
                k: v for k, v in runner.samplers.items() if k in args.samplers
            }
            runner.samplers = available_samplers
        
        return runner.run_evaluation(benchmark_data)


class ImprovedResultsManager:
    """Enhanced results manager with improved formatting."""

    @staticmethod
    def save_results(results: Dict, output_dir: str, args):
        """Save evaluation results with enhanced formatting."""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if args.output_format in ["json", "both"]:
            json_path = os.path.join(output_dir, f"improved_evaluation_results_{timestamp}.json")
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"JSON results saved to: {json_path}")

        if args.output_format in ["csv", "both"]:
            csv_path = os.path.join(output_dir, f"improved_evaluation_results_{timestamp}.csv")
            ImprovedResultsManager._save_csv_results(results, csv_path)
            logger.info(f"CSV results saved to: {csv_path}")

        if args.save_details:
            details_path = os.path.join(output_dir, f"improved_detailed_results_{timestamp}.json")
            with open(details_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Detailed results saved to: {details_path}")

    @staticmethod
    def _save_csv_results(results: Dict, csv_path: str):
        """Save results in CSV format with improved structure."""
        statistics = results.get("statistics", {})
        config = results.get("config", {})
        
        # Prepare CSV data
        csv_data = []
        
        # Add header with metadata
        csv_data.append(["Improved DCBS Evaluation Results", results.get("evaluation_completed_at", "")])
        csv_data.append(["Model", config.get("model", "Unknown")])
        csv_data.append(["Benchmark", config.get("benchmark", "Unknown")])
        csv_data.append(["Total Examples", config.get("total_examples", 0)])
        csv_data.append(["CoT Enabled", config.get("include_cot", True)])
        csv_data.append(["Caching Enabled", config.get("enable_caching", True)])
        csv_data.append([])  # Empty row for separation
        
        # Add results table header
        csv_data.append(["Method", "Accuracy (%)", "Correct", "Total", "95% CI Low", "95% CI High", "Avg Time (ms)"])
        
        # Sort methods by accuracy (descending)
        sorted_methods = sorted(
            statistics.items(), 
            key=lambda x: x[1].get('accuracy', 0), 
            reverse=True
        )
        
        # Add data for each method
        for method, stats in sorted_methods:
            ci = stats.get("confidence_interval", (0, 0))
            csv_data.append([
                method.upper(),
                f"{stats.get('accuracy', 0):.2f}",
                stats.get("correct", 0),
                stats.get("total", 0),
                f"{ci[0]:.2f}" if ci else "N/A",
                f"{ci[1]:.2f}" if ci else "N/A",
                f"{stats.get('avg_time_ms', 0):.2f}"
            ])
        
        # Write CSV file
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)

    @staticmethod
    def print_summary(results: Dict, is_sweep: bool = False):
        """Print an enhanced summary of evaluation results."""
        print("\n" + "=" * 80)
        print("IMPROVED DCBS EVALUATION RESULTS")
        print("=" * 80)
        
        config = results.get("config", {})
        print(f"Model: {config.get('model', 'Unknown')}")
        print(f"Examples: {config.get('total_examples', 0):,}")
        print(f"CoT: {'Enabled' if config.get('include_cot', True) else 'Disabled'}")
        print(f"Caching: {'Enabled' if config.get('enable_caching', True) else 'Disabled'}")
        print(f"Completed: {results.get('evaluation_completed_at', 'Unknown')}")
        print("-" * 80)

        if is_sweep:
            ImprovedResultsManager._print_sweep_summary(results)
        else:
            ImprovedResultsManager._print_single_summary(results)

        print("-" * 80)
        print("Improvements in this version:")
        print("  • Proper conversation flow (LLM only completes assistant messages)")
        print("  • KV caching for improved performance")
        print("  • Increased token limits (500 vs 200)")
        print("  • Enhanced debugging and logging")
        print("  • Removed ChatTemplateManager dependency")
        print("=" * 80)

    @staticmethod
    def _print_single_summary(results: Dict):
        """Print summary for single configuration results."""
        statistics = results["statistics"]

        # Sort methods by accuracy (descending)
        sorted_methods = sorted(
            statistics.items(), 
            key=lambda x: x[1]["accuracy"], 
            reverse=True
        )

        print(f"{'Rank':<6} {'Method':<10} {'Accuracy':<12} {'95% CI':<20} {'N':<8} {'Time (ms)':<12}")
        print("-" * 80)

        for rank, (method, stats) in enumerate(sorted_methods, 1):
            ci = stats.get("confidence_interval", (0, 0))
            ci_str = f"({ci[0]:.1f}, {ci[1]:.1f})"

            # Add marker for best method
            rank_display = f"[1] {rank}" if rank == 1 else f"    {rank}"

            print(
                f"{rank_display:<6} "
                f"{method.upper():<10} "
                f"{stats['accuracy']:.2f}%{'':<6} "
                f"{ci_str:<20} "
                f"{stats['total']:<8} "
                f"{stats.get('avg_time_ms', 0):.2f}"
            )

    @staticmethod
    def _print_sweep_summary(results: Dict):
        """Print summary for parameter sweep results."""
        print("PARAMETER SWEEP RESULTS")
        print("-" * 80)
        
        for i, sweep_result in enumerate(results["sweep_results"]):
            params = sweep_result["sweep_params"]
            statistics = sweep_result["statistics"]
            
            print(f"\nConfiguration {i+1}: k={params['k']}, top_n={params['top_n']}, top_p={params['top_p']}")
            print(f"{'Method':<12} {'Accuracy':<12} {'Correct/Total':<15} {'Time (ms)':<12}")
            print("-" * 60)
            
            sorted_methods = sorted(
                statistics.items(), 
                key=lambda x: x[1]["accuracy"], 
                reverse=True
            )
            
            for method, stats in sorted_methods:
                print(
                    f"{method.upper():<12} "
                    f"{stats['accuracy']:.2f}%{'':<6} "
                    f"{stats['correct']}/{stats['total']:<10} "
                    f"{stats.get('avg_time_ms', 0):.2f}"
                )


class ImprovedEvaluationFramework:
    """Enhanced evaluation framework with all improvements."""

    def __init__(self):
        self.args = ArgumentParserSetup.parse_args()
        self.config = ConfigBuilder.from_yaml_and_args(self.args.config, self.args)

    def run(self):
        """Run the improved evaluation framework."""
        # Setup enhanced logging
        log_level = self.args.log_level or self.config.log_level
        setup_logging(log_level=log_level)

        logger.info("Starting improved DCBS evaluation framework")
        logger.info("Improvements: Proper conversation flow, KV caching, enhanced logging")
        logger.info(f"Configuration: {self.config}")

        try:
            benchmark_data = load_benchmark_data(self.config.benchmark_path)

            is_sweep = any([self.args.sweep_k, self.args.sweep_top_n, self.args.sweep_top_p])
            
            if is_sweep:
                results = self._run_parameter_sweep(benchmark_data)
            else:
                results = self._run_single_evaluation(benchmark_data)

            self._save_and_display_results(results, is_sweep)

            logger.info("Improved evaluation completed successfully!")

        except KeyboardInterrupt:
            logger.info("Evaluation interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    def _run_parameter_sweep(self, benchmark_data: List[dict]) -> Dict:
        """Run parameter sweep evaluation."""
        logger.info("Running parameter sweep evaluation...")
        sweep_runner = ImprovedParameterSweepRunner(self.config)
        return sweep_runner.run_sweep(benchmark_data, self.args)

    def _run_single_evaluation(self, benchmark_data: List[dict]) -> Dict:
        """Run single configuration evaluation."""
        logger.info("Running single configuration evaluation...")
        runner = ImprovedEvaluationRunner(self.config)

        if self.args.samplers:
            available_samplers = {
                k: v for k, v in runner.samplers.items() if k in self.args.samplers
            }
            runner.samplers = available_samplers
            logger.info(f"Using samplers: {list(available_samplers.keys())}")

        return runner.run_evaluation(benchmark_data)

    def _save_and_display_results(self, results: Dict, is_sweep: bool):
        """Save results and generate visualizations."""
        ImprovedResultsManager.save_results(results, self.config.output_dir, self.args)

        if not is_sweep:
            logger.info("Generating visualizations...")
            generate_all_visualizations(results, self.config.output_dir)

        ImprovedResultsManager.print_summary(results, is_sweep)


def main():
    """Main entry point for the improved evaluation framework."""
    framework = ImprovedEvaluationFramework()
    framework.run()


if __name__ == "__main__":
    main() 