#!/usr/bin/env python3
"""
Compare different clustering methods for DCBS evaluation.

This script runs evaluations with different clustering methods 
(KMeans, DBSCAN, Hierarchical) and compares their performance.
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List

from src.errors import eval_logger as logger, setup_logging
from src.evaluation_core import EvaluationConfig, EvaluationRunner, load_benchmark_data
from src.visualization import generate_all_visualizations


class ClusteringMethodComparison:
    """Compare different clustering methods for DCBS."""
    
    def __init__(self, base_config: EvaluationConfig):
        self.base_config = base_config
        
    def run_comparison(self, benchmark_data: List[dict], methods: List[str]) -> Dict:
        """
        Run evaluation with different clustering methods.
        
        Args:
            benchmark_data: List of benchmark examples
            methods: List of clustering methods to test
            
        Returns:
            Dictionary containing comparison results
        """
        all_results = {}
        
        for method in methods:
            logger.info(f"\nEvaluating with clustering method: {method}")
            
            # Create config for this method
            config = EvaluationConfig(
                model_name=self.base_config.model_name,
                benchmark_path=self.base_config.benchmark_path,
                output_dir=self.base_config.output_dir,
                limit=self.base_config.limit,
                top_p=self.base_config.top_p,
                k=self.base_config.k,
                top_n=self.base_config.top_n,
                include_cot=self.base_config.include_cot,
                log_level=self.base_config.log_level,
                load_in_4bit=self.base_config.load_in_4bit,
                enable_caching=self.base_config.enable_caching,
                clustering_method=method,
                dbscan_eps=self.base_config.dbscan_eps,
                dbscan_min_samples=self.base_config.dbscan_min_samples,
                hierarchical_linkage=self.base_config.hierarchical_linkage,
            )
            
            # Run evaluation
            runner = EvaluationRunner(config)
            results = runner.run_evaluation(benchmark_data)
            
            # Store results
            all_results[method] = results
            
        return self._create_comparison_summary(all_results)
    
    def _create_comparison_summary(self, all_results: Dict) -> Dict:
        """Create a summary comparing all clustering methods."""
        summary = {
            "comparison_type": "clustering_methods",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "model": self.base_config.model_name,
                "examples": self.base_config.limit,
                "k": self.base_config.k,
                "top_n": self.base_config.top_n,
            },
            "methods": {},
            "detailed_results": all_results
        }
        
        # Extract key metrics for each method
        for method, results in all_results.items():
            stats = results.get("statistics", {})
            dcbs_stats = stats.get("dcbs", {})
            
            summary["methods"][method] = {
                "accuracy": dcbs_stats.get("accuracy", 0),
                "correct": dcbs_stats.get("correct", 0),
                "total": dcbs_stats.get("total", 0),
                "avg_time_ms": dcbs_stats.get("avg_time_ms", 0),
                "confidence_interval": dcbs_stats.get("confidence_interval", [0, 0])
            }
        
        return summary
    
    def save_comparison_results(self, results: Dict, output_dir: str):
        """Save comparison results to file."""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"clustering_comparison_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Comparison results saved to: {filepath}")
        
        # Print summary table
        self._print_comparison_table(results)
    
    def _print_comparison_table(self, results: Dict):
        """Print a formatted comparison table."""
        print("\n" + "=" * 80)
        print("CLUSTERING METHOD COMPARISON RESULTS")
        print("=" * 80)
        print(f"Model: {results['config']['model']}")
        print(f"Examples: {results['config']['examples']}")
        print(f"Parameters: k={results['config']['k']}, top_n={results['config']['top_n']}")
        print("-" * 80)
        print(f"{'Method':<15} {'Accuracy':<12} {'Correct/Total':<15} {'Avg Time (ms)':<15} {'95% CI'}")
        print("-" * 80)
        
        # Sort methods by accuracy
        methods = sorted(results["methods"].items(), 
                        key=lambda x: x[1]["accuracy"], 
                        reverse=True)
        
        for method, stats in methods:
            ci = stats["confidence_interval"]
            print(f"{method:<15} "
                  f"{stats['accuracy']:.2f}%{'':<6} "
                  f"{stats['correct']}/{stats['total']:<12} "
                  f"{stats['avg_time_ms']:.2f}{'':<10} "
                  f"({ci[0]:.1f}, {ci[1]:.1f})")
        
        print("=" * 80)


def main():
    """Main function to run clustering method comparison."""
    parser = argparse.ArgumentParser(
        description="Compare different clustering methods for DCBS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model to use"
    )
    
    parser.add_argument(
        "--benchmark",
        type=str,
        default="data/arc_easy_full.json",
        help="Path to benchmark JSON file"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of examples to evaluate"
    )
    
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["kmeans", "dbscan", "hierarchical"],
        default=["kmeans", "dbscan", "hierarchical"],
        help="Clustering methods to compare"
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of clusters for methods that use k"
    )
    
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Top-n tokens for DCBS"
    )
    
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.3,
        help="DBSCAN epsilon parameter"
    )
    
    parser.add_argument(
        "--hierarchical-linkage",
        type=str,
        choices=["ward", "complete", "average", "single"],
        default="average",
        help="Linkage criterion for hierarchical clustering"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/clustering_comparison",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model with 4-bit quantization"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    logger.info("Starting clustering method comparison")
    logger.info(f"Methods to compare: {args.methods}")
    
    # Create base configuration
    base_config = EvaluationConfig(
        model_name=args.model,
        benchmark_path=args.benchmark,
        output_dir=args.output_dir,
        limit=args.limit,
        top_p=0.9,  # Not used for DCBS
        k=args.k,
        top_n=args.top_n,
        include_cot=True,
        log_level=args.log_level,
        load_in_4bit=args.load_in_4bit,
        enable_caching=True,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=2,
        hierarchical_linkage=args.hierarchical_linkage,
    )
    
    # Load benchmark data
    benchmark_data = load_benchmark_data(base_config.benchmark_path)
    
    # Run comparison
    comparison = ClusteringMethodComparison(base_config)
    results = comparison.run_comparison(benchmark_data, args.methods)
    
    # Save results
    comparison.save_comparison_results(results, args.output_dir)
    
    logger.info("Clustering method comparison completed!")


if __name__ == "__main__":
    main() 