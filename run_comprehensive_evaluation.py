#!/usr/bin/env python3
"""
Comprehensive DCBS Evaluation Script

This script runs a complete evaluation of DCBS across multiple datasets and clustering methods,
with full disagreement tracking at the token level.

Usage:
    python run_comprehensive_evaluation.py [--limit N] [--quick-test]
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(cmd: list, description: str) -> tuple[bool, str]:
    """Run a command and return success status and output."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        # Run with real-time output instead of capturing
        print("Starting evaluation... (this may take several minutes)")
        print("Press Ctrl+C to interrupt if needed.")
        
        result = subprocess.run(
            cmd, 
            check=False
        )
        
        success = result.returncode == 0
        if success:
            print(f"\n{description} completed successfully!")
        else:
            print(f"\n{description} failed with return code {result.returncode}")
        
        return success, ""
        
    except KeyboardInterrupt:
        print(f"\n{description} interrupted by user")
        return False, "interrupted"
    except Exception as e:
        print(f"{description} failed with exception: {e}")
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive DCBS evaluation")
    parser.add_argument(
        "--limit", 
        type=int, 
        default=100,
        help="Number of examples per dataset (default: 100)"
    )
    parser.add_argument(
        "--quick-test", 
        action="store_true",
        help="Run quick test with 10 examples per dataset"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["arc_easy", "arc_challenge", "hellaswag", "mmlu_stem", "all"],
        default=["all"],
        help="Datasets to evaluate (default: all)"
    )
    parser.add_argument(
        "--clustering-methods",
        nargs="+", 
        choices=["dbscan", "hierarchical", "kmeans"],
        default=["dbscan", "hierarchical"],
        help="Clustering methods to test (default: dbscan, hierarchical)"
    )
    
    args = parser.parse_args()
    
    # Adjust limit for quick test
    if args.quick_test:
        limit = 10
        print("QUICK TEST MODE: Using 10 examples per dataset")
    else:
        limit = args.limit
        print(f"FULL EVALUATION MODE: Using {limit} examples per dataset")
    
    # Generate run ID
    run_id = datetime.now().strftime("comprehensive_%Y%m%d_%H%M%S")
    print(f"Run ID: {run_id}")
    
    # Create results directory
    results_dir = Path("results") / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    
    # Determine datasets to run
    if "all" in args.datasets:
        datasets = ["arc_easy", "arc_challenge", "hellaswag", "mmlu_stem"]
    else:
        datasets = args.datasets
    
    clustering_methods = args.clustering_methods
    
    print(f"Will evaluate:")
    print(f"   Datasets: {', '.join(datasets)}")
    print(f"   Clustering methods: {', '.join(clustering_methods)}")
    print(f"   Examples per dataset: {limit}")
    
    # Auto-proceed for unattended runs
    if not args.quick_test:
        print("\nProceeding with full evaluation (unattended mode)")
    else:
        print("\nProceeding with quick test")
    
    # Track all results
    all_results = []
    
    # Run evaluations for each dataset and clustering method combination
    for dataset in datasets:
        for clustering_method in clustering_methods:
            
            # Skip if hierarchical with MMLU (might be too slow)
            if dataset == "mmlu_stem" and clustering_method == "hierarchical" and limit > 50:
                print(f"WARNING: Skipping hierarchical clustering with {dataset} (large dataset)")
                continue
            
            eval_name = f"{dataset}_{clustering_method}"
            
            # Build command
            cmd = [
                "python", "compare_methods.py",
                "--model", "meta-llama/Llama-3.2-1B-Instruct",
                "--limit", str(limit),
                "--samplers", "greedy", "dcbs",
                "--datasets", dataset,
                "--clustering-method", clustering_method,
                "--enable-cluster-history",
                "--debug-mode",
                "--enable-disagreement-tracking",
                "--run-id", f"{run_id}_{eval_name}",
            ]
            
            # Add hierarchical-specific parameters
            if clustering_method == "hierarchical":
                cmd.extend(["--hierarchical-linkage", "average"])
            
            # Run evaluation
            success, output = run_command(cmd, f"{eval_name.upper()} Evaluation")
            
            all_results.append({
                "dataset": dataset,
                "clustering_method": clustering_method,
                "success": success,
                "eval_name": eval_name,
            })
            
            if not success:
                print(f"WARNING: {eval_name} failed, continuing with next evaluation...")
    
    # Summary report
    print(f"\n{'='*80}")
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Run ID: {run_id}")
    print(f"Total evaluations: {len(all_results)}")
    
    successful = [r for r in all_results if r["success"]]
    failed = [r for r in all_results if not r["success"]]
    
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\nSUCCESSFUL EVALUATIONS:")
        for result in successful:
            print(f"   • {result['eval_name']}")
    
    if failed:
        print(f"\nFAILED EVALUATIONS:")
        for result in failed:
            print(f"   • {result['eval_name']}")
    
    # Generate analysis commands
    print(f"\nANALYSIS COMMANDS:")
    print(f"To analyze results, run:")
    
    for result in successful:
        run_dir = f"runs/{run_id}_{result['eval_name']}"
        print(f"   python scripts/summarise_logs.py {run_dir}/events.jsonl")
    
    print(f"\nTo view disagreement details:")
    print(f"   ls results/{run_id}*/")
    print(f"   python -c \"import json; print(json.dumps(json.load(open('results/evaluation_results_*.json')), indent=2))\"")
    
    # Create summary file
    summary_file = results_dir / "evaluation_summary.json"
    import json
    with open(summary_file, 'w') as f:
        json.dump({
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "limit": limit,
                "datasets": datasets,
                "clustering_methods": clustering_methods,
                "quick_test": args.quick_test
            },
            "results": all_results,
            "summary": {
                "total": len(all_results),
                "successful": len(successful),
                "failed": len(failed)
            }
        }, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    
    if len(successful) == len(all_results):
        print("\nALL EVALUATIONS COMPLETED SUCCESSFULLY!")
        return 0
    else:
        print(f"\nWARNING: {len(failed)} evaluations failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 