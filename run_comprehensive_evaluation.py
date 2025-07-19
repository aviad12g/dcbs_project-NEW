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
import json

import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow


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


def run_baseline_evaluation(
    dataset: str,
    limit: int,
    run_id: str,
    results_dir: Path,
    batch_size: int,
) -> Path:
    """Run baseline evaluation (greedy) and return results file path."""
    baseline_samplers = ["greedy"]
    eval_name = f"{dataset}_baseline"
    
    cmd = [
        "python", "compare_methods.py",
        "--model", "meta-llama/Llama-3.2-1B-Instruct",
        "--limit", str(limit),
        "--samplers", *baseline_samplers,
        "--datasets", dataset,
        "--run-id", f"{run_id}_{eval_name}",
    ]
    
    if batch_size:
        cmd.extend(["--batch-size", str(batch_size)])
        
    success, _ = run_command(cmd, f"{eval_name.upper()} Evaluation")
    
    if not success:
        raise RuntimeError(f"Baseline evaluation failed for {dataset}")
        
    # Find the results file
    results_file = results_dir / f"evaluation_results_{run_id}_{eval_name}.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Could not find baseline results file: {results_file}")
        
    return results_file


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    mlflow.start_run()
    
    # Log Hydra config as MLflow parameters
    mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

    # Adjust limit for quick test
    if cfg.quick_test:
        limit = 10
        print("QUICK TEST MODE: Using 10 examples per dataset")
    else:
        limit = cfg.limit
        print(f"FULL EVALUATION MODE: Using {limit} examples per dataset")
    
    run_id = datetime.now().strftime("comprehensive_%Y%m%d_%H%M%S")
    print(f"Run ID: {run_id}")
    
    results_dir = Path(cfg.results_dir) / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    
    if "all" in cfg.datasets:
        datasets = ["arc_easy", "arc_challenge", "hellaswag", "mmlu_stem"]
    else:
        datasets = cfg.datasets
    
    clustering_methods = cfg.clustering_methods
    
    print(f"Will evaluate:")
    print(f"   Datasets: {', '.join(datasets)}")
    print(f"   Clustering methods: {', '.join(clustering_methods)}")
    print(f"   Examples per dataset: {limit}")
    print(f"   Samplers: {', '.join(cfg.samplers)}")
    if cfg.dcbs_params.use_elbow_method:
        print(f"   Using elbow method for k-means (slower but more accurate)")
    if cfg.batch_size:
        print(f"   Batch size: {cfg.batch_size}")
    print(f"   Cluster history: {'Enabled' if not cfg.disable_cluster_history else 'Disabled'}")
    print(f"   Debug mode: {'Enabled' if not cfg.disable_debug_mode else 'Disabled'}")
    
    if not cfg.quick_test:
        print("\nProceeding with full evaluation (unattended mode)")
    else:
        print("\nProceeding with quick test")
    
    all_results = []
    
    for dataset in datasets:
        
        baseline_results_file = run_baseline_evaluation(
            dataset, limit, run_id, results_dir, cfg.batch_size
        )
        
        for clustering_method in clustering_methods:
            
            if dataset == "mmlu_stem" and clustering_method == "hierarchical" and limit > 50:
                print(f"WARNING: Skipping hierarchical clustering with {dataset} (large dataset)")
                continue
            
            eval_name = f"{dataset}_{clustering_method}"
            
            cmd = [
                "python", "compare_methods.py",
                "--model", cfg.model_path,
                "--limit", str(limit),
                "--baseline-results-file", str(baseline_results_file),
            ]

            cmd.extend(["--samplers", *cfg.samplers])

            cmd.extend([
                "--datasets", dataset,
                "--clustering-method", clustering_method,
                "--enable-disagreement-tracking",
                "--run-id", f"{run_id}_{eval_name}",
                "--dbscan-eps", str(cfg.dcbs_params.dbscan_eps),
                "--dbscan-min-samples", str(cfg.dcbs_params.dbscan_min_samples),
                "--hierarchical-linkage", cfg.dcbs_params.hierarchical_linkage,
                "--k", str(cfg.dcbs_params.k),
                "--top-n", str(cfg.dcbs_params.top_n),
                "--dominance-weight", str(cfg.dcbs_params.dominance_weight),
                "--min-cluster-size", str(cfg.dcbs_params.min_cluster_size),
            ])
            
            if cfg.disable_cluster_history:
                cmd.extend(["--disable-cluster-history"])
            else:
                cmd.extend(["--enable-cluster-history"])
            
            if cfg.disable_debug_mode:
                cmd.extend(["--disable-debug-mode"])
            else:
                cmd.extend(["--debug-mode"])
            
            if cfg.batch_size:
                cmd.extend(["--batch-size", str(cfg.batch_size)])
            
            if cfg.dcbs_params.use_elbow_method:
                cmd.extend(["--use-elbow-method"])
            
            success, output = run_command(cmd, f"{eval_name.upper()} Evaluation")
            
            all_results.append({
                "dataset": dataset,
                "clustering_method": clustering_method,
                "success": success,
                "eval_name": eval_name,
            })
            
            if not success:
                print(f"WARNING: {eval_name} failed, continuing with next evaluation...")
    
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
    
    print(f"\nANALYSIS COMMANDS:")
    print(f"To analyze results, run:")
    
    for result in successful:
        run_dir = f"runs/{run_id}_{result['eval_name']}"
        print(f"   python scripts/summarise_logs.py {run_dir}/events.jsonl")
    
    print(f"\nTo view disagreement details:")
    print(f"   ls results/{run_id}*/")
    print(f"   python -c \"import json; print(json.dumps(json.load(open('results/evaluation_results_*.json')), indent=2))\"")
    
    summary_file = results_dir / "evaluation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "parameters": OmegaConf.to_container(cfg, resolve=True),
            "results": all_results,
            "summary": {
                "total": len(all_results),
                "successful": len(successful),
                "failed": len(failed)
            }
        }, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    
    mlflow.log_artifact(str(summary_file))
    mlflow.log_metrics({
        "total_evaluations": len(all_results),
        "successful_evaluations": len(successful),
        "failed_evaluations": len(failed)
    })

    if len(successful) == len(all_results):
        print("\nALL EVALUATIONS COMPLETED SUCCESSFULLY!")
        mlflow.end_run(status="FINISHED")
        return 0
    else:
        print(f"\nWARNING: {len(failed)} evaluations failed.")
        mlflow.end_run(status="FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())