#!/usr/bin/env python3
"""
DCBS Clustering Optimisation Script ("AdamIC" grade)
====================================================

This standalone script performs an exhaustive grid search over multiple
clustering algorithms (K-means, DBSCAN and Hierarchical) and their key
hyper-parameters to discover the *best* DCBS configuration for a given
benchmark.

Key features:
• Disables Chain-of-Thought (CoT) reasoning for much faster iterations.
• Evaluates *only* the DCBS sampler to minimise GPU time.
• Automatically resumes GPU-optimised batch sizing provided by the core
  framework.
• Produces a concise leaderboard (accuracy & avg latency) and writes a full
  JSON log to *results/dcbs_clustering_optim_<timestamp>.json*.

Example usage (all arguments are optional)::

    python scripts/optimize_clustering.py \
        --config configs/dcbs_config.yaml \
        --benchmark data/arc_easy_full.json \
        --limit 50

The default search space is intentionally compact so it runs on a single
consumer GPU (<10 runs). Tweak the grids below for deeper sweeps.
"""

import argparse
import itertools
import json
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np

from src.evaluation_core.config import EvaluationConfig
from src.evaluation_core.runner import EvaluationRunner
from src.evaluation_core.utils import load_benchmark_data
from src.errors import eval_logger as logger

# ---------------------------------------------------------------------------
# Search space definitions – tweak to taste
# ---------------------------------------------------------------------------
KMEANS_GRID = {
    "k": [4, 8, 16],
    "top_n": [30, 50],  # candidate pool for clustering
}

DBSCAN_GRID = {
    "dbscan_eps": [0.2, 0.3, 0.5],
    "dbscan_min_samples": [1, 2, 3],
}

HIERARCHICAL_GRID = {
    "k": [4, 8, 16],
    "hierarchical_linkage": ["average", "complete"],
}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def cartesian_product(grid: Dict[str, List]) -> List[Dict[str, object]]:
    """Expand a parameter grid into a list of dictionaries (sklearn-style)."""
    keys = list(grid.keys())
    values_product = itertools.product(*(grid[k] for k in keys))
    return [{k: v for k, v in zip(keys, combo)} for combo in values_product]


def build_eval_config(base_cfg: EvaluationConfig, overrides: Dict[str, object]) -> EvaluationConfig:
    """Create a new EvaluationConfig with *overrides* applied immutably."""
    cfg = deepcopy(base_cfg)

    # Mandatory tweaks for optimisation sweeps
    cfg.include_cot = False  # disable CoT for speed

    # Apply method-specific overrides
    for k, v in overrides.items():
        setattr(cfg, k, v)

    return cfg


# ---------------------------------------------------------------------------
# Main optimisation routine
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Grid search DCBS clustering parameters")
    parser.add_argument("--config", default="configs/dcbs_config.yaml", help="Base YAML config path")
    parser.add_argument("--benchmark", default=None, help="Override benchmark JSON path or dataset key (e.g. arc_challenge)")
    parser.add_argument("--model", default=None, help="HuggingFace model name or path to load (overrides YAML)")
    parser.add_argument("--limit", type=int, default=50, help="Limit examples for quick sweeps")
    parser.add_argument("--batch-size", type=int, default=None, dest="batch_size", help="Fixed batch size to bypass auto search (recommended for large models)")
    parser.add_argument("--dry-run", action="store_true", help="List planned runs without executing")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load *validated* YAML config into an EvaluationConfig instance
    # ------------------------------------------------------------------
    from src.config_builder import ConfigBuilder  # local import to avoid cli deps early
    class _Empty:  # Dummy argparse namespace – we override via code
        pass
    dummy_args = _Empty()
    base_config = ConfigBuilder.from_yaml_and_args(args.config, dummy_args)  # type: ignore

    # Model & benchmark overrides
    if args.model:
        base_config.model_name = args.model
    if args.benchmark:
        base_config.benchmark_path = args.benchmark
    if args.limit:
        base_config.limit = args.limit
    if args.batch_size:
        base_config.batch_size = args.batch_size

    # ------------------------------------------------------------------
    # Build parameter combinations for each clustering method
    # ------------------------------------------------------------------
    search_space: List[Dict[str, object]] = []

    # K-means (spherical cosine) – the framework's default implementation
    for combo in cartesian_product(KMEANS_GRID):
        combo.update({"clustering_method": "kmeans"})
        search_space.append(combo)

    # DBSCAN
    for combo in cartesian_product(DBSCAN_GRID):
        combo.update({"clustering_method": "dbscan"})
        search_space.append(combo)

    # Hierarchical
    for combo in cartesian_product(HIERARCHICAL_GRID):
        combo.update({"clustering_method": "hierarchical"})
        search_space.append(combo)

    logger.info(f"Planned {len(search_space)} optimisation runs")

    if args.dry_run:
        for i, params in enumerate(search_space, 1):
            print(f"{i:02d}. {params}")
        return 0

    # ------------------------------------------------------------------
    # Load benchmark data ONCE (re-used by all runs)
    # ------------------------------------------------------------------
    benchmark_data = load_benchmark_data(base_config.benchmark_path)
    if args.limit and args.limit < len(benchmark_data):
        benchmark_data = benchmark_data[: args.limit]
        logger.info(f"Using first {args.limit} examples for the sweep")

    results_log = []
    best_score = -1.0
    best_params = {}

    for run_idx, param_overrides in enumerate(search_space, 1):
        run_id = f"opt_{param_overrides['clustering_method']}_{run_idx}_{int(time.time())}"
        cfg = build_eval_config(base_config, param_overrides)

        logger.info("=" * 80)
        logger.info(f"Run {run_idx}/{len(search_space)} – parameters: {param_overrides}")

        # Execute evaluation – only DCBS sampler to save time
        runner = EvaluationRunner(
            cfg,
            requested_samplers=["dcbs"],
            run_id=run_id,
            enable_checkpointing=False,
        )

        eval_results = runner.run_evaluation(benchmark_data)
        dcbs_stats = eval_results["statistics"].get("dcbs", {})

        # Detect & skip runs that were aborted because the model fell back to CPU.
        if dcbs_stats.get("accuracy", 0.0) == -1.0 or dcbs_stats.get("skipped_cpu_fallback"):
            logger.warning("Run skipped due to CPU fallback – omitting from leaderboard")
            continue

        accuracy = dcbs_stats.get("accuracy", 0.0)
        avg_time_ms = dcbs_stats.get("avg_time_ms", 0.0)

        logger.info(
            f"Completed run {run_idx} – Accuracy: {accuracy:.2f}%, Avg latency: {avg_time_ms:.1f} ms"
        )

        record = {
            "run_id": run_id,
            "params": param_overrides,
            "accuracy": accuracy,
            "avg_time_ms": avg_time_ms,
            "correct": dcbs_stats.get("correct"),
            "total": dcbs_stats.get("total"),
        }
        results_log.append(record)

        # Track best configuration (accuracy first, latency tie-breaker)
        if (accuracy > best_score) or (
            np.isclose(accuracy, best_score) and avg_time_ms < record["avg_time_ms"]
        ):
            best_score = accuracy
            best_params = param_overrides

    # ------------------------------------------------------------------
    # Save full log & print leaderboard
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = Path("results") / f"dcbs_clustering_optim_{timestamp}.json"
    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=2)
    logger.info(f"Full optimisation log written to {out_file}")

    # Pretty leaderboard
    print("\n" + "=" * 70)
    print("DCBS CLUSTERING OPTIMISATION – SUMMARY")
    print("=" * 70)
    # Sort by accuracy desc, latency asc
    leaderboard = sorted(results_log, key=lambda r: (-r["accuracy"], r["avg_time_ms"]))
    print(f"{'Rank':<6}{'Method':<12}{'Params':<40}{'Acc':<8}{'Time (ms)':<10}")
    print("-" * 70)
    for rank, rec in enumerate(leaderboard, 1):
        method = rec["params"]["clustering_method"]
        param_str = {k: v for k, v in rec["params"].items() if k != "clustering_method"}
        print(
            f"{rank:<6}{method:<12}{str(param_str):<40}{rec['accuracy']:.2f}%   {rec['avg_time_ms']:.1f}"
        )

    print("-" * 70)
    print(f"BEST CONFIG → {best_params} with {best_score:.2f}% accuracy")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main()) 