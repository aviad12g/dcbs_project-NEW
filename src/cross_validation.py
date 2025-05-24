"""
Cross-validation support for robust DCBS evaluation.

This module provides k-fold cross-validation and stratified sampling
for more reliable performance estimates.
"""

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from evaluation_core import EvaluationConfig, EvaluationRunner


@dataclass
class CrossValidationConfig:
    """Configuration for cross-validation."""

    n_folds: int = 5
    stratify: bool = True
    shuffle: bool = True
    random_state: int = 42


@dataclass
class FoldResults:
    """Results from a single fold."""

    fold_id: int
    train_size: int
    test_size: int
    method_accuracies: Dict[str, float]
    method_times: Dict[str, float]


class StratifiedKFold:
    """Stratified k-fold cross-validation for binary classification."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, data: List[Dict]) -> Iterator[Tuple[List[int], List[int]]]:
        """Generate stratified train/test splits."""
        # Group by correct answer for stratification
        class_0_indices = []
        class_1_indices = []

        for i, example in enumerate(data):
            correct_option = example.get("correct_option", "1")
            if correct_option == "1":
                class_0_indices.append(i)
            else:
                class_1_indices.append(i)

        # Shuffle if requested
        if self.shuffle:
            random.seed(self.random_state)
            random.shuffle(class_0_indices)
            random.shuffle(class_1_indices)

        # Create folds
        class_0_folds = self._create_folds(class_0_indices, self.n_splits)
        class_1_folds = self._create_folds(class_1_indices, self.n_splits)

        # Generate train/test splits
        for fold in range(self.n_splits):
            test_indices = class_0_folds[fold] + class_1_folds[fold]
            train_indices = []

            for other_fold in range(self.n_splits):
                if other_fold != fold:
                    train_indices.extend(class_0_folds[other_fold])
                    train_indices.extend(class_1_folds[other_fold])

            yield train_indices, test_indices

    def _create_folds(self, indices: List[int], n_splits: int) -> List[List[int]]:
        """Create balanced folds from indices."""
        folds = [[] for _ in range(n_splits)]

        for i, idx in enumerate(indices):
            fold_id = i % n_splits
            folds[fold_id].append(idx)

        return folds


class CrossValidator:
    """Cross-validation runner for DCBS evaluation."""

    def __init__(self, config: CrossValidationConfig):
        self.config = config

        if config.stratify:
            self.splitter = StratifiedKFold(
                n_splits=config.n_folds,
                shuffle=config.shuffle,
                random_state=config.random_state,
            )
        else:
            # Simple k-fold without stratification
            self.splitter = self._simple_kfold

    def cross_validate(
        self, eval_config: EvaluationConfig, benchmark_data: List[Dict]
    ) -> Dict:
        """Perform cross-validation evaluation."""
        fold_results = []

        for fold_id, (train_indices, test_indices) in enumerate(
            self.splitter.split(benchmark_data)
        ):
            print(f"Running fold {fold_id + 1}/{self.config.n_folds}")

            # Create test data for this fold
            test_data = [benchmark_data[i] for i in test_indices]

            # Run evaluation on test fold
            runner = EvaluationRunner(eval_config)
            results = runner.run_evaluation(test_data)

            # Extract results
            method_accuracies = {
                method: stats["accuracy"]
                for method, stats in results["statistics"].items()
            }

            method_times = {
                method: stats.get("avg_time_ms", 0)
                for method, stats in results["statistics"].items()
            }

            fold_result = FoldResults(
                fold_id=fold_id,
                train_size=len(train_indices),
                test_size=len(test_indices),
                method_accuracies=method_accuracies,
                method_times=method_times,
            )

            fold_results.append(fold_result)

        return self._aggregate_results(fold_results)

    def _aggregate_results(self, fold_results: List[FoldResults]) -> Dict:
        """Aggregate results across folds."""
        methods = list(fold_results[0].method_accuracies.keys())

        aggregated = {
            "cross_validation_summary": {
                "n_folds": len(fold_results),
                "total_examples": sum(fr.test_size for fr in fold_results),
            },
            "method_statistics": {},
            "fold_details": [],
        }

        # Aggregate statistics for each method
        for method in methods:
            accuracies = [fr.method_accuracies[method] for fr in fold_results]
            times = [fr.method_times[method] for fr in fold_results]

            aggregated["method_statistics"][method] = {
                "mean_accuracy": np.mean(accuracies),
                "std_accuracy": np.std(accuracies),
                "min_accuracy": np.min(accuracies),
                "max_accuracy": np.max(accuracies),
                "accuracy_95_ci": self._calculate_ci(accuracies),
                "mean_time_ms": np.mean(times),
                "std_time_ms": np.std(times),
                "fold_accuracies": accuracies,
            }

        # Store fold details
        for fr in fold_results:
            aggregated["fold_details"].append(
                {
                    "fold_id": fr.fold_id,
                    "train_size": fr.train_size,
                    "test_size": fr.test_size,
                    "accuracies": fr.method_accuracies,
                    "times": fr.method_times,
                }
            )

        return aggregated

    def _calculate_ci(
        self, values: List[float], confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for cross-validation results."""
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)  # Sample standard deviation

        # Use t-distribution for small samples
        from scipy import stats as scipy_stats

        t_value = scipy_stats.t.ppf((1 + confidence) / 2, df=n - 1)
        margin = t_value * std / np.sqrt(n)

        return (mean - margin, mean + margin)

    def _simple_kfold(self, data: List[Dict]) -> Iterator[Tuple[List[int], List[int]]]:
        """Simple k-fold without stratification."""
        n_samples = len(data)
        indices = list(range(n_samples))

        if self.config.shuffle:
            random.seed(self.config.random_state)
            random.shuffle(indices)

        fold_size = n_samples // self.config.n_folds

        for fold in range(self.config.n_folds):
            start = fold * fold_size
            if fold == self.config.n_folds - 1:
                # Last fold gets remaining samples
                end = n_samples
            else:
                end = start + fold_size

            test_indices = indices[start:end]
            train_indices = indices[:start] + indices[end:]

            yield train_indices, test_indices


def run_cross_validation(
    eval_config: EvaluationConfig,
    benchmark_data: List[Dict],
    cv_config: Optional[CrossValidationConfig] = None,
) -> Dict:
    """
    Convenience function to run cross-validation.

    Args:
        eval_config: Evaluation configuration
        benchmark_data: Benchmark dataset
        cv_config: Cross-validation configuration (uses defaults if None)

    Returns:
        Cross-validation results with aggregated statistics
    """
    if cv_config is None:
        cv_config = CrossValidationConfig()

    validator = CrossValidator(cv_config)
    return validator.cross_validate(eval_config, benchmark_data)
