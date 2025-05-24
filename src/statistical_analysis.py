"""
Advanced statistical analysis for DCBS evaluation results.

This module provides sophisticated statistical tests, effect size calculations,
and power analysis for comparing sampling methods.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as stats


@dataclass
class EffectSize:
    """Effect size calculation results."""

    cohens_d: float
    interpretation: str
    confidence_interval: Tuple[float, float]


@dataclass
class StatisticalTest:
    """Statistical test results."""

    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[EffectSize]
    interpretation: str
    significant: bool


class AdvancedStatistics:
    """Advanced statistical analysis for sampling method comparison."""

    @staticmethod
    def mcnemar_test(
        method1_results: List[bool], method2_results: List[bool]
    ) -> StatisticalTest:
        """
        McNemar's test for paired binary outcomes.
        More appropriate than chi-square for comparing dependent samples.
        """
        # Create contingency table
        both_correct = sum(
            1 for a, b in zip(method1_results, method2_results) if a and b
        )
        method1_only = sum(
            1 for a, b in zip(method1_results, method2_results) if a and not b
        )
        method2_only = sum(
            1 for a, b in zip(method1_results, method2_results) if not a and b
        )
        both_wrong = sum(
            1 for a, b in zip(method1_results, method2_results) if not a and not b
        )

        # McNemar's test statistic
        if method1_only + method2_only == 0:
            return StatisticalTest(
                test_name="McNemar's Test",
                statistic=0.0,
                p_value=1.0,
                effect_size=None,
                interpretation="No discordant pairs - methods perform identically",
                significant=False,
            )

        # Continuity correction for small samples
        statistic = (abs(method1_only - method2_only) - 1) ** 2 / (
            method1_only + method2_only
        )
        p_value = 1 - stats.chi2.cdf(statistic, df=1)

        # Effect size (odds ratio)
        if method2_only == 0:
            odds_ratio = float("inf")
        else:
            odds_ratio = method1_only / method2_only

        interpretation = f"Odds ratio: {odds_ratio:.3f}"
        if p_value < 0.05:
            interpretation += " (statistically significant difference)"

        return StatisticalTest(
            test_name="McNemar's Test",
            statistic=statistic,
            p_value=p_value,
            effect_size=None,
            interpretation=interpretation,
            significant=p_value < 0.05,
        )

    @staticmethod
    def cohens_d_proportions(p1: float, n1: int, p2: float, n2: int) -> EffectSize:
        """
        Calculate Cohen's d for proportions with confidence interval.
        """
        # Pooled standard deviation for proportions
        pooled_p = (p1 * n1 + p2 * n2) / (n1 + n2)
        pooled_sd = np.sqrt(pooled_p * (1 - pooled_p))

        if pooled_sd == 0:
            cohens_d = 0.0
        else:
            cohens_d = (p1 - p2) / pooled_sd

        # Confidence interval for Cohen's d
        se_d = np.sqrt((n1 + n2) / (n1 * n2) + cohens_d**2 / (2 * (n1 + n2)))
        ci_lower = cohens_d - 1.96 * se_d
        ci_upper = cohens_d + 1.96 * se_d

        # Interpretation
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        return EffectSize(
            cohens_d=cohens_d,
            interpretation=interpretation,
            confidence_interval=(ci_lower, ci_upper),
        )

    @staticmethod
    def bootstrap_confidence_interval(
        data: List[float], n_bootstrap: int = 10000, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Bootstrap confidence interval for any statistic.
        More robust than normal approximation for small samples.
        """
        bootstrap_stats = []
        n = len(data)

        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(np.mean(bootstrap_sample))

        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        return (
            np.percentile(bootstrap_stats, lower_percentile),
            np.percentile(bootstrap_stats, upper_percentile),
        )

    @staticmethod
    def power_analysis(
        effect_size: float, alpha: float = 0.05, power: float = 0.8
    ) -> int:
        """
        Calculate required sample size for given effect size and power.
        """
        from scipy.stats import norm

        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)

        # For two-proportion z-test
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

        return int(np.ceil(n))

    @staticmethod
    def multiple_comparisons_correction(
        p_values: List[float], method: str = "bonferroni"
    ) -> List[float]:
        """
        Apply multiple comparisons correction.
        """
        if method == "bonferroni":
            return [min(p * len(p_values), 1.0) for p in p_values]
        elif method == "holm":
            # Holm-Bonferroni method
            sorted_indices = np.argsort(p_values)
            corrected = [0.0] * len(p_values)

            for i, idx in enumerate(sorted_indices):
                corrected[idx] = min(p_values[idx] * (len(p_values) - i), 1.0)
                if i > 0:
                    corrected[idx] = max(
                        corrected[idx], corrected[sorted_indices[i - 1]]
                    )

            return corrected
        else:
            raise ValueError(f"Unknown correction method: {method}")


def comprehensive_statistical_analysis(results: Dict) -> Dict:
    """
    Perform comprehensive statistical analysis on evaluation results.
    """
    statistics = results["statistics"]
    methods = list(statistics.keys())

    analysis = {
        "pairwise_comparisons": {},
        "effect_sizes": {},
        "power_analysis": {},
        "recommendations": [],
    }

    # Pairwise comparisons
    for i, method1 in enumerate(methods):
        for method2 in methods[i + 1 :]:
            stats1 = statistics[method1]
            stats2 = statistics[method2]

            # McNemar's test (if we had individual results)
            # For now, use two-proportion z-test
            p1 = stats1["accuracy"] / 100
            n1 = stats1["total"]
            p2 = stats2["accuracy"] / 100
            n2 = stats2["total"]

            # Two-proportion z-test
            pooled_p = (stats1["correct"] + stats2["correct"]) / (n1 + n2)
            se = np.sqrt(pooled_p * (1 - pooled_p) * (1 / n1 + 1 / n2))

            if se > 0:
                z_stat = (p1 - p2) / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            else:
                z_stat = 0
                p_value = 1.0

            # Effect size
            effect_size = AdvancedStatistics.cohens_d_proportions(p1, n1, p2, n2)

            comparison_key = f"{method1}_vs_{method2}"
            analysis["pairwise_comparisons"][comparison_key] = {
                "z_statistic": z_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "effect_size": effect_size.cohens_d,
                "effect_interpretation": effect_size.interpretation,
            }

            analysis["effect_sizes"][comparison_key] = effect_size

    # Multiple comparisons correction
    p_values = [comp["p_value"] for comp in analysis["pairwise_comparisons"].values()]
    corrected_p = AdvancedStatistics.multiple_comparisons_correction(p_values, "holm")

    for i, (key, comp) in enumerate(analysis["pairwise_comparisons"].items()):
        comp["corrected_p_value"] = corrected_p[i]
        comp["significant_corrected"] = corrected_p[i] < 0.05

    # Power analysis recommendations
    for method in methods:
        stats_method = statistics[method]
        current_n = stats_method["total"]

        # Calculate power for detecting small, medium, large effects
        for effect_name, effect_size in [
            ("small", 0.2),
            ("medium", 0.5),
            ("large", 0.8),
        ]:
            required_n = AdvancedStatistics.power_analysis(effect_size)
            analysis["power_analysis"][f"{method}_{effect_name}_effect"] = {
                "current_n": current_n,
                "required_n": required_n,
                "adequately_powered": current_n >= required_n,
            }

    # Generate recommendations
    best_method = max(methods, key=lambda m: statistics[m]["accuracy"])
    analysis["recommendations"].append(f"Best performing method: {best_method}")

    significant_differences = [
        key
        for key, comp in analysis["pairwise_comparisons"].items()
        if comp["significant_corrected"]
    ]

    if significant_differences:
        analysis["recommendations"].append(
            f"Statistically significant differences found: {', '.join(significant_differences)}"
        )
    else:
        analysis["recommendations"].append(
            "No statistically significant differences after multiple comparisons correction"
        )

    return analysis
