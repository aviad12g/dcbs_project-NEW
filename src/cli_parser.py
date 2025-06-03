"""
Command-line argument parsing for the evaluation framework.

This module provides the ArgumentParserSetup class to handle
command-line argument parsing in a modular way.
"""

import argparse
from typing import List, Optional


class ArgumentParserSetup:
    """Handles command-line argument parsing for the evaluation framework."""

    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        """
        Create and configure the argument parser.
        
        Returns:
            Configured ArgumentParser instance
        """
        parser = argparse.ArgumentParser(
            description="Unified DCBS evaluation framework with advanced features",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        # Core configuration
        parser.add_argument(
            "--config",
            type=str,
            default="configs/study_config.yaml",
            help="Path to YAML configuration file",
        )

        parser.add_argument(
            "--model",
            type=str,
            help="HuggingFace model name or path (overrides config)",
        )

        parser.add_argument(
            "--benchmark",
            type=str,
            help="Path to benchmark JSON file (overrides config)",
        )

        parser.add_argument(
            "--output-dir",
            type=str,
            help="Output directory for results and visualizations (overrides config)",
        )

        # Evaluation parameters
        parser.add_argument(
            "--limit",
            type=int,
            help="Limit number of examples for testing (overrides config)",
        )

        parser.add_argument(
            "--top-p",
            type=float,
            help="Top-p value for nucleus sampling (overrides config)",
        )

        parser.add_argument(
            "--k", type=int, help="Number of clusters for DCBS (overrides config)"
        )

        parser.add_argument(
            "--top-n",
            type=int,
            help="Top-n tokens to consider for DCBS clustering (overrides config)",
        )

        # Clustering method configuration
        parser.add_argument(
            "--clustering-method",
            type=str,
            choices=["kmeans", "dbscan", "hierarchical"],
            default=None,
            help="Clustering method for DCBS (uses config default if not specified)",
        )

        parser.add_argument(
            "--dbscan-eps",
            type=float,
            default=0.3,
            help="DBSCAN epsilon parameter (maximum distance between samples)",
        )

        parser.add_argument(
            "--dbscan-min-samples",
            type=int,
            default=2,
            help="DBSCAN minimum samples in neighborhood",
        )

        parser.add_argument(
            "--hierarchical-linkage",
            type=str,
            choices=["ward", "complete", "average", "single"],
            default="average",
            help="Linkage criterion for hierarchical clustering",
        )

        # Advanced features
        parser.add_argument(
            "--no-cot",
            action="store_true",
            help="Disable chain-of-thought reasoning (overrides config)",
        )

        parser.add_argument(
            "--disable-cache",
            action="store_true",
            help="Disable DCBS caching for performance comparison",
        )

        parser.add_argument(
            "--sweep-k",
            nargs="+",
            type=int,
            help="Sweep over multiple k values for DCBS",
        )

        parser.add_argument(
            "--sweep-top-n",
            nargs="+",
            type=int,
            help="Sweep over multiple top-n values for DCBS",
        )

        parser.add_argument(
            "--sweep-top-p",
            nargs="+",
            type=float,
            help="Sweep over multiple top-p values",
        )

        # Output and logging
        parser.add_argument(
            "--log-level",
            type=str,
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Logging level (overrides config)",
        )

        parser.add_argument(
            "--save-details", 
            action="store_true", 
            help="Save detailed per-example results"
        )

        parser.add_argument(
            "--output-format",
            choices=["json", "csv", "both"],
            default="json",
            help="Output format for results",
        )

        # Performance options
        parser.add_argument(
            "--load-in-4bit", 
            action="store_true", 
            help="Load model with 4-bit quantization"
        )

        parser.add_argument(
            "--batch-size",
            type=int,
            default=5,
            help="Batch size for processing examples",
        )

        parser.add_argument(
            "--memory-profiling",
            action="store_true",
            help="Enable detailed memory profiling",
        )

        # Sampler selection
        parser.add_argument(
            "--samplers",
            nargs="+",
            choices=["greedy", "top_p", "dcbs", "random"],
            help="Specify which samplers to evaluate (default: all)",
        )

        return parser

    @staticmethod
    def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
        """
        Parse command-line arguments.
        
        Args:
            args: Optional list of arguments for testing

        Returns:
            Parsed arguments namespace
        """
        parser = ArgumentParserSetup.create_parser()
        return parser.parse_args(args) 