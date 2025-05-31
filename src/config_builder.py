"""
Configuration building and management for the evaluation framework.

This module provides the ConfigBuilder class to handle loading YAML
configuration files and merging them with command-line arguments.
"""

import argparse
from typing import Dict

from src.errors import eval_logger as logger
from src.evaluation_core import EvaluationConfig
from src.config_schema import validate_config_file, ConfigValidator


class ConfigBuilder:
    """Handles configuration loading and merging for the evaluation framework."""

    @staticmethod
    def load_yaml_config(config_path: str) -> Dict:
        """
        Load and validate configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Dictionary containing the validated configuration
            
        Raises:
            ConfigurationError: If file cannot be loaded
            ValidationError: If validation fails
        """
        return validate_config_file(config_path)

    @staticmethod
    def merge_config_with_args(yaml_config: Dict, args: argparse.Namespace) -> EvaluationConfig:
        """
        Create an EvaluationConfig by merging YAML config with command-line arguments.
        
        Args:
            yaml_config: Validated configuration loaded from YAML file
            args: Parsed command-line arguments
            
        Returns:
            EvaluationConfig instance with merged configuration
        """
        # Start with validated YAML config values
        model_name = yaml_config.get("model_path", "meta-llama/Llama-3.2-1B")
        benchmark_path = yaml_config.get("benchmark", "data/arc_easy_full.json")
        output_dir = yaml_config.get("output_dir", "results")
        limit = yaml_config.get("limit")

        # DCBS parameters from validated config
        dcbs_params = yaml_config.get("dcbs_params", {})
        top_p = yaml_config.get("p_values", [0.9])[0] if yaml_config.get("p_values") else 0.9
        k = dcbs_params.get("k", 8)
        top_n = dcbs_params.get("top_n", 50)

        # Clustering parameters (defaults)
        clustering_method = "kmeans"
        dbscan_eps = 0.3
        dbscan_min_samples = 2
        hierarchical_linkage = "average"

        # Other parameters from validated config
        include_cot = yaml_config.get("include_cot", True)
        log_level = yaml_config.get("log_level", "INFO")
        load_in_4bit = yaml_config.get("load_in_4bit", False)
        enable_caching = yaml_config.get("enable_caching", True)

        # Override with command-line arguments using a mapping approach
        arg_overrides = {
            "model": "model_name",
            "benchmark": "benchmark_path", 
            "output_dir": "output_dir",
            "limit": "limit",
            "top_p": "top_p",
            "k": "k",
            "top_n": "top_n",
            "log_level": "log_level",
            "load_in_4bit": "load_in_4bit",
            "clustering_method": "clustering_method",
            "dbscan_eps": "dbscan_eps",
            "dbscan_min_samples": "dbscan_min_samples",
            "hierarchical_linkage": "hierarchical_linkage",
        }

        # Apply overrides from command-line arguments
        config_values = {
            "model_name": model_name,
            "benchmark_path": benchmark_path,
            "output_dir": output_dir,
            "limit": limit,
            "top_p": top_p,
            "k": k,
            "top_n": top_n,
            "log_level": log_level,
            "load_in_4bit": load_in_4bit,
            "clustering_method": clustering_method,
            "dbscan_eps": dbscan_eps,
            "dbscan_min_samples": dbscan_min_samples,
            "hierarchical_linkage": hierarchical_linkage,
        }

        for arg_name, config_key in arg_overrides.items():
            arg_value = getattr(args, arg_name.replace("-", "_"), None)
            if arg_value is not None:
                config_values[config_key] = arg_value
                logger.info(f"Command-line override: {config_key} = {arg_value}")

        # Handle special cases
        if hasattr(args, "no_cot") and args.no_cot:
            include_cot = False
            logger.info("Command-line override: include_cot = False")

        if hasattr(args, "disable_cache") and args.disable_cache:
            enable_caching = False
            logger.info("Command-line override: enable_caching = False")

        # Validate final configuration values
        final_config = EvaluationConfig(
            model_name=config_values["model_name"],
            benchmark_path=config_values["benchmark_path"],
            output_dir=config_values["output_dir"],
            limit=config_values["limit"],
            top_p=config_values["top_p"],
            k=config_values["k"],
            top_n=config_values["top_n"],
            include_cot=include_cot,
            log_level=config_values["log_level"],
            load_in_4bit=config_values["load_in_4bit"],
            enable_caching=enable_caching,
            clustering_method=config_values["clustering_method"],
            dbscan_eps=config_values["dbscan_eps"],
            dbscan_min_samples=config_values["dbscan_min_samples"],
            hierarchical_linkage=config_values["hierarchical_linkage"],
        )

        logger.info(f"Final configuration created: {final_config}")
        return final_config

    @classmethod
    def from_yaml_and_args(cls, config_path: str, args: argparse.Namespace) -> EvaluationConfig:
        """
        Factory method to create EvaluationConfig from YAML file and command-line arguments.
        
        Args:
            config_path: Path to the YAML configuration file
            args: Parsed command-line arguments
            
        Returns:
            EvaluationConfig instance
        """
        yaml_config = cls.load_yaml_config(config_path)
        return cls.merge_config_with_args(yaml_config, args) 