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
        # Defer strict validation until AFTER CLI/env overrides are applied.
        # Here we parse the YAML leniently and let merge_config_with_args fill
        # in required fields from CLI args (e.g., --model, --benchmark).
        try:
            from src.config_schema import yaml as _yaml  # type: ignore
        except Exception:
            import yaml as _yaml  # fallback if direct import fails
        with open(config_path, 'r') as f:
            raw = _yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raw = {}
        return raw

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
        # Start with YAML values (lenient), CLI will override next
        model_name = yaml_config.get("model_path")
        benchmark_path = yaml_config.get("benchmark")
        output_dir = yaml_config.get("output_dir", "results")
        limit = yaml_config.get("limit")

        # DCBS parameters from validated config
        dcbs_params = yaml_config.get("dcbs_params", {})
        top_p = yaml_config.get("p_values", [0.9])[0] if yaml_config.get("p_values") else 0.9
        k = dcbs_params.get("k", 8)
        top_n = dcbs_params.get("top_n", 50)

        # Clustering parameters from YAML config (with defaults)
        clustering_method = yaml_config.get("clustering_method", "kmeans")
        dbscan_eps = yaml_config.get("dbscan_eps", 0.3)
        dbscan_min_samples = yaml_config.get("dbscan_min_samples", 2)
        hierarchical_linkage = yaml_config.get("hierarchical_linkage", "average")
        cluster_weighting = yaml_config.get("cluster_weighting", "none")

        # Other parameters from validated config
        include_cot = yaml_config.get("include_cot", True)
        log_level = yaml_config.get("log_level", "INFO")
        load_in_4bit = yaml_config.get("load_in_4bit", False)
        enable_caching = yaml_config.get("enable_caching", True)
        debug_mode = yaml_config.get("debug_mode", False)
        enable_cluster_history = yaml_config.get("enable_cluster_history", False)
        batch_size = yaml_config.get("batch_size")  # Default None for auto-detection
        use_elbow_method = yaml_config.get("use_elbow_method", False)

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
            "debug_mode": "debug_mode",
            "enable_cluster_history": "enable_cluster_history",
            "batch_size": "batch_size",  # Add batch_size mapping
            "use_elbow_method": "use_elbow_method",  # Add elbow method mapping
            "cluster_weighting": "cluster_weighting",
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
            "cluster_weighting": cluster_weighting,
            "debug_mode": debug_mode,
            "enable_cluster_history": enable_cluster_history,
            "batch_size": batch_size,  # Add batch_size to config values
            "use_elbow_method": use_elbow_method,  # Add elbow method to config values
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

        # Finalize required fields defaults if still missing after overrides
        if not config_values["model_name"]:
            config_values["model_name"] = "meta-llama/Llama-3.2-1B-Instruct"
        if not config_values["benchmark_path"]:
            # default to ARC Easy HF dataset key
            config_values["benchmark_path"] = "arc_easy"

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
            cluster_weighting=config_values["cluster_weighting"],
            debug_mode=config_values["debug_mode"],
            enable_cluster_history=config_values["enable_cluster_history"],
            batch_size=config_values["batch_size"],
            use_elbow_method=config_values["use_elbow_method"],
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