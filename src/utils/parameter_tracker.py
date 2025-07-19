"""
Parameter tracking system for DCBS evaluation.

This module provides comprehensive parameter tracking for all samplers,
clustering methods, and evaluation configurations to ensure reproducibility
and complete traceability of experimental results.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import time
import platform
import sys

logger = logging.getLogger(__name__)


@dataclass
class SamplerConfig:
    """Configuration for a specific sampler."""
    name: str
    type: str
    parameters: Dict[str, Any]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ClusteringConfig:
    """Configuration for clustering methods."""
    method: str
    parameters: Dict[str, Any]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class EvaluationMetadata:
    """Metadata about the evaluation environment and setup."""
    timestamp: str
    python_version: str
    platform: str
    hostname: str
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class ParameterTracker:
    """
    Comprehensive parameter tracking for evaluation reproducibility.
    
    This class captures and stores all parameters used in evaluations,
    including sampler configurations, clustering parameters, model settings,
    and evaluation metadata.
    """
    
    def __init__(self):
        """Initialize the parameter tracker."""
        self.sampler_configs: Dict[str, SamplerConfig] = {}
        self.clustering_configs: Dict[str, ClusteringConfig] = {}
        self.model_config: Dict[str, Any] = {}
        self.evaluation_config: Dict[str, Any] = {}
        self.metadata: Optional[EvaluationMetadata] = None
        
        # Initialize metadata
        self._initialize_metadata()
    
    def _initialize_metadata(self) -> None:
        """Initialize evaluation metadata."""
        import socket
        
        self.metadata = EvaluationMetadata(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            python_version=sys.version,
            platform=platform.platform(),
            hostname=socket.gethostname(),
            git_commit=self._get_git_commit(),
            git_branch=self._get_git_branch()
        )
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.debug(f"Could not get git commit: {e}")
        return None
    
    def _get_git_branch(self) -> Optional[str]:
        """Get current git branch."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.debug(f"Could not get git branch: {e}")
        return None
    
    def record_sampler_config(self, sampler_name: str, sampler_type: str, config: Dict[str, Any]) -> None:
        """
        Record complete sampler configuration.
        
        Args:
            sampler_name: Name/identifier of the sampler
            sampler_type: Type of sampler (e.g., 'DCBSSampler', 'GreedySampler')
            config: Dictionary of sampler parameters
        """
        self.sampler_configs[sampler_name] = SamplerConfig(
            name=sampler_name,
            type=sampler_type,
            parameters=config.copy(),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        logger.debug(f"Recorded sampler config for {sampler_name}: {config}")
    
    def record_clustering_config(self, method: str, params: Dict[str, Any]) -> None:
        """
        Record clustering method parameters.
        
        Args:
            method: Clustering method name (e.g., 'dbscan', 'kmeans')
            params: Dictionary of clustering parameters
        """
        self.clustering_configs[method] = ClusteringConfig(
            method=method,
            parameters=params.copy(),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        logger.debug(f"Recorded clustering config for {method}: {params}")
    
    def record_model_config(self, config: Dict[str, Any]) -> None:
        """
        Record model configuration.
        
        Args:
            config: Dictionary of model parameters
        """
        self.model_config = config.copy()
        logger.debug(f"Recorded model config: {config}")
    
    def record_evaluation_config(self, config: Dict[str, Any]) -> None:
        """
        Record evaluation configuration.
        
        Args:
            config: Dictionary of evaluation parameters
        """
        self.evaluation_config = config.copy()
        logger.debug(f"Recorded evaluation config: {config}")
    
    def get_sampler_config(self, sampler_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific sampler.
        
        Args:
            sampler_name: Name of the sampler
            
        Returns:
            Sampler configuration dictionary or None if not found
        """
        if sampler_name in self.sampler_configs:
            return self.sampler_configs[sampler_name].to_dict()
        return None
    
    def get_clustering_config(self, method: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific clustering method.
        
        Args:
            method: Clustering method name
            
        Returns:
            Clustering configuration dictionary or None if not found
        """
        if method in self.clustering_configs:
            return self.clustering_configs[method].to_dict()
        return None
    
    def get_full_configuration(self) -> Dict[str, Any]:
        """
        Return complete configuration for result storage.
        
        Returns:
            Dictionary containing all tracked parameters and metadata
        """
        return {
            'metadata': self.metadata.to_dict() if self.metadata else {},
            'model_config': self.model_config,
            'evaluation_config': self.evaluation_config,
            'sampler_configs': {
                name: config.to_dict() 
                for name, config in self.sampler_configs.items()
            },
            'clustering_configs': {
                method: config.to_dict() 
                for method, config in self.clustering_configs.items()
            }
        }
    
    def validate_configuration(self) -> List[str]:
        """
        Validate that all necessary configurations have been recorded.
        
        Returns:
            List of validation warnings/errors
        """
        warnings = []
        
        if not self.sampler_configs:
            warnings.append("No sampler configurations recorded")
        
        if not self.model_config:
            warnings.append("No model configuration recorded")
        
        if not self.evaluation_config:
            warnings.append("No evaluation configuration recorded")
        
        # Check for common required parameters
        for sampler_name, config in self.sampler_configs.items():
            if config.type == 'DCBSSampler':
                required_params = ['k', 'top_n', 'clustering_method']
                missing_params = [
                    param for param in required_params 
                    if param not in config.parameters
                ]
                if missing_params:
                    warnings.append(
                        f"DCBS sampler '{sampler_name}' missing parameters: {missing_params}"
                    )
        
        return warnings
    
    def clear(self) -> None:
        """Clear all tracked parameters."""
        self.sampler_configs.clear()
        self.clustering_configs.clear()
        self.model_config.clear()
        self.evaluation_config.clear()
        self._initialize_metadata()
        logger.debug("Parameter tracker cleared")


# Global parameter tracker instance
_global_tracker: Optional[ParameterTracker] = None


def get_parameter_tracker() -> ParameterTracker:
    """
    Get the global parameter tracker instance.
    
    Returns:
        Global ParameterTracker instance
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ParameterTracker()
    return _global_tracker


def reset_parameter_tracker() -> None:
    """Reset the global parameter tracker."""
    global _global_tracker
    _global_tracker = None


# Convenience functions for common operations
def record_sampler_config(sampler_name: str, sampler_type: str, config: Dict[str, Any]) -> None:
    """Record sampler configuration using global tracker."""
    get_parameter_tracker().record_sampler_config(sampler_name, sampler_type, config)


def record_clustering_config(method: str, params: Dict[str, Any]) -> None:
    """Record clustering configuration using global tracker."""
    get_parameter_tracker().record_clustering_config(method, params)


def record_model_config(config: Dict[str, Any]) -> None:
    """Record model configuration using global tracker."""
    get_parameter_tracker().record_model_config(config)


def record_evaluation_config(config: Dict[str, Any]) -> None:
    """Record evaluation configuration using global tracker."""
    get_parameter_tracker().record_evaluation_config(config)


def get_full_configuration() -> Dict[str, Any]:
    """Get full configuration using global tracker."""
    return get_parameter_tracker().get_full_configuration()


def validate_configuration() -> List[str]:
    """Validate configuration using global tracker."""
    return get_parameter_tracker().validate_configuration()


def extract_sampler_parameters(sampler) -> Dict[str, Any]:
    """
    Extract parameters from a sampler instance.
    
    Args:
        sampler: Sampler instance
        
    Returns:
        Dictionary of sampler parameters
    """
    params = {}
    sampler_type = type(sampler).__name__
    
    # Extract common parameters based on sampler type
    if hasattr(sampler, 'get_params'):
        # Use get_params method if available
        params = sampler.get_params()
    else:
        # Extract parameters manually based on sampler type
        if sampler_type == 'DCBSSampler':
            if hasattr(sampler, 'clusterer'):
                params['k'] = getattr(sampler.clusterer, 'k', None)
                params['clustering_method'] = type(sampler.clusterer).__name__
            if hasattr(sampler, 'candidate_selector'):
                params['top_n'] = getattr(sampler.candidate_selector, 'top_n', None)
            if hasattr(sampler, 'enable_caching'):
                params['enable_caching'] = sampler.enable_caching
        elif sampler_type == 'TopPSampler':
            params['p'] = getattr(sampler, 'p', None)
        elif sampler_type == 'TemperatureSampler':
            params['temperature'] = getattr(sampler, 'temperature', None)
        elif sampler_type == 'TopKSampler':
            params['k'] = getattr(sampler, 'k', None)
    
    return params


def extract_clustering_parameters(clusterer) -> Dict[str, Any]:
    """
    Extract parameters from a clustering instance.
    
    Args:
        clusterer: Clustering instance
        
    Returns:
        Dictionary of clustering parameters
    """
    params = {}
    clusterer_type = type(clusterer).__name__
    
    if clusterer_type == 'DBSCANClusterer':
        params['eps'] = getattr(clusterer, 'eps', None)
        params['min_samples'] = getattr(clusterer, 'min_samples', None)
        params['metric'] = getattr(clusterer, 'metric', None)
    elif clusterer_type == 'KMeansClusterer':
        params['k'] = getattr(clusterer, 'k', None)
        params['random_seed'] = getattr(clusterer, 'random_seed', None)
        params['enable_adaptive_k'] = getattr(clusterer, 'enable_adaptive_k', None)
    elif clusterer_type == 'HierarchicalClusterer':
        params['k'] = getattr(clusterer, 'k', None)
        params['linkage'] = getattr(clusterer, 'linkage', None)
        params['metric'] = getattr(clusterer, 'metric', None)
    
    return params