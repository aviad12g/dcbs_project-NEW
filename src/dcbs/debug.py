"""
Debug utilities for DCBS.

This module provides debug functionality for the DCBS algorithm, including
debug mode resolution, logging, and cluster history tracking.
"""

import os
from typing import List, Optional, Dict, Any

import numpy as np

from .constants import (
    ENV_DCBS_DEBUG_MODE,
    ENV_DCBS_ENABLE_CLUSTER_HISTORY,
    ENV_DCBS_DEBUG_OUTPUT_FILE,
    DEBUG_ENV_TRUE_VALUES,
    DEBUG_ENV_FALSE_VALUES,
)


class DCBSDebugger:
    """Handles debug functionality for DCBS algorithm."""
    
    def __init__(
        self,
        debug_mode: Optional[bool] = None,
        enable_cluster_history: Optional[bool] = None,
        debug_output_file: Optional[str] = None,
    ):
        """
        Initialize the debugger.
        
        Args:
            debug_mode: Enable debug logging (default: False)
            enable_cluster_history: Track cluster decisions (default: False)
            debug_output_file: File path for debug output
        """
        self._debug_mode = self._resolve_debug_mode(debug_mode)
        self._enable_cluster_history = self._resolve_cluster_history(enable_cluster_history)
        self._debug_output_file = debug_output_file or os.environ.get(ENV_DCBS_DEBUG_OUTPUT_FILE)
        
        # Initialize debug tracking
        self._cluster_history = [] if self._enable_cluster_history else None
        self._debug_stats = {"total_samples": 0, "clustering_calls": 0, "cache_hits": 0}
    
    def _resolve_debug_mode(self, debug_mode: Optional[bool]) -> bool:
        """Resolve debug mode from parameter or environment variable."""
        if debug_mode is not None:
            return debug_mode
        
        # Check environment variable
        env_debug = os.environ.get(ENV_DCBS_DEBUG_MODE, "").lower()
        if env_debug in DEBUG_ENV_TRUE_VALUES:
            return True
        elif env_debug in DEBUG_ENV_FALSE_VALUES:
            return False
        
        # Default to False
        return False
    
    def _resolve_cluster_history(self, enable_cluster_history: Optional[bool]) -> bool:
        """Resolve cluster history tracking from parameter or environment variable."""
        if enable_cluster_history is not None:
            return enable_cluster_history
        
        # Check environment variable
        env_history = os.environ.get(ENV_DCBS_ENABLE_CLUSTER_HISTORY, "").lower()
        if env_history in DEBUG_ENV_TRUE_VALUES:
            return True
        elif env_history in DEBUG_ENV_FALSE_VALUES:
            return False
        
        # Default to False (only enable if debug mode is on)
        return self._debug_mode
    
    def log_debug(self, message: str) -> None:
        """Log debug message if debug mode is enabled."""
        if not self._debug_mode:
            return
        
        debug_msg = f"[DCBS DEBUG] {message}"
        
        if self._debug_output_file:
            try:
                with open(self._debug_output_file, "a") as f:
                    f.write(f"{debug_msg}\n")
            except IOError:
                # Fallback to print if file writing fails
                print(debug_msg)
        else:
            print(debug_msg)
    
    def record_cluster_decision(
        self,
        candidate_ids: list,
        labels: np.ndarray,
        selected_cluster: int,
        selected_token: int,
        num_clusters: int,
    ) -> None:
        """Record clustering decision for analysis if enabled."""
        if not self._enable_cluster_history or self._cluster_history is None:
            return
        
        decision = {
            "candidate_count": len(candidate_ids),
            "num_clusters": len(set(labels)),
            "selected_cluster": selected_cluster,
            "selected_token": selected_token,
            "cluster_sizes": [np.sum(labels == i) for i in range(num_clusters)]
        }
        
        self._cluster_history.append(decision)
        
        # Log if debug mode is also enabled
        if self._debug_mode:
            self.log_debug(f"Cluster decision: {decision}")
    
    def increment_stat(self, stat_name: str, amount: int = 1) -> None:
        """Increment a debug statistic."""
        if stat_name in self._debug_stats:
            self._debug_stats[stat_name] += amount
    
    def get_stats(self) -> Dict[str, Any]:
        """Get debugging statistics."""
        stats = self._debug_stats.copy()
        stats["debug_mode"] = self._debug_mode
        stats["cluster_history_enabled"] = self._enable_cluster_history
        stats["cluster_history_count"] = len(self._cluster_history) if self._cluster_history else 0
        return stats
    
    def get_cluster_history(self) -> Optional[List[dict]]:
        """Get cluster decision history if enabled."""
        return self._cluster_history.copy() if self._cluster_history else None
    
    def clear_debug_data(self) -> None:
        """Clear debug data and statistics."""
        if self._cluster_history:
            self._cluster_history.clear()
        self._debug_stats = {"total_samples": 0, "clustering_calls": 0, "cache_hits": 0}
    
    @property
    def debug_mode(self) -> bool:
        """Whether debug mode is enabled."""
        return self._debug_mode
    
    @property
    def cluster_history_enabled(self) -> bool:
        """Whether cluster history tracking is enabled."""
        return self._enable_cluster_history 