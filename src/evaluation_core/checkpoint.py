"""
Checkpoint management for evaluation resumption.

This module provides checkpointing functionality to save evaluation progress
and resume from interruptions.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from src.errors import eval_logger as logger
from src.utils.serialization import SerializationUtils, SerializationError


@dataclass
class CheckpointState:
    """State information for checkpointing."""
    run_id: str
    timestamp: str
    total_examples: int
    completed_examples: int
    current_example_idx: int
    sampler_states: Dict[str, Any]
    results: List[Dict]
    config: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization with enhanced type handling."""
        try:
            # Use enhanced serialization to handle numpy types and other non-JSON objects
            return SerializationUtils.convert_to_json_serializable(asdict(self))
        except SerializationError as e:
            logger.error(f"Failed to serialize checkpoint state: {e}")
            # Return minimal state for recovery
            return {
                "serialization_error": str(e),
                "error_type": e.obj_type,
                "error_path": e.obj_path,
                "run_id": getattr(self, 'run_id', None),
                "timestamp": getattr(self, 'timestamp', None),
                "total_examples": getattr(self, 'total_examples', 0),
                "completed_examples": getattr(self, 'completed_examples', 0),
            }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CheckpointState':
        """Create from dictionary loaded from JSON."""
        return cls(**data)


class CheckpointManager:
    """Manages checkpointing for evaluation runs."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints", save_interval: int = 5):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_interval: Save checkpoint every N examples
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.save_interval = save_interval
        self.last_save_time = time.time()
        
    def get_checkpoint_path(self, run_id: str) -> Path:
        """Get checkpoint file path for a run."""
        return self.checkpoint_dir / f"checkpoint_{run_id}.json"
    
    def save_checkpoint(self, state: CheckpointState) -> None:
        """Save checkpoint state to disk with enhanced serialization."""
        checkpoint_path = self.get_checkpoint_path(state.run_id)
        
        try:
            # Convert state to JSON-serializable format
            serializable_state = state.to_dict()
            
            # Save to temporary file first, then atomic rename
            temp_path = checkpoint_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(serializable_state, f, indent=2)
            
            # Atomic rename (with Windows fallback)
            try:
                temp_path.rename(checkpoint_path)
            except OSError as rename_error:
                # Windows file locking issue - use copy + delete fallback
                logger.warning(f"Atomic rename failed ({rename_error}), using copy fallback")
                import shutil
                if checkpoint_path.exists():
                    checkpoint_path.unlink()  # Remove existing file first
                shutil.copy2(temp_path, checkpoint_path)
                temp_path.unlink()  # Remove temp file
            
            logger.info(f"Checkpoint saved: {state.completed_examples}/{state.total_examples} examples")
            self.last_save_time = time.time()
            
        except SerializationError as e:
            logger.error(f"Failed to serialize checkpoint state: {e}")
            logger.error(f"Problematic object type: {e.obj_type} at path: {e.obj_path}")
            
            # Try to save minimal checkpoint for recovery
            try:
                minimal_state = {
                    "serialization_error": str(e),
                    "run_id": state.run_id,
                    "timestamp": state.timestamp,
                    "total_examples": state.total_examples,
                    "completed_examples": state.completed_examples,
                    "current_example_idx": state.current_example_idx,
                }
                
                temp_path = checkpoint_path.with_suffix('.tmp')
                with open(temp_path, 'w') as f:
                    json.dump(minimal_state, f, indent=2)
                temp_path.rename(checkpoint_path)
                
                logger.info(f"Minimal checkpoint saved for recovery: {state.completed_examples}/{state.total_examples} examples")
                
            except Exception as recovery_error:
                logger.error(f"Failed to save even minimal checkpoint: {recovery_error}")
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            # Still log what temp files exist for debugging
            temp_files = list(self.checkpoint_dir.glob(f"checkpoint_{state.run_id}*.tmp"))
            if temp_files:
                logger.info(f"Temporary checkpoint files found: {[f.name for f in temp_files]}")
    
    def load_checkpoint(self, run_id: str) -> Optional[CheckpointState]:
        """Load checkpoint state from disk."""
        checkpoint_path = self.get_checkpoint_path(run_id)
        
        if not checkpoint_path.exists():
            return None
            
        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            
            # Check if this is a recovery checkpoint from serialization error
            if "serialization_error" in data:
                logger.warning(f"Loading recovery checkpoint due to previous serialization error: {data['serialization_error']}")
                # Create minimal checkpoint state from recovery data
                state = CheckpointState(
                    run_id=data.get("run_id", "unknown"),
                    timestamp=data.get("timestamp", "unknown"),
                    total_examples=data.get("total_examples", 0),
                    completed_examples=data.get("completed_examples", 0),
                    current_example_idx=data.get("current_example_idx", 0),
                    sampler_states={},
                    results=[],
                    config={}
                )
            else:
                # Normal checkpoint loading
                state = CheckpointState.from_dict(data)
            
            logger.info(f"Checkpoint loaded: {state.completed_examples}/{state.total_examples} examples")
            return state
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def should_save_checkpoint(self, examples_since_last_save: int) -> bool:
        """Determine if it's time to save a checkpoint."""
        return examples_since_last_save >= self.save_interval
    
    def cleanup_checkpoint(self, run_id: str) -> None:
        """Remove checkpoint file after successful completion."""
        checkpoint_path = self.get_checkpoint_path(run_id)
        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info(f"Checkpoint cleaned up for run {run_id}")
        except Exception as e:
            logger.warning(f"Failed to cleanup checkpoint: {e}")
    
    def list_checkpoints(self) -> List[str]:
        """List available checkpoint run IDs."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.json"))
        run_ids = []
        
        for file in checkpoint_files:
            try:
                run_id = file.stem.replace("checkpoint_", "")
                run_ids.append(run_id)
            except Exception:
                continue
                
        return sorted(run_ids)

    def _configs_match(self, config_a: Dict, config_b: Dict, keys_to_compare: Optional[List[str]] = None) -> bool:
        """Return True if configs are equivalent on a selected subset of keys.

        We deliberately compare only *core* hyper-parameters that define an
        evaluation run so that cosmetic differences (e.g. logging level) do
        not block resumption.  If *keys_to_compare* is None we use a sensible
        default list.
        """
        if config_a is None or config_b is None:
            return False

        if keys_to_compare is None:
            keys_to_compare = [
                "model_name",
                "benchmark_path",
                "k",
                "top_n",
                "top_p",
                "include_cot",
                "clustering_method",
                "dbscan_eps",
                "dbscan_min_samples",
                "hierarchical_linkage",
            ]

        for key in keys_to_compare:
            if config_a.get(key) != config_b.get(key):
                return False
        return True

    def find_latest_matching_checkpoint(self, target_config: Dict) -> Optional[CheckpointState]:
        """Return the *most recent* checkpoint compatible with *target_config*.

        Compatibility is determined by a relaxed config comparison (see
        ``_configs_match``).  Among all compatible checkpoints we pick the one
        with the highest ``completed_examples`` value; if tied we take the one
        with the newest timestamp.
        """
        best_state: Optional[CheckpointState] = None

        for run_id in self.list_checkpoints():
            state = self.load_checkpoint(run_id)
            if not state:
                continue
            if not self._configs_match(target_config, state.config):
                continue

            if best_state is None:
                best_state = state
                continue

            # Prefer checkpoint with more progress, break ties by timestamp
            if state.completed_examples > best_state.completed_examples:
                best_state = state
            elif (
                state.completed_examples == best_state.completed_examples and
                state.timestamp > best_state.timestamp
            ):
                best_state = state

        return best_state 