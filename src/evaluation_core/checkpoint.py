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
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
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
        """Save checkpoint state to disk."""
        checkpoint_path = self.get_checkpoint_path(state.run_id)
        
        try:
            # Save to temporary file first, then atomic rename
            temp_path = checkpoint_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)
            
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