"""
Centralized logging utility for DCBS disagreement analysis.

This module provides structured logging capabilities that output to both console
and JSONL files for detailed analysis of sampler disagreements.
"""

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jsonlines


class DisagreementLogger:
    """Logger for tracking disagreements between DCBS and greedy sampling."""
    
    def __init__(self, run_id: Optional[str] = None, base_dir: str = "runs"):
        """
        Initialize the disagreement logger.
        
        Args:
            run_id: Unique identifier for this run (auto-generated if None)
            base_dir: Base directory for storing logs
        """
        self.run_id = run_id or self._generate_run_id()
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / self.run_id
        
        # Create run directory
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log files
        self.events_file = self.run_dir / "events.jsonl"
        self.summary_file = self.run_dir / "summary.json"
        
        # Counters for end-of-sequence analysis
        self.counters = {
            "total_sequences": 0,
            "total_disagreements": 0,
            "disagree_greedy_correct": 0,
            "disagree_dcbs_correct": 0,
            "disagree_both_wrong": 0,
            "disagree_both_correct": 0,
        }
        
        # Initialize events file
        self._write_run_metadata()
    
    def _generate_run_id(self) -> str:
        """Generate a unique run ID based on timestamp and UUID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"{timestamp}_{short_uuid}"
    
    def _write_run_metadata(self) -> None:
        """Write initial metadata about this run."""
        metadata = {
            "event_type": "run_start",
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "version": "1.0",
        }
        
        with jsonlines.open(self.events_file, mode='a') as writer:
            writer.write(metadata)
    
    def log_disagreement(
        self,
        sequence_id: str,
        timestep: int,
        context_tokens: List[int],
        context_text: str,
        greedy_token: int,
        greedy_prob: float,
        dcbs_token: int,
        dcbs_prob: float,
        cluster_id: Optional[int] = None,
        cluster_centroid: Optional[List[float]] = None,
        top_k_probs: Optional[Dict[int, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a disagreement between DCBS and greedy sampling.
        
        Args:
            sequence_id: Unique identifier for this sequence
            timestep: Position in the sequence where disagreement occurred
            context_tokens: Full context window as token IDs
            context_text: Human-readable context text
            greedy_token: Token chosen by greedy sampling
            greedy_prob: Probability of greedy token
            dcbs_token: Token chosen by DCBS
            dcbs_prob: Probability of DCBS token
            cluster_id: ID of cluster chosen by DCBS
            cluster_centroid: Centroid of chosen cluster
            top_k_probs: Top-k token probabilities (optional)
            metadata: Additional metadata
        """
        event = {
            "event_type": "token_disagreement",
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "sequence_id": sequence_id,
            "timestep": timestep,
            "context": {
                "tokens": context_tokens,
                "text": context_text,
                "length": len(context_tokens),
            },
            "greedy": {
                "token": greedy_token,
                "probability": greedy_prob,
            },
            "dcbs": {
                "token": dcbs_token,
                "probability": dcbs_prob,
                "cluster_id": cluster_id,
                "cluster_centroid": cluster_centroid,
            },
            "top_k_probs": top_k_probs or {},
            "metadata": metadata or {},
        }
        
        with jsonlines.open(self.events_file, mode='a') as writer:
            writer.write(event)
        
        self.counters["total_disagreements"] += 1
    
    def log_sequence_end(
        self,
        sequence_id: str,
        greedy_answer: str,
        dcbs_answer: str,
        correct_answer: str,
        greedy_correct: bool,
        dcbs_correct: bool,
        total_disagreements: int,
        dataset: Optional[str] = None,
        question: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log the end of a sequence with final answer comparison.
        
        Args:
            sequence_id: Unique identifier for this sequence
            greedy_answer: Final answer from greedy sampling
            dcbs_answer: Final answer from DCBS
            correct_answer: Ground truth answer
            greedy_correct: Whether greedy answer was correct
            dcbs_correct: Whether DCBS answer was correct
            total_disagreements: Number of disagreements in this sequence
            dataset: Dataset name (e.g., "arc_easy")
            question: The question text
            metadata: Additional metadata
        """
        event = {
            "event_type": "sequence_end",
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "sequence_id": sequence_id,
            "dataset": dataset,
            "question": question,
            "answers": {
                "greedy": greedy_answer,
                "dcbs": dcbs_answer,
                "correct": correct_answer,
            },
            "correctness": {
                "greedy_correct": greedy_correct,
                "dcbs_correct": dcbs_correct,
            },
            "disagreement_count": total_disagreements,
            "metadata": metadata or {},
        }
        
        with jsonlines.open(self.events_file, mode='a') as writer:
            writer.write(event)
        
        # Update counters
        self.counters["total_sequences"] += 1
        
        if total_disagreements > 0:  # Only count if there were disagreements
            if greedy_correct and not dcbs_correct:
                self.counters["disagree_greedy_correct"] += 1
            elif dcbs_correct and not greedy_correct:
                self.counters["disagree_dcbs_correct"] += 1
            elif greedy_correct and dcbs_correct:
                self.counters["disagree_both_correct"] += 1
            else:
                self.counters["disagree_both_wrong"] += 1
    
    def save_summary(self) -> None:
        """Save summary statistics to JSON file."""
        summary = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "counters": self.counters,
            "analysis": self._compute_analysis(),
        }
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _compute_analysis(self) -> Dict[str, Any]:
        """Compute analysis metrics from counters."""
        total_disagreed = (
            self.counters["disagree_greedy_correct"] +
            self.counters["disagree_dcbs_correct"] +
            self.counters["disagree_both_wrong"] +
            self.counters["disagree_both_correct"]
        )
        
        if total_disagreed == 0:
            return {"note": "No sequences with disagreements found"}
        
        return {
            "disagreement_rate": self.counters["total_disagreements"] / max(1, self.counters["total_sequences"]),
            "when_disagreed": {
                "greedy_correct_rate": self.counters["disagree_greedy_correct"] / total_disagreed,
                "dcbs_correct_rate": self.counters["disagree_dcbs_correct"] / total_disagreed,
                "both_correct_rate": self.counters["disagree_both_correct"] / total_disagreed,
                "both_wrong_rate": self.counters["disagree_both_wrong"] / total_disagreed,
            },
            "total_sequences_with_disagreements": total_disagreed,
        }
    
    def get_run_dir(self) -> Path:
        """Get the run directory path."""
        return self.run_dir
    
    def get_events_file(self) -> Path:
        """Get the events JSONL file path."""
        return self.events_file


def create_disagreement_logger(run_id: Optional[str] = None, base_dir: str = "runs") -> DisagreementLogger:
    """
    Factory function to create a disagreement logger.
    
    Args:
        run_id: Unique identifier for this run (auto-generated if None)
        base_dir: Base directory for storing logs
        
    Returns:
        Configured DisagreementLogger instance
    """
    return DisagreementLogger(run_id=run_id, base_dir=base_dir) 