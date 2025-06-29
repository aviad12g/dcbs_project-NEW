#!/usr/bin/env python3
"""
Log summarization script for DCBS disagreement analysis.

This script reads JSONL event logs and produces summary tables showing
when DCBS and greedy sampling disagreed and who was correct.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import jsonlines
from tabulate import tabulate


def load_events(events_file: Path) -> List[Dict]:
    """Load events from JSONL file."""
    events = []
    with jsonlines.open(events_file) as reader:
        for event in reader:
            events.append(event)
    return events


def analyze_disagreements(events: List[Dict]) -> Dict:
    """Analyze disagreement events and compute statistics."""
    disagreements = [e for e in events if e.get("event_type") == "token_disagreement"]
    sequence_ends = [e for e in events if e.get("event_type") == "sequence_end"]
    
    # Group disagreements by sequence
    disagreements_by_seq = {}
    for disagreement in disagreements:
        seq_id = disagreement["sequence_id"]
        if seq_id not in disagreements_by_seq:
            disagreements_by_seq[seq_id] = []
        disagreements_by_seq[seq_id].append(disagreement)
    
    # Analyze final outcomes
    outcome_stats = {
        "total_sequences": len(sequence_ends),
        "sequences_with_disagreements": 0,
        "disagree_greedy_correct": 0,
        "disagree_dcbs_correct": 0,
        "disagree_both_correct": 0,
        "disagree_both_wrong": 0,
        "total_token_disagreements": len(disagreements),
    }
    
    sequence_details = []
    
    for seq_end in sequence_ends:
        seq_id = seq_end["sequence_id"]
        disagreement_count = seq_end.get("disagreement_count", 0)
        
        if disagreement_count > 0:
            outcome_stats["sequences_with_disagreements"] += 1
            
            greedy_correct = seq_end["correctness"]["greedy_correct"]
            dcbs_correct = seq_end["correctness"]["dcbs_correct"]
            
            if greedy_correct and not dcbs_correct:
                outcome_stats["disagree_greedy_correct"] += 1
                outcome = "greedy_wins"
            elif dcbs_correct and not greedy_correct:
                outcome_stats["disagree_dcbs_correct"] += 1
                outcome = "dcbs_wins"
            elif greedy_correct and dcbs_correct:
                outcome_stats["disagree_both_correct"] += 1
                outcome = "both_correct"
            else:
                outcome_stats["disagree_both_wrong"] += 1
                outcome = "both_wrong"
            
            sequence_details.append({
                "sequence_id": seq_id,
                "dataset": seq_end.get("dataset", "unknown"),
                "disagreement_count": disagreement_count,
                "outcome": outcome,
                "greedy_answer": seq_end["answers"]["greedy"],
                "dcbs_answer": seq_end["answers"]["dcbs"],
                "correct_answer": seq_end["answers"]["correct"],
            })
    
    return {
        "stats": outcome_stats,
        "sequence_details": sequence_details,
        "disagreements_by_seq": disagreements_by_seq,
    }


def print_summary_table(analysis: Dict) -> None:
    """Print a formatted summary table."""
    stats = analysis["stats"]
    
    print("=" * 60)
    print("DCBS vs Greedy Disagreement Analysis")
    print("=" * 60)
    
    # Basic statistics
    print(f"\nOverall Statistics:")
    print(f"  Total sequences: {stats['total_sequences']}")
    print(f"  Sequences with disagreements: {stats['sequences_with_disagreements']}")
    print(f"  Total token-level disagreements: {stats['total_token_disagreements']}")
    
    if stats['sequences_with_disagreements'] > 0:
        disagreement_rate = stats['sequences_with_disagreements'] / stats['total_sequences'] * 100
        print(f"  Sequence disagreement rate: {disagreement_rate:.1f}%")
        
        avg_disagreements = stats['total_token_disagreements'] / stats['sequences_with_disagreements']
        print(f"  Avg disagreements per sequence: {avg_disagreements:.1f}")
    
    # "Who was right" table
    if stats['sequences_with_disagreements'] > 0:
        print(f"\nWhen sequences had disagreements:")
        
        table_data = [
            ["Greedy was right", stats['disagree_greedy_correct']],
            ["DCBS was right", stats['disagree_dcbs_correct']],
            ["Both were right", stats['disagree_both_correct']],
            ["Both were wrong", stats['disagree_both_wrong']],
        ]
        
        total_disagreed = stats['sequences_with_disagreements']
        table_data_with_pct = []
        for row in table_data:
            count = row[1]
            pct = count / total_disagreed * 100
            table_data_with_pct.append([row[0], count, f"{pct:.1f}%"])
        
        print(tabulate(
            table_data_with_pct,
            headers=["Outcome", "Count", "Percentage"],
            tablefmt="grid"
        ))


def print_detailed_analysis(analysis: Dict, show_details: bool = False) -> None:
    """Print detailed analysis of disagreements."""
    if not show_details:
        return
    
    sequence_details = analysis["sequence_details"]
    
    if not sequence_details:
        print("\nNo sequences with disagreements found.")
        return
    
    print(f"\nDetailed Sequence Analysis:")
    print("=" * 80)
    
    # Group by outcome
    by_outcome = {}
    for seq in sequence_details:
        outcome = seq["outcome"]
        if outcome not in by_outcome:
            by_outcome[outcome] = []
        by_outcome[outcome].append(seq)
    
    for outcome, sequences in by_outcome.items():
        print(f"\n{outcome.replace('_', ' ').title()} ({len(sequences)} sequences):")
        
        table_data = []
        for seq in sequences[:10]:  # Show first 10
            table_data.append([
                seq["sequence_id"][:12] + "...",
                seq["dataset"],
                seq["disagreement_count"],
                seq["greedy_answer"][:20] + "..." if len(seq["greedy_answer"]) > 20 else seq["greedy_answer"],
                seq["dcbs_answer"][:20] + "..." if len(seq["dcbs_answer"]) > 20 else seq["dcbs_answer"],
            ])
        
        if table_data:
            print(tabulate(
                table_data,
                headers=["Sequence ID", "Dataset", "Disagreements", "Greedy Answer", "DCBS Answer"],
                tablefmt="simple"
            ))
        
        if len(sequences) > 10:
            print(f"... and {len(sequences) - 10} more sequences")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Summarize DCBS disagreement logs")
    parser.add_argument("events_file", help="Path to events.jsonl file")
    parser.add_argument("--detailed", action="store_true", help="Show detailed sequence analysis")
    parser.add_argument("--run-dir", help="Run directory (alternative to events_file)")
    
    args = parser.parse_args()
    
    # Determine events file path
    if args.run_dir:
        events_file = Path(args.run_dir) / "events.jsonl"
    else:
        events_file = Path(args.events_file)
    
    if not events_file.exists():
        print(f"Error: Events file not found: {events_file}")
        return 1
    
    # Load and analyze events
    print(f"Loading events from: {events_file}")
    events = load_events(events_file)
    
    if not events:
        print("No events found in file.")
        return 1
    
    print(f"Loaded {len(events)} events")
    
    # Analyze disagreements
    analysis = analyze_disagreements(events)
    
    # Print summary
    print_summary_table(analysis)
    print_detailed_analysis(analysis, args.detailed)
    
    return 0


if __name__ == "__main__":
    exit(main()) 