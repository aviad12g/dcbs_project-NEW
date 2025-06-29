"""
Multi-dataset evaluation runner with disagreement tracking.

This module extends the existing EvaluationRunner to support multiple datasets
and optional disagreement tracking between DCBS and greedy sampling.
"""

import time
from typing import Dict, List, Optional

try:
    from data_loaders import load_dataset
except ImportError:
    # Fallback for when data_loaders is not in path
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from data_loaders import load_dataset
from src.errors import eval_logger as logger
from src.logger import create_disagreement_logger
from src.evaluation_core.config import EvaluationConfig
from src.evaluation_core.runner import EvaluationRunner
from src.evaluation_core.disagreement_tracker import DisagreementTracker, DisagreementAwareQuestionAnswerer


class MultiDatasetEvaluationRunner:
    """Evaluation runner that supports multiple datasets with disagreement tracking."""
    
    def __init__(
        self, 
        config: EvaluationConfig, 
        requested_samplers: List[str] = None,
        enable_disagreement_tracking: bool = False,
        run_id: Optional[str] = None,
    ):
        """
        Initialize multi-dataset evaluation runner.
        
        Args:
            config: Evaluation configuration
            requested_samplers: List of samplers to evaluate
            enable_disagreement_tracking: Whether to enable disagreement tracking
            run_id: Optional run identifier for disagreement logs
        """
        self.config = config
        self.requested_samplers = requested_samplers
        self.enable_disagreement_tracking = enable_disagreement_tracking
        self.run_id = run_id
        
        # Initialize disagreement logger if enabled
        self.disagreement_logger = None
        if self.enable_disagreement_tracking:
            self.disagreement_logger = create_disagreement_logger(run_id=self.run_id)
            logger.info(f"Disagreement tracking enabled. Run ID: {self.disagreement_logger.run_id}")
            logger.info(f"Logs will be saved to: {self.disagreement_logger.get_run_dir()}")
    
    def run_evaluation(self, datasets: List[str] = None) -> Dict:
        """
        Run evaluation across multiple datasets.
        
        Args:
            datasets: List of dataset names or None to use config default
            
        Returns:
            Dictionary containing results for all datasets
        """
        if not datasets:
            datasets = getattr(self.config, 'datasets', ['arc_easy'])
        
        # Handle "all" dataset selection
        if datasets == ["all"]:
            datasets = ["arc_easy", "arc_challenge", "hellaswag", "mmlu_stem"]
        
        logger.info(f"Running evaluation on datasets: {datasets}")
        
        all_results = {}
        
        for dataset_name in datasets:
            logger.info(f"\n" + "="*60)
            logger.info(f"EVALUATING DATASET: {dataset_name.upper()}")
            logger.info(f"="*60)
            
            try:
                # Load dataset
                dataset_data = self._load_dataset(dataset_name)
                if not dataset_data:
                    logger.warning(f"Skipping empty dataset: {dataset_name}")
                    continue
                
                # Run evaluation for this dataset
                dataset_results = self._evaluate_single_dataset(dataset_name, dataset_data)
                all_results[dataset_name] = dataset_results
                
                # Save intermediate results
                self._save_dataset_results(dataset_name, dataset_results)
                
            except Exception as e:
                logger.error(f"Failed to evaluate dataset {dataset_name}: {e}")
                all_results[dataset_name] = {"error": str(e)}
                continue
        
        # Save final summary if disagreement tracking is enabled
        if self.disagreement_logger:
            self.disagreement_logger.save_summary()
            logger.info(f"Disagreement analysis saved to: {self.disagreement_logger.get_run_dir()}")
        
        # Create combined results
        combined_results = {
            "datasets": all_results,
            "summary": self._create_summary(all_results),
            "evaluation_completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "datasets": datasets,
                "model": self.config.model_name,
                "samplers": self.requested_samplers,
                "disagreement_tracking": self.enable_disagreement_tracking,
                "run_id": self.run_id,
            }
        }
        
        return combined_results
    
    def _load_dataset(self, dataset_name: str) -> List[Dict]:
        """Load and standardize a single dataset."""
        try:
            # Load dataset with limit if specified
            limit = self.config.limit
            dataset_data = load_dataset(dataset_name, limit)
            
            # Convert new dataset format to existing evaluation format
            converted_data = []
            for example in dataset_data:
                converted_example = {
                    "id": example["id"],
                    "question": example["question"],
                    "options": example["choices"],  # Convert choices -> options
                    "correct_option": str(int(example["correct_option"]) + 1),  # Convert 0-based to 1-based
                    "correct_answer": example["correct_answer"],
                }
                # Add dataset-specific metadata
                if "subject" in example:
                    converted_example["subject"] = example["subject"]
                if "activity_label" in example:
                    converted_example["activity_label"] = example["activity_label"]
                
                converted_data.append(converted_example)
            
            logger.info(f"Loaded {len(converted_data)} examples from {dataset_name}")
            return converted_data
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            return []
    
    def _evaluate_single_dataset(self, dataset_name: str, dataset_data: List[Dict]) -> Dict:
        """Evaluate a single dataset."""
        # Create standard evaluation runner
        runner = EvaluationRunner(self.config, self.requested_samplers)
        
        # Standard evaluation for now (disagreement tracking integration can be added later)
        return runner.run_evaluation(dataset_data)
    
    def _save_dataset_results(self, dataset_name: str, results: Dict) -> None:
        """Save results for a single dataset."""
        if self.disagreement_logger:
            output_file = self.disagreement_logger.get_run_dir() / f"{dataset_name}_results.json"
            import json
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Dataset results saved to: {output_file}")
    
    def _create_summary(self, all_results: Dict) -> Dict:
        """Create summary statistics across all datasets."""
        summary = {
            "total_datasets": len(all_results),
            "successful_datasets": len([r for r in all_results.values() if "error" not in r]),
            "by_dataset": {},
        }
        
        for dataset_name, results in all_results.items():
            if "error" in results:
                summary["by_dataset"][dataset_name] = {"status": "error", "error": results["error"]}
            else:
                stats = results.get("statistics", {})
                summary["by_dataset"][dataset_name] = {
                    "status": "success",
                    "samplers": {
                        sampler: f"{data.get('accuracy', 0):.1f}% ({data.get('correct', 0)}/{data.get('total', 0)})"
                        for sampler, data in stats.items()
                    }
                }
        
        return summary 