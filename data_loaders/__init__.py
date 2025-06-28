"""
Dataset registry for DCBS evaluation.

This module provides a centralized registry for loading and accessing
various evaluation datasets using the HuggingFace datasets library.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from datasets import load_dataset as hf_load_dataset


class DatasetRegistry:
    """Registry for evaluation datasets with standardized interface."""
    
    DATASETS = {
        "arc_easy": {
            "name": "allenai/ai2_arc",
            "config": "ARC-Easy", 
            "split": "test",
            "description": "ARC Easy - grade-school science questions",
            "question_field": "question",
            "choices_field": "choices",
            "answer_field": "answerKey",
        },
        "arc_challenge": {
            "name": "allenai/ai2_arc",
            "config": "ARC-Challenge",
            "split": "test", 
            "description": "ARC Challenge - harder grade-school science questions",
            "question_field": "question",
            "choices_field": "choices",
            "answer_field": "answerKey",
        },
        "hellaswag": {
            "name": "hellaswag",
            "config": None,
            "split": "validation",
            "description": "HellaSwag - commonsense reasoning about physical situations",
            "question_field": "ctx",
            "choices_field": "endings",
            "answer_field": "label",
        },
        "mmlu_stem": {
            "name": "cais/mmlu",
            "config": "all",
            "split": "test",
            "description": "MMLU STEM - college-level STEM questions",
            "question_field": "question",
            "choices_field": "choices", 
            "answer_field": "answer",
            "filter_subjects": [
                "abstract_algebra", "anatomy", "astronomy", "college_biology",
                "college_chemistry", "college_computer_science", "college_mathematics",
                "college_physics", "computer_security", "conceptual_physics",
                "electrical_engineering", "elementary_mathematics", "high_school_biology",
                "high_school_chemistry", "high_school_computer_science", "high_school_mathematics", 
                "high_school_physics", "high_school_statistics", "machine_learning",
                "medical_genetics", "nutrition", "professional_medicine", "virology"
            ]
        }
    }
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all available dataset keys."""
        return list(cls.DATASETS.keys())
    
    @classmethod
    def get_dataset_info(cls, dataset_key: str) -> Dict:
        """Get metadata about a dataset."""
        if dataset_key not in cls.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_key}. Available: {cls.list_datasets()}")
        return cls.DATASETS[dataset_key].copy()
    
    @classmethod
    def load_dataset_raw(cls, dataset_key: str, limit: Optional[int] = None) -> Dict:
        """
        Load raw dataset from HuggingFace.
        
        Args:
            dataset_key: Dataset identifier
            limit: Maximum number of examples to load
            
        Returns:
            Raw dataset object
        """
        info = cls.get_dataset_info(dataset_key)
        
        # Load dataset
        if info["config"]:
            dataset = hf_load_dataset(info["name"], info["config"], trust_remote_code=True)[info["split"]]
        else:
            dataset = hf_load_dataset(info["name"], trust_remote_code=True)[info["split"]]
        
        # Apply STEM filtering for MMLU
        if dataset_key == "mmlu_stem" and "filter_subjects" in info:
            dataset = dataset.filter(lambda x: x["subject"] in info["filter_subjects"])
        
        # Apply limit
        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))
            
        return dataset
    
    @classmethod 
    def load_dataset_standardized(
        cls, 
        dataset_key: str, 
        limit: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Load dataset in standardized format for evaluation.
        
        Args:
            dataset_key: Dataset identifier
            limit: Maximum number of examples to load
            save_path: Optional path to save processed dataset
            
        Returns:
            List of standardized examples
        """
        info = cls.get_dataset_info(dataset_key)
        raw_dataset = cls.load_dataset_raw(dataset_key, limit)
        
        standardized = []
        
        for i, example in enumerate(raw_dataset):
            try:
                std_example = cls._standardize_example(example, info, dataset_key, i)
                standardized.append(std_example)
            except Exception as e:
                print(f"Warning: Failed to process example {i} from {dataset_key}: {e}")
                continue
        
        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(standardized, f, indent=2)
            print(f"Saved {len(standardized)} examples to {save_path}")
        
        return standardized
    
    @classmethod
    def _standardize_example(cls, example: Dict, info: Dict, dataset_key: str, index: int) -> Dict:
        """Convert raw example to standardized format."""
        # Extract fields
        question = example[info["question_field"]]
        answer_field = example[info["answer_field"]]
        
        # Handle choices
        if info["choices_field"] == "choices" and isinstance(example["choices"], dict):
            # ARC format: {"text": [...], "label": [...]}
            choices = example["choices"]["text"]
        elif info["choices_field"] == "endings":
            # HellaSwag format: list of strings
            choices = example["endings"]
        else:
            # Direct list format
            choices = example[info["choices_field"]]
        
        # Handle answer format
        if isinstance(answer_field, str):
            if answer_field.isdigit():
                # Numeric string format (0, 1, 2, 3)
                correct_option = answer_field
            else:
                # Letter format (A, B, C, D)
                correct_option = str(ord(answer_field.upper()) - ord('A'))
        elif isinstance(answer_field, int):
            # Numeric format (0, 1, 2, 3)
            correct_option = str(answer_field)
        else:
            correct_option = str(answer_field)
        
        # Create standardized format
        # Validate correct_option
        correct_idx = int(correct_option)
        if correct_idx >= len(choices):
            raise ValueError(f"Correct option {correct_idx} out of range for {len(choices)} choices")
        
        std_example = {
            "id": f"{dataset_key}_{index}",
            "dataset": dataset_key,
            "question": question,
            "choices": choices,
            "correct_option": correct_option,
            "correct_answer": choices[correct_idx],
        }
        
        # Add dataset-specific metadata
        if dataset_key == "mmlu_stem":
            std_example["subject"] = example.get("subject", "unknown")
        elif dataset_key == "hellaswag":
            std_example["activity_label"] = example.get("activity_label", "unknown")
            std_example["ctx_a"] = example.get("ctx_a", "")
            std_example["ctx_b"] = example.get("ctx_b", "")
        
        return std_example


def load_dataset(dataset_key: str, limit: Optional[int] = None) -> List[Dict]:
    """
    Convenience function to load a dataset in standardized format.
    
    Args:
        dataset_key: Dataset identifier (arc_easy, arc_challenge, hellaswag, mmlu_stem)
        limit: Maximum number of examples to load
        
    Returns:
        List of standardized examples
    """
    return DatasetRegistry.load_dataset_standardized(dataset_key, limit)


def load_multiple_datasets(
    dataset_keys: Union[str, List[str]], 
    limit_per_dataset: Optional[int] = None
) -> Dict[str, List[Dict]]:
    """
    Load multiple datasets.
    
    Args:
        dataset_keys: Dataset keys or "all" for all datasets
        limit_per_dataset: Limit per individual dataset
        
    Returns:
        Dictionary mapping dataset keys to standardized examples
    """
    if dataset_keys == "all":
        dataset_keys = DatasetRegistry.list_datasets()
    elif isinstance(dataset_keys, str):
        dataset_keys = [dataset_keys]
    
    results = {}
    for key in dataset_keys:
        print(f"Loading {key}...")
        try:
            results[key] = load_dataset(key, limit_per_dataset)
            print(f"  Loaded {len(results[key])} examples from {key}")
        except Exception as e:
            print(f"  Failed to load {key}: {e}")
            results[key] = []
    
    return results


# Convenience aliases
load_arc_easy = lambda limit=None: load_dataset("arc_easy", limit)
load_arc_challenge = lambda limit=None: load_dataset("arc_challenge", limit) 
load_hellaswag = lambda limit=None: load_dataset("hellaswag", limit)
load_mmlu_stem = lambda limit=None: load_dataset("mmlu_stem", limit) 