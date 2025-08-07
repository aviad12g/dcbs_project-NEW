"""
Message template generation for evaluation.

This module handles the creation of chat messages for
reasoning and final answer generation.
"""

from typing import Dict, List


class MessageTemplateGenerator:
    """Generates message templates for LLM interactions."""
    
    def create_reasoning_messages(self, sentence: str, options: List[str]) -> List[Dict[str, str]]:
        """Create a reasoning prompt without few-shot examples (performance-opt mode)."""
        options_str = self._format_options(options)

        return [
            {
                "role": "system",
                "content": "You are an expert at solving multiple-choice questions. Provide step-by-step reasoning, then state the final answer letter (A, B, C, or D)."
            },
            {
                "role": "user",
                "content": f"Question:\n{sentence}\n\nOptions:\n{options_str}\n\nPlease think step-by-step and then give the final answer."
            }
        ]



    def create_direct_answer_messages(self, sentence: str, options: List[str]) -> List[Dict[str, str]]:
        """Create messages for direct answer without reasoning."""
        options_str = self._format_options(options)
        
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides direct answers."
            },
            {
                "role": "user", 
                "content": f"{sentence}\n\n{options_str}"
            }
        ]

    def _format_options(self, options: List[str]) -> str:
        """Format options with letter labels."""
        formatted = ""
        for i, option in enumerate(options):
            label = chr(ord("A") + i)
            formatted += f"{label}. {option}\n"
        return formatted.strip() 