"""
Message template generation for evaluation.

This module handles the creation of chat messages for
reasoning and final answer generation.
"""

from typing import Dict, List


class MessageTemplateGenerator:
    """Generates message templates for LLM interactions."""
    
    def create_reasoning_messages(self, sentence: str, options: List[str]) -> List[Dict[str, str]]:
        """Create a simple, direct prompt for reasoning using a few-shot example."""
        options_str = self._format_options(options)

        # A few-shot prompt provides a high-quality example to guide the model.
        # This is a standard and effective technique for improving CoT reasoning.
        few_shot_example = """
Example Question:
What is the primary function of the mitochondria in an animal cell?
A. To store water
B. To produce energy
C. To control the cell's growth
D. To protect the cell from invaders

Reasoning:
1.  Analyze the question: The question asks for the primary function of mitochondria.
2.  Evaluate the options:
    A. Water storage is primarily handled by vacuoles, which are small in animal cells. This is not the primary function.
    B. Mitochondria are known as the "powerhouses" of the cell. They perform cellular respiration to generate ATP, which is the main energy currency. This is a strong candidate.
    C. The cell's growth and activities are controlled by the nucleus.
    D. Protection is primarily the function of the cell membrane and, in some organisms, the cell wall.
3.  Conclusion: Based on biological principles, the primary function of mitochondria is energy production.

The final answer is B.
"""

        return [
            {
                "role": "system",
                "content": "You are an expert at solving multiple-choice questions. Follow the user's format exactly."
            },
            {
                "role": "user",
                "content": f"{few_shot_example}\n---\nNew Question:\n{sentence}\n\n{options_str}"
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