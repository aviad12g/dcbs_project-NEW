"""
Message template generation for evaluation.

This module handles the creation of chat messages for
reasoning and final answer generation.
"""

from typing import Dict, List


class MessageTemplateGenerator:
    """Generates message templates for LLM interactions."""
    
    def create_reasoning_messages(self, sentence: str, options: List[str]) -> List[Dict[str, str]]:
        """Create messages for the reasoning step."""
        options_str = self._format_options(options)
        
        # FIXED: Better system prompt that's less likely to be echoed
        # and provides clear instructions for reasoning
        return [
            {
                "role": "system", 
                "content": "You are a helpful assistant. When answering multiple choice questions, first explain your reasoning clearly, then state your final answer."
            },
            {
                "role": "user",
                "content": f"{sentence}\n\n{options_str}\n\nPlease explain your reasoning step by step, then give your final answer."
            }
        ]

    def create_final_answer_messages(
        self, 
        reasoning_messages: List[Dict[str, str]], 
        reasoning_response: str
    ) -> List[Dict[str, str]]:
        """Create messages for the final answer step."""
        # Build on the previous conversation
        messages = reasoning_messages.copy()
        
        # Add the assistant's reasoning response
        messages.append({
            "role": "assistant",
            "content": reasoning_response
        })
        
        # FIXED: More specific final answer prompt
        messages.append({
            "role": "user", 
            "content": "Based on your reasoning above, what is the correct answer? Please respond with just the letter (A, B, C, or D)."
        })
        
        return messages

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