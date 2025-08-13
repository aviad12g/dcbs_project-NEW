"""
Message template generation for evaluation.

This module handles the creation of chat messages for
reasoning and final answer generation.
"""

import os
from typing import Dict, List


class MessageTemplateGenerator:
    """Generates message templates for LLM interactions."""
    
    def __init__(self) -> None:
        # Toggle few-shot exemplars via environment variable to avoid CLI/plumbing churn
        self.enable_few_shot = os.environ.get("DCBS_ENABLE_FEW_SHOT", "0") in ("1", "true", "True")
        self._few_shot_pairs = [
            # Format: (user_message_content, assistant_message_content)
            (
                "Question:\nWater boils at what temperature at sea level?\n\nOptions:\nA. 10°C\nB. 50°C\nC. 100°C\nD. 150°C\nPlease think step by step and then give the final answer.",
                "At sea level water boils at 100°C due to standard atmospheric pressure. The final answer is option C."
            ),
            (
                "Question:\nWhich gas do plants primarily take in for photosynthesis?\n\nOptions:\nA. Oxygen\nB. Nitrogen\nC. Carbon dioxide\nD. Hydrogen\nPlease think step by step and then give the final answer.",
                "Plants use carbon dioxide in photosynthesis to produce glucose. The final answer is option C."
            ),
            (
                "Question:\nWhich organ pumps blood throughout the human body?\n\nOptions:\nA. Liver\nB. Heart\nC. Lung\nD. Kidney\nPlease think step by step and then give the final answer.",
                "Blood is circulated by the heart, which contracts to push blood through vessels. The final answer is option B."
            ),
            (
                "Question:\nWhat force keeps planets in orbit around the Sun?\n\nOptions:\nA. Magnetism\nB. Friction\nC. Gravity\nD. Electricity\nPlease think step by step and then give the final answer.",
                "The Sun's gravity pulls planets inward while their motion keeps them in orbit. The final answer is option C."
            ),
            (
                "Question:\nWhich metal is liquid at room temperature?\n\nOptions:\nA. Mercury\nB. Iron\nC. Aluminum\nD. Copper\nPlease think step by step and then give the final answer.",
                "Mercury is the only common metal liquid at room temperature. The final answer is option A."
            ),
            (
                "Question:\nWhat is the main gas in Earth's atmosphere?\n\nOptions:\nA. Oxygen\nB. Carbon dioxide\nC. Nitrogen\nD. Helium\nPlease think step by step and then give the final answer.",
                "About 78% of Earth's atmosphere is nitrogen. The final answer is option C."
            ),
        ]
    
    def create_reasoning_messages(self, sentence: str, options: List[str]) -> List[Dict[str, str]]:
        """Create a reasoning prompt without few-shot examples (performance-opt mode)."""
        options_str = self._format_options(options)
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": "You are an expert at solving multiple-choice questions. Provide concise chain-of-thought reasoning, then, when asked for the final answer, respond with JUST the single letter (A, B, C, or D) and nothing else."
            }
        ]
        if self.enable_few_shot:
            for u, a in self._few_shot_pairs:
                messages.append({"role": "user", "content": u})
                messages.append({"role": "assistant", "content": a})
        messages.append({
            "role": "user",
            "content": f"Question:\n{sentence}\n\nOptions:\n{options_str}\n\nPlease think step by step and then give the final answer."
        })
        return messages

    def create_final_answer_messages(self, reasoning_messages: List[Dict[str, str]], reasoning: str) -> List[Dict[str, str]]:
        """Compatibility helper used in tests to build final answer prompts."""
        return reasoning_messages + [
            {"role": "assistant", "content": reasoning},
            {"role": "user", "content": "What is your final answer? Respond with just the letter (A, B, C, or D)."},
        ]



    def create_direct_answer_messages(self, sentence: str, options: List[str]) -> List[Dict[str, str]]:
        """Create messages for direct answer without reasoning."""
        options_str = self._format_options(options)
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides direct answers as a single letter (A, B, C, or D)."
            }
        ]
        if self.enable_few_shot:
            # Provide few-shot for direct answers with minimal reasoning and explicit letter
            few_shot_direct = [
                (
                    "Question:\nThe Earth orbits which star?\n\nOptions:\nA. Sirius\nB. Polaris\nC. The Sun\nD. Betelgeuse",
                    "C"
                ),
                (
                    "Question:\nWhich planet is known as the Red Planet?\n\nOptions:\nA. Venus\nB. Mars\nC. Jupiter\nD. Mercury",
                    "B"
                ),
            ]
            for u, a in few_shot_direct:
                messages.append({"role": "user", "content": u})
                messages.append({"role": "assistant", "content": a})
        messages.append({
            "role": "user",
            "content": f"{sentence}\n\n{options_str}"
        })
        return messages

    def _format_options(self, options: List[str]) -> str:
        """Format options with letter labels."""
        formatted = ""
        for i, option in enumerate(options):
            label = chr(ord("A") + i)
            formatted += f"{label}. {option}\n"
        return formatted.strip() 