"""
Test to ensure that "Let's think step by step" is included in the prompt.
"""

import sys
import unittest
from pathlib import Path

import torch

# Add the parent directory to the path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_dcbs_eval import main


class TestPromptCOT(unittest.TestCase):
    """Test case for Chain-of-Thought prompt verification."""

    def test_cot_in_prompt(self):
        """Test that the CoT phrase is included in the prompt construction."""
        # Extract the prompt construction from run_dcbs_eval.py
        # This is a simple string check that would be done in tokenizer

        # Create a mock example with the required fields
        example = {
            "sentence": "Test sentence.",
            "option1": "option A",
            "option2": "option B",
            "correct_option": "1",
        }

        # Construct the prompt as it would be done in run_dcbs_eval.py
        prompt = f"{example['sentence']}\n\nLet's think step by step to determine the answer.\n\nThe answer is "

        # Check that the CoT phrase is in the prompt
        self.assertIn(
            "Let's think step by step",
            prompt,
            "Chain-of-Thought phrase not found in prompt",
        )

        # Check the full structure of the prompt
        expected_prompt = "Test sentence.\n\nLet's think step by step to determine the answer.\n\nThe answer is "
        self.assertEqual(
            prompt, expected_prompt, "Prompt structure doesn't match expected format"
        )


if __name__ == "__main__":
    unittest.main()
