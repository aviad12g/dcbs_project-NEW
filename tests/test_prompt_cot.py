"""
Unit tests for chain-of-thought prompting functionality.

This module tests the chain-of-thought prompt construction and 
reasoning capabilities used in the evaluation framework.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

# Add parent directory to path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation_core.message_templates import MessageTemplateGenerator


class TestChainOfThoughtPrompts(unittest.TestCase):
    """Test chain-of-thought prompting functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = MessageTemplateGenerator()

    def test_cot_prompt_construction(self):
        """Test that chain-of-thought prompts are constructed correctly."""
        sentence = "What color is the sky on a clear day?"
        options = ["red", "blue", "green", "yellow"]
        
        # Test CoT message creation
        cot_messages = self.generator.create_reasoning_messages(sentence, options)
        
        # Verify structure
        self.assertEqual(len(cot_messages), 2)
        self.assertEqual(cot_messages[0]["role"], "system")
        self.assertEqual(cot_messages[1]["role"], "user")
        
        # Verify content contains the question and options
        user_content = cot_messages[1]["content"]
        self.assertIn(sentence, user_content)
        for option in options:
            self.assertIn(option, user_content)
        
        # Verify it asks for reasoning
        self.assertIn("step by step", user_content.lower())

    def test_final_answer_prompt_construction(self):
        """Test that final answer prompts include reasoning."""
        sentence = "What color is the sky on a clear day?"
        options = ["red", "blue", "green", "yellow"]
        reasoning = "The sky appears blue due to light scattering in the atmosphere."
        
        # First create reasoning messages
        reasoning_messages = self.generator.create_reasoning_messages(sentence, options)
        
        # Test final answer message creation
        final_messages = self.generator.create_final_answer_messages(reasoning_messages, reasoning)
        
        # Verify structure: system, user, assistant, user
        self.assertEqual(len(final_messages), 4)
        self.assertEqual(final_messages[0]["role"], "system")
        self.assertEqual(final_messages[1]["role"], "user")
        self.assertEqual(final_messages[2]["role"], "assistant")
        self.assertEqual(final_messages[3]["role"], "user")
        
        # Verify content includes reasoning in assistant message
        assistant_content = final_messages[2]["content"]
        self.assertIn(reasoning, assistant_content)
        
        # Verify initial user message includes question and options
        initial_user_content = final_messages[1]["content"]
        self.assertIn(sentence, initial_user_content)
        for option in options:
            self.assertIn(option, initial_user_content)

    def test_prompt_format_consistency(self):
        """Test that prompts maintain consistent formatting."""
        test_cases = [
            {
                "sentence": "Short question?",
                "options": ["A", "B"]
            },
            {
                "sentence": "This is a much longer question that contains multiple clauses and complex reasoning requirements?",
                "options": ["Option 1", "Option 2", "Option 3", "Option 4"]
            }
        ]
        
        for case in test_cases:
            cot_messages = self.generator.create_reasoning_messages(case["sentence"], case["options"])
            
            # Verify consistent role structure
            self.assertEqual(cot_messages[0]["role"], "system")
            self.assertEqual(cot_messages[1]["role"], "user")
            
            # Verify all content is strings
            self.assertIsInstance(cot_messages[0]["content"], str)
            self.assertIsInstance(cot_messages[1]["content"], str)


if __name__ == "__main__":
    unittest.main()
