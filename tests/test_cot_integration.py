"""
Integration tests for CoT parser with question answerer.

This module tests that the CoT parser is properly integrated
into the question answering pipeline.
"""

import unittest
from unittest.mock import MagicMock, patch

from src.evaluation_core.question_answerer import QuestionAnswerer
from src.dcbs import SamplingContext


class TestCoTIntegration(unittest.TestCase):
    """Test CoT parser integration with question answerer."""

    def setUp(self):
        """Set up test environment."""
        # Create mock model and tokenizer
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.chat_template = None  # No chat template
        
        # Create mock context
        self.mock_context = SamplingContext(
            embedding_layer=None,
            tokenizer=self.mock_tokenizer,
            device=None
        )
        
        # Create question answerer
        self.question_answerer = QuestionAnswerer(
            self.mock_model, self.mock_tokenizer, self.mock_context
        )

    def test_cot_parser_initialization(self):
        """Test that CoT parser is properly initialized."""
        self.assertIsNotNone(self.question_answerer.cot_parser)
        self.assertEqual(
            type(self.question_answerer.cot_parser).__name__, 
            'CoTResponseParser'
        )

    def test_reasoning_extraction_in_answer(self):
        """Test that reasoning is properly extracted in answer_question."""
        # Mock reasoning response with template text
        template_reasoning = "LM that thinks step by step before answering\n\nThis is a physics problem about force and acceleration. According to Newton's second law, F = ma."
        
        # Mock the token generator methods
        self.question_answerer.token_generator.generate_batch_with_kv_cache = MagicMock(
            return_value=([template_reasoning], [None])
        )
        self.question_answerer.token_generator.get_logits_for_prompt = MagicMock(
            return_value=MagicMock()
        )
        
        # Mock the message generator
        self.question_answerer.message_generator.create_reasoning_messages = MagicMock(
            return_value=[{"role": "user", "content": "test"}]
        )
        self.question_answerer.message_generator.create_final_answer_messages = MagicMock(
            return_value=[{"role": "user", "content": "test"}]
        )
        
        # Mock the token resolver
        self.question_answerer.token_resolver.get_answer_token_ids = MagicMock(
            return_value={"A": 1, "B": 2}
        )
        
        # Mock the _calculate_answer_probabilities method
        self.question_answerer._calculate_answer_probabilities = MagicMock(
            return_value={"A": 0.7, "B": 0.3}
        )
        
        # Mock the _format_prompt method
        self.question_answerer._format_prompt = MagicMock(
            return_value="test prompt"
        )
        
        # Mock sampler
        mock_sampler = MagicMock()
        mock_sampler.sample.return_value = 1
        
        # Mock tokenizer decode
        self.mock_tokenizer.decode.return_value = "A"
        
        # Test question answering
        result = self.question_answerer.answer_question(
            "What is Newton's second law?",
            ["F = ma", "E = mc²"],
            mock_sampler,
            include_cot=True
        )
        
        # Verify that reasoning was cleaned
        self.assertIn('reasoning', result)
        self.assertIn('raw_reasoning', result)
        self.assertIn('reasoning_quality', result)
        
        # Template text should be removed from reasoning
        self.assertNotIn("LM that thinks step by step", result['reasoning'])
        
        # Good content should be preserved
        self.assertIn("Newton's second law", result['reasoning'])
        self.assertIn("F = ma", result['reasoning'])
        
        # Raw reasoning should contain original text
        self.assertIn("LM that thinks step by step", result['raw_reasoning'])

    def test_cot_parser_template_detection(self):
        """Test that CoT parser correctly detects template responses."""
        parser = self.question_answerer.cot_parser
        
        # Test template detection
        template_response = "LM that thinks step by step before answering"
        self.assertTrue(parser.is_template_response(template_response))
        
        # Test good response detection
        good_response = "This is a chemistry problem. The reaction between sodium and water produces sodium hydroxide and hydrogen gas."
        self.assertFalse(parser.is_template_response(good_response))

    def test_cot_parser_reasoning_extraction(self):
        """Test that CoT parser correctly extracts reasoning."""
        parser = self.question_answerer.cot_parser
        
        # Test extraction with template text
        mixed_response = """I need to think through this step by step.
        
        This is a biology question about photosynthesis. Plants use chlorophyll to convert sunlight, carbon dioxide, and water into glucose and oxygen.
        
        The process occurs in the chloroplasts of plant cells and is essential for life on Earth."""
        
        extracted = parser.extract_reasoning(mixed_response)
        
        # Template should be removed
        self.assertNotIn("I need to think through this step by step", extracted)
        
        # Good content should be preserved
        self.assertIn("photosynthesis", extracted)
        self.assertIn("chlorophyll", extracted)
        self.assertIn("chloroplasts", extracted)

    def test_cot_parser_quality_validation(self):
        """Test that CoT parser correctly validates reasoning quality."""
        parser = self.question_answerer.cot_parser
        
        # High quality reasoning
        high_quality = """This problem involves calculating the area of a circle. The formula is A = πr².
        Given that the radius is 5 meters, we can substitute: A = π × 25 = 78.54 square meters."""
        
        self.assertTrue(parser.validate_reasoning_quality(high_quality))
        
        # Low quality reasoning
        low_quality = "The answer is A."
        self.assertFalse(parser.validate_reasoning_quality(low_quality))


if __name__ == '__main__':
    unittest.main()