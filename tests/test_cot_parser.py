"""
Tests for Chain-of-Thought response parsing.

This module tests the CoTResponseParser class and related functions
to ensure proper extraction of reasoning content from model responses.
"""

import unittest

from src.utils.cot_parser import (
    CoTResponseParser,
    extract_reasoning,
    is_template_response,
    validate_reasoning_quality,
    parse_cot_response,
)


class TestCoTResponseParser(unittest.TestCase):
    """Test cases for CoTResponseParser class."""

    def setUp(self):
        """Set up test environment."""
        self.parser = CoTResponseParser()

    def test_extract_reasoning_with_template_text(self):
        """Test extraction of reasoning while filtering out template text."""
        # Response with template text at the beginning
        response = """LM that thinks step by step before answering
        
        To solve this problem, I need to analyze each option carefully.
        
        Option A states that water boils at 100°C at sea level. This is correct because at standard atmospheric pressure (1 atm), water's boiling point is indeed 100°C.
        
        Option B claims water boils at 90°C. This would only be true at reduced atmospheric pressure, such as at high altitudes.
        
        Therefore, Option A is the correct answer."""
        
        reasoning = self.parser.extract_reasoning(response)
        
        # Should not contain template text
        self.assertNotIn("LM that thinks step by step", reasoning)
        
        # Should contain actual reasoning
        self.assertIn("Option A states that water boils", reasoning)
        self.assertIn("This is correct because", reasoning)
        self.assertIn("Therefore, Option A", reasoning)
        
        # Should be substantial content
        self.assertGreater(len(reasoning), 100)

    def test_extract_reasoning_pure_template(self):
        """Test extraction from pure template responses."""
        template_responses = [
            "LM that thinks step by step before answering",
            "I need to think through this step by step",
            "Let me analyze this question carefully",
            "",
            "...",
            "---",
        ]
        
        for response in template_responses:
            reasoning = self.parser.extract_reasoning(response)
            self.assertEqual(reasoning, "", f"Template response should return empty string: {response}")

    def test_extract_reasoning_good_content(self):
        """Test extraction of good reasoning content."""
        good_response = """Looking at this physics problem, I need to consider the relationship between force, mass, and acceleration.
        
        According to Newton's second law, F = ma, where F is force, m is mass, and a is acceleration.
        
        Given that the mass is 10 kg and the acceleration is 5 m/s², I can calculate:
        F = 10 kg × 5 m/s² = 50 N
        
        Therefore, the force required is 50 Newtons."""
        
        reasoning = self.parser.extract_reasoning(good_response)
        
        # Should preserve the good content
        self.assertIn("Newton's second law", reasoning)
        self.assertIn("F = ma", reasoning)
        self.assertIn("10 kg × 5 m/s²", reasoning)
        self.assertIn("50 Newtons", reasoning)
        
        # Should be substantial
        self.assertGreater(len(reasoning), 200)

    def test_is_template_response_detection(self):
        """Test detection of template responses."""
        # Template responses
        template_responses = [
            "LM that thinks step by step before answering",
            "I need to think through this step by step",
            "Let me analyze this question carefully",
            "",
            "...",
            "To solve this problem:",  # Short generic starter
        ]
        
        for response in template_responses:
            self.assertTrue(
                self.parser.is_template_response(response),
                f"Should detect as template: {response}"
            )
        
        # Non-template responses
        good_responses = [
            "The answer is B because photosynthesis requires chlorophyll to convert sunlight into chemical energy.",
            "Looking at the data, we can see that the temperature increased by 15°C over the 10-minute period, indicating a rapid heating process.",
            "This is incorrect because the law of conservation of energy states that energy cannot be created or destroyed, only transformed.",
        ]
        
        for response in good_responses:
            self.assertFalse(
                self.parser.is_template_response(response),
                f"Should not detect as template: {response}"
            )

    def test_validate_reasoning_quality(self):
        """Test validation of reasoning quality."""
        # High quality reasoning
        high_quality = """This problem involves calculating the area of a circle. The formula for the area of a circle is A = πr², where r is the radius.
        
        Given that the radius is 5 meters, I can substitute this value:
        A = π × (5)² = π × 25 = 25π ≈ 78.54 square meters
        
        Therefore, the area is approximately 78.54 square meters."""
        
        self.assertTrue(self.parser.validate_reasoning_quality(high_quality))
        
        # Low quality reasoning
        low_quality_examples = [
            "",  # Empty
            "Yes.",  # Too short
            "I think the answer is A.",  # No reasoning
            "This is correct.",  # No explanation
            "LM that thinks step by step before answering",  # Template
        ]
        
        for reasoning in low_quality_examples:
            self.assertFalse(
                self.parser.validate_reasoning_quality(reasoning),
                f"Should reject low quality reasoning: {reasoning}"
            )

    def test_parse_cot_response_comprehensive(self):
        """Test comprehensive parsing of CoT responses."""
        # Good response with some template text
        response = """I need to think through this step by step.
        
        This is a question about chemical reactions. When sodium (Na) reacts with water (H₂O), it produces sodium hydroxide (NaOH) and hydrogen gas (H₂).
        
        The balanced chemical equation is: 2Na + 2H₂O → 2NaOH + H₂
        
        This reaction is highly exothermic, meaning it releases a lot of heat. The hydrogen gas produced can even ignite due to the heat generated.
        
        Therefore, the correct answer is that sodium reacts vigorously with water to produce sodium hydroxide and hydrogen gas."""
        
        result = self.parser.parse_cot_response(response)
        
        # Check structure
        self.assertIn('reasoning', result)
        self.assertIn('is_template', result)
        self.assertIn('is_valid', result)
        self.assertIn('original_length', result)
        self.assertIn('cleaned_length', result)
        self.assertIn('extraction_ratio', result)
        
        # Check values
        self.assertFalse(result['is_template'])  # Not primarily template
        self.assertTrue(result['is_valid'])  # Good quality reasoning
        self.assertGreater(result['cleaned_length'], 100)  # Substantial content
        self.assertGreater(result['extraction_ratio'], 0.5)  # Good extraction ratio
        
        # Check reasoning content
        reasoning = result['reasoning']
        self.assertNotIn("I need to think through this step by step", reasoning)  # Template removed
        self.assertIn("chemical reactions", reasoning)  # Content preserved
        self.assertIn("2Na + 2H₂O → 2NaOH + H₂", reasoning)  # Equation preserved

    def test_parse_cot_response_template_only(self):
        """Test parsing of template-only responses."""
        template_response = "LM that thinks step by step before answering"
        
        result = self.parser.parse_cot_response(template_response)
        
        self.assertTrue(result['is_template'])
        self.assertFalse(result['is_valid'])
        self.assertEqual(result['reasoning'], "")
        self.assertEqual(result['cleaned_length'], 0)
        self.assertEqual(result['extraction_ratio'], 0.0)

    def test_reasoning_indicators_detection(self):
        """Test detection of reasoning indicators."""
        responses_with_indicators = [
            "This is correct because the evidence supports it.",
            "However, we must consider the alternative explanation.",
            "Therefore, the conclusion follows logically.",
            "Since the temperature is rising, we can expect expansion.",
            "Given that the pressure is constant, volume will increase.",
            "Based on the data, we can conclude that the hypothesis is supported.",
        ]
        
        for response in responses_with_indicators:
            reasoning = self.parser.extract_reasoning(response)
            is_valid = self.parser.validate_reasoning_quality(reasoning)
            # These should generally be considered valid (though length matters too)
            if len(reasoning) > 20:  # Only if they meet minimum length
                self.assertTrue(is_valid, f"Should validate reasoning with indicators: {response}")

    def test_multiline_template_filtering(self):
        """Test filtering of multi-line template content."""
        response = """Let me think about this step by step.
        I need to analyze each option carefully.
        
        Looking at option A: This describes photosynthesis, which is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen.
        
        Looking at option B: This describes cellular respiration, which is the opposite process where glucose is broken down to release energy.
        
        The question asks about the process that produces oxygen as a byproduct. This is clearly photosynthesis.
        
        Therefore, option A is correct."""
        
        reasoning = self.parser.extract_reasoning(response)
        
        # Template lines should be removed
        self.assertNotIn("Let me think about this step by step", reasoning)
        self.assertNotIn("I need to analyze each option carefully", reasoning)
        
        # Good content should be preserved
        self.assertIn("photosynthesis", reasoning)
        self.assertIn("cellular respiration", reasoning)
        self.assertIn("Therefore, option A is correct", reasoning)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def test_extract_reasoning_function(self):
        """Test the convenience extract_reasoning function."""
        response = "LM that thinks step by step before answering\n\nThis is good reasoning because it explains the concept clearly."
        
        reasoning = extract_reasoning(response)
        
        self.assertNotIn("LM that thinks step by step", reasoning)
        self.assertIn("good reasoning because", reasoning)

    def test_is_template_response_function(self):
        """Test the convenience is_template_response function."""
        self.assertTrue(is_template_response("LM that thinks step by step before answering"))
        self.assertFalse(is_template_response("This is a detailed explanation of the scientific process."))

    def test_validate_reasoning_quality_function(self):
        """Test the convenience validate_reasoning_quality function."""
        good_reasoning = "This is correct because the scientific evidence clearly demonstrates that photosynthesis requires chlorophyll to convert light energy into chemical energy."
        bad_reasoning = "Yes."
        
        self.assertTrue(validate_reasoning_quality(good_reasoning))
        self.assertFalse(validate_reasoning_quality(bad_reasoning))

    def test_parse_cot_response_function(self):
        """Test the convenience parse_cot_response function."""
        response = "This is a good explanation because it provides clear reasoning and evidence."
        
        result = parse_cot_response(response)
        
        self.assertIsInstance(result, dict)
        self.assertIn('reasoning', result)
        self.assertIn('is_valid', result)


if __name__ == '__main__':
    unittest.main()