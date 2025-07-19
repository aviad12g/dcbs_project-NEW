"""
Chain-of-Thought response parsing utilities.

This module provides utilities for extracting meaningful reasoning content
from LLM responses, filtering out template text and boilerplate phrases.
"""

import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class CoTResponseParser:
    """Parser for extracting meaningful reasoning from Chain-of-Thought responses."""
    
    def __init__(self):
        """Initialize the parser with common template patterns."""
        # Common template phrases that should be filtered out
        self.template_patterns = [
            # Generic thinking phrases
            r"LM that thinks step by step before answering",
            r"I need to think through this step by step",
            r"Let me analyze this question carefully",
            r"Let me think about this step by step",
            r"I'll work through this systematically",
            r"Let me break this down step by step",
            
            # Instruction-following phrases
            r"I should analyze each option",
            r"I need to analyze each option carefully",
            r"I need to consider each choice",
            r"Let me examine each possibility",
            
            # Generic reasoning starters (when they appear alone)
            r"^To solve this problem[,:]?\s*$",
            r"^To answer this question[,:]?\s*$",
            r"^Looking at this question[,:]?\s*$",
            
            # Empty or minimal responses
            r"^\s*$",  # Empty lines
            r"^[.]{3,}$",  # Just dots
            r"^[-]{3,}$",  # Just dashes
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) 
                                 for pattern in self.template_patterns]
        
        # Patterns that indicate substantive reasoning
        self.reasoning_indicators = [
            r"because",
            r"therefore",
            r"however",
            r"since",
            r"given that",
            r"considering",
            r"based on",
            r"according to",
            r"this means",
            r"as a result",
            r"consequently",
            r"furthermore",
            r"moreover",
            r"in contrast",
            r"on the other hand",
            r"for example",
            r"specifically",
            r"in particular",
        ]
        
        self.reasoning_indicators_compiled = [re.compile(pattern, re.IGNORECASE) 
                                            for pattern in self.reasoning_indicators]
    
    def extract_reasoning(self, response: str) -> str:
        """
        Extract actual reasoning content from response.
        
        Args:
            response: Raw response from the model
            
        Returns:
            Cleaned reasoning content, or empty string if no meaningful content found
        """
        if not response or not isinstance(response, str):
            return ""
        
        # Split into sentences/lines for processing
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check if line matches any template pattern
            is_template = False
            for pattern in self.compiled_patterns:
                if pattern.search(line):
                    is_template = True
                    break
            
            if not is_template:
                cleaned_lines.append(line)
        
        # Join cleaned lines
        cleaned_response = '\n'.join(cleaned_lines).strip()
        
        # If response is too short or doesn't contain reasoning indicators, 
        # it might be just template text
        if len(cleaned_response) < 20:  # Very short responses are likely templates
            return ""
        
        return cleaned_response
    
    def is_template_response(self, response: str) -> bool:
        """
        Check if response is primarily template text.
        
        Args:
            response: Response to check
            
        Returns:
            True if response appears to be mostly template text
        """
        if not response or not isinstance(response, str):
            return True
        
        response = response.strip()
        
        # Check if entire response matches a template pattern
        for pattern in self.compiled_patterns:
            if pattern.search(response):
                # If the match covers most of the response, it's likely a template
                match = pattern.search(response)
                if match and len(match.group(0)) > len(response) * 0.7:
                    return True
        
        # Check for very short responses (likely templates)
        if len(response) < 15:
            return True
        
        # Check if response contains any reasoning indicators
        has_reasoning_indicators = any(
            pattern.search(response) for pattern in self.reasoning_indicators_compiled
        )
        
        # If no reasoning indicators and short, likely template
        if not has_reasoning_indicators and len(response) < 50:
            return True
        
        return False
    
    def validate_reasoning_quality(self, reasoning: str) -> bool:
        """
        Validate that reasoning contains substantive content.
        
        Args:
            reasoning: Reasoning text to validate
            
        Returns:
            True if reasoning appears to be of good quality
        """
        if not reasoning or not isinstance(reasoning, str):
            return False
        
        reasoning = reasoning.strip()
        
        # Minimum length check
        if len(reasoning) < 20:
            return False
        
        # Check for reasoning indicators
        has_indicators = any(
            pattern.search(reasoning) for pattern in self.reasoning_indicators_compiled
        )
        
        # Check for question words (what, why, how, etc.)
        question_words = ['what', 'why', 'how', 'when', 'where', 'which']
        has_question_words = any(word in reasoning.lower() for word in question_words)
        
        # Check for logical structure (sentences with proper punctuation)
        sentences = re.split(r'[.!?]+', reasoning)
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # Quality criteria
        quality_score = 0
        
        if has_indicators:
            quality_score += 3  # Reasoning indicators are very important
        if has_question_words:
            quality_score += 1
        if len(meaningful_sentences) >= 2:
            quality_score += 2
        if len(reasoning) > 100:
            quality_score += 1
        if len(reasoning) > 50:  # Medium length bonus
            quality_score += 1
        
        # Require minimum quality score (lowered to be less strict)
        return quality_score >= 3
    
    def parse_cot_response(self, response: str) -> dict:
        """
        Parse a Chain-of-Thought response and return analysis.
        
        Args:
            response: Raw CoT response from model
            
        Returns:
            Dictionary with parsing results:
            - reasoning: Extracted reasoning content
            - is_template: Whether response appears to be template
            - is_valid: Whether reasoning is of good quality
            - original_length: Length of original response
            - cleaned_length: Length of cleaned reasoning
        """
        original_length = len(response) if response else 0
        
        # Extract reasoning
        reasoning = self.extract_reasoning(response)
        cleaned_length = len(reasoning)
        
        # Check if template
        is_template = self.is_template_response(response)
        
        # Validate quality
        is_valid = self.validate_reasoning_quality(reasoning)
        
        return {
            'reasoning': reasoning,
            'is_template': is_template,
            'is_valid': is_valid,
            'original_length': original_length,
            'cleaned_length': cleaned_length,
            'extraction_ratio': cleaned_length / max(original_length, 1)
        }


# Global parser instance for convenience
default_parser = CoTResponseParser()


def extract_reasoning(response: str) -> str:
    """
    Convenience function to extract reasoning using default parser.
    
    Args:
        response: Raw CoT response
        
    Returns:
        Cleaned reasoning content
    """
    return default_parser.extract_reasoning(response)


def is_template_response(response: str) -> bool:
    """
    Convenience function to check if response is template using default parser.
    
    Args:
        response: Response to check
        
    Returns:
        True if response appears to be template text
    """
    return default_parser.is_template_response(response)


def validate_reasoning_quality(reasoning: str) -> bool:
    """
    Convenience function to validate reasoning quality using default parser.
    
    Args:
        reasoning: Reasoning to validate
        
    Returns:
        True if reasoning is of good quality
    """
    return default_parser.validate_reasoning_quality(reasoning)


def parse_cot_response(response: str) -> dict:
    """
    Convenience function to parse CoT response using default parser.
    
    Args:
        response: Raw CoT response
        
    Returns:
        Dictionary with parsing results
    """
    return default_parser.parse_cot_response(response)