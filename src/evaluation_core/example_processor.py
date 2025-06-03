"""
Example processing with proper conversation flow.

This module implements the correct two-step conversation flow:
1. LLM completes assistant reasoning response
2. LLM completes assistant final answer response

Key features:
- Never let LLM complete 'user' messages
- Use proper add_generation_prompt=True for both steps
- Implement KV caching for efficiency
- Increase token limits to avoid truncation
- Log final chat messages for debugging
"""

import time
from typing import Dict, List, Optional, Tuple

import torch
from transformers import Cache

from src.dcbs import SamplingContext
from src.errors import eval_logger as logger
from src.token_utils import AnswerTokenResolver
from .question_answerer import QuestionAnswerer


class ExampleProcessor:
    """Example processor with correct conversation flow and KV caching."""

    def __init__(self, model, tokenizer, context: SamplingContext):
        self.model = model
        self.tokenizer = tokenizer
        self.context = context
        self.device = context.device
        self.question_answerer = QuestionAnswerer(model, tokenizer, context)

    def process_example(self, example: Dict, sampler, include_cot: bool = True) -> Dict:
        """
        Process a single example using the improved two-step conversation flow with optimized KV caching.
        
        Args:
            example: Example data
            sampler: Sampler to use for generation
            include_cot: Whether to include chain-of-thought reasoning
            
        Returns:
            Processed example with results
        """
        start_time = time.time()
        
        # Extract example data
        if "question" in example:
            sentence = example["question"]
            options = example["options"]
            correct_option = example.get("correct_option", "1")
            correct_idx = int(correct_option) - 1
            correct_answer = options[correct_idx]
        else:
            raise ValueError("Example must have 'question' field")

        result = {
            "id": example.get("id", "unknown"),
            "sentence": sentence,
            "options": options,
            "correct_answer": correct_answer,
            "correct_option": correct_option,
        }

        # Use QuestionAnswerer to get the answer
        answer_result = self.question_answerer.answer_question(
            sentence, options, sampler, include_cot
        )
        
        # Extract results
        result["cot_reasoning"] = answer_result.get("reasoning")
        result["answer_ids"] = answer_result["answer_ids"]
        result["filter_tokens"] = answer_result["filter_tokens"]
        result["correct_id"] = answer_result["answer_ids"][correct_answer]
        result["logits"] = answer_result["logits"]
        result["answer_probs"] = answer_result["answer_probs"]
        result["processing_time"] = time.time() - start_time

        return result

    def evaluate_with_sampler(
        self, processed_result: Dict, sampler, sampler_name: str
    ) -> Dict:
        """
        Evaluate a processed example with a specific sampler.
        
        REVERTED: Each sampler now generates fresh logits independently
        to ensure DCBS clustering works correctly.
        
        Args:
            processed_result: Result from process_example
            sampler: Sampler to use
            sampler_name: Name for logging
            
        Returns:
            Evaluation result
        """
        start_time = time.time()
        
        # REVERTED: Generate fresh logits for each sampler independently
        sentence = processed_result["sentence"]
        options = processed_result["options"]
        include_cot = processed_result.get("cot_reasoning") is not None
        
        # Get fresh answer result for this sampler
        answer_result = self.question_answerer.answer_question(
            sentence, options, sampler, include_cot
        )
        
        logits = answer_result["logits"]
        filter_tokens = answer_result["filter_tokens"]
        correct_id = processed_result["correct_id"]
        
        # Sample using the specified sampler with fresh logits
        pred_id = sampler.sample(logits, filter_tokens=filter_tokens)
        
        # Check correctness
        correct = (pred_id == correct_id)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            "sampler": sampler_name,
            "pred_id": pred_id,
            "correct": correct,
            "elapsed_ms": elapsed_ms
        } 