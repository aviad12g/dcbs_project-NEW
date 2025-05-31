"""
Question answering orchestration.

This module combines message templates and token generation
to answer multiple choice questions with or without reasoning.
"""

import time
from typing import Dict, List, Optional, Tuple

import torch

from src.dcbs import SamplingContext
from src.errors import eval_logger as logger
from src.token_utils import AnswerTokenResolver

from .message_templates import MessageTemplateGenerator
from .token_generator import TokenGenerator


class QuestionAnswerer:
    """Orchestrates question answering with various samplers."""
    
    def __init__(self, model, tokenizer, context: SamplingContext):
        self.model = model
        self.tokenizer = tokenizer
        self.context = context
        self.device = context.device
        
        # Initialize components
        self.message_generator = MessageTemplateGenerator()
        self.token_generator = TokenGenerator(model, tokenizer, self.device)
        self.token_resolver = AnswerTokenResolver(tokenizer)
    
    def answer_question(
        self,
        question: str,
        options: List[str],
        sampler,
        include_cot: bool = True
    ) -> Dict:
        """
        Answer a multiple choice question.
        
        Args:
            question: The question text
            options: List of answer options
            sampler: Sampler to use for generation
            include_cot: Whether to include chain-of-thought reasoning
            
        Returns:
            Dictionary with answer details including:
            - selected_answer: The chosen answer letter
            - reasoning: The reasoning text (if include_cot is True)
            - answer_probs: Probabilities for each answer option
            - pred_token_id: The predicted token ID
        """
        start_time = time.time()
        
        if include_cot:
            result = self._answer_with_reasoning(question, options, sampler)
        else:
            result = self._answer_directly(question, options, sampler)
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def _answer_with_reasoning(
        self,
        question: str,
        options: List[str],
        sampler
    ) -> Dict:
        """Answer with chain-of-thought reasoning."""
        # Step 1: Generate reasoning
        reasoning_messages = self.message_generator.create_reasoning_messages(question, options)
        reasoning_response, reasoning_cache = self.token_generator.generate_with_kv_cache(
            reasoning_messages, sampler, max_new_tokens=500
        )
        
        # Step 2: Generate final answer
        final_messages = self.message_generator.create_final_answer_messages(
            reasoning_messages, reasoning_response
        )
        
        # Log final chat for debugging
        logger.debug("Final conversation flow:")
        for i, msg in enumerate(final_messages):
            role = msg['role']
            content = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
            logger.debug(f"  {i+1}. {role}: {content}")
        
        # Create the final prompt
        final_prompt = self.tokenizer.apply_chat_template(
            final_messages, tokenize=False, add_generation_prompt=True
        )
        
        # Add "The final answer is option" to the prompt
        final_prompt += "The final answer is option"
        
        logger.debug(f"Final answer prompt ends with: ...{final_prompt[-50:]}")
        
        # Get logits for final answer
        logits = self.token_generator.get_logits_for_prompt(final_prompt)
        
        # Get answer token mappings and probabilities
        answer_ids = self.token_resolver.get_answer_token_ids(options)
        answer_probs = self._calculate_answer_probabilities(logits, answer_ids)
        
        # Sample the answer
        filter_tokens = set(answer_ids.values())
        pred_token_id = sampler.sample(logits, filter_tokens=filter_tokens)
        
        # Find which answer was selected
        selected_answer = None
        for answer, token_id in answer_ids.items():
            if token_id == pred_token_id:
                selected_answer = answer
                break
        
        return {
            'selected_answer': selected_answer,
            'reasoning': reasoning_response,
            'answer_probs': answer_probs,
            'pred_token_id': pred_token_id,
            'answer_ids': answer_ids,
            'filter_tokens': filter_tokens,
            'logits': logits
        }
    
    def _answer_directly(
        self,
        question: str,
        options: List[str],
        sampler
    ) -> Dict:
        """Answer directly without reasoning."""
        # Create messages for direct answer
        messages = self.message_generator.create_direct_answer_messages(question, options)
        
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Add the answer prompt to the assistant's message
        prompt += "The correct answer is option"
        
        logger.debug(f"Direct answer prompt: {prompt[-100:]}")
        
        # Get logits
        logits = self.token_generator.get_logits_for_prompt(prompt)
        
        # Get answer token mappings and probabilities
        answer_ids = self.token_resolver.get_answer_token_ids(options)
        answer_probs = self._calculate_answer_probabilities(logits, answer_ids)
        
        # Sample the answer
        filter_tokens = set(answer_ids.values())
        pred_token_id = sampler.sample(logits, filter_tokens=filter_tokens)
        
        # Find which answer was selected
        selected_answer = None
        for answer, token_id in answer_ids.items():
            if token_id == pred_token_id:
                selected_answer = answer
                break
        
        return {
            'selected_answer': selected_answer,
            'reasoning': None,
            'answer_probs': answer_probs,
            'pred_token_id': pred_token_id,
            'answer_ids': answer_ids,
            'filter_tokens': filter_tokens,
            'logits': logits
        }
    
    def _calculate_answer_probabilities(
        self,
        logits: torch.Tensor,
        answer_ids: Dict[str, int]
    ) -> Dict[str, float]:
        """Calculate probabilities for each answer option."""
        all_probs = torch.softmax(logits, dim=0)
        answer_probs = {
            option: all_probs[token_id].item()
            for option, token_id in answer_ids.items()
        }
        return answer_probs 