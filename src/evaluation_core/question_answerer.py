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
from src.utils.cot_parser import CoTResponseParser

from .message_templates import MessageTemplateGenerator
from .token_generator import TokenGenerator


import re

class QuestionAnswerer:
    """Orchestrates question answering with various samplers."""
    
    def __init__(self, model, tokenizer, context: SamplingContext):
        self.model = model
        self.tokenizer = tokenizer
        self.context = context
        self.device = context.device
        
        # Check if chat template is available
        self.has_chat_template = (
            hasattr(tokenizer, 'chat_template') and 
            tokenizer.chat_template is not None
        )
        
        # Initialize components
        self.message_generator = MessageTemplateGenerator()
        self.token_generator = TokenGenerator(model, tokenizer, self.device)
        self.token_resolver = AnswerTokenResolver(tokenizer)
        self.cot_parser = CoTResponseParser()
    
    def _format_prompt(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """
        Format messages into a prompt, using chat template if available or fallback formatting.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            add_generation_prompt: Whether to add generation prompt
            
        Returns:
            Formatted prompt string
        """
        if self.has_chat_template:
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=add_generation_prompt
                )
            except Exception as e:
                logger.warning(f"Chat template failed, using fallback: {e}")
                # Fall through to simple formatting
        
        # Simple fallback formatting
        prompt = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'user':
                prompt += f"User: {content}\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n"
            elif role == 'system':
                prompt += f"System: {content}\n"
        
        if add_generation_prompt and not prompt.endswith("Assistant: "):
            prompt += "Assistant: "
            
        return prompt
    


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
    
    def answer_questions_batch(
        self,
        questions: List[str],
        options_list: List[List[str]],
        sampler,
        include_cot: bool = True
    ) -> List[Dict]:
        """
        Answer multiple questions in parallel with batch processing.
        
        Args:
            questions: List of question texts
            options_list: List of answer options for each question
            sampler: Sampler to use for generation
            include_cot: Whether to include chain-of-thought reasoning
            
        Returns:
            List of answer dictionaries, one per question
        """
        start_time = time.time()
        logger.debug(f"Starting batch question answering for {len(questions)} questions")
        
        if not questions:
            return []
        
        if include_cot:
            results = self._answer_batch_with_reasoning(questions, options_list, sampler)
        else:
            results = self._answer_batch_directly(questions, options_list, sampler)
        
        # Add timing information
        total_time = time.time() - start_time
        for result in results:
            result['processing_time'] = total_time / len(results)  # Average per question
        
        logger.debug(f"Batch question answering completed in {total_time:.2f}s")
        return results
    
    def _answer_with_reasoning(
        self,
        question: str,
        options: List[str],
        sampler
    ) -> Dict:
        """Answer with chain-of-thought reasoning."""
        # Step 1: Generate reasoning
        reasoning_messages = self.message_generator.create_reasoning_messages(question, options)
        # FIXED: Increased max_new_tokens and better error handling
        reasoning_responses, reasoning_caches = self.token_generator.generate_batch_with_kv_cache(
            [reasoning_messages], sampler, max_new_tokens=800
        )
        reasoning_response = reasoning_responses[0]
        
        # Clean the reasoning response using the CoT parser
        cleaned_reasoning = self.cot_parser.extract_reasoning(reasoning_response)

        # Step 2: Generate final answer
        final_messages = self.message_generator.create_final_answer_messages(
            reasoning_messages, cleaned_reasoning
        )
        
        # Log final chat for debugging
        logger.debug("Final conversation flow:")
        for i, msg in enumerate(final_messages):
            role = msg['role']
            content = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
            logger.debug(f"  {i+1}. {role}: {content}")
        
        # Create the final prompt using our flexible formatter
        final_prompt = self._format_prompt(final_messages, add_generation_prompt=True)
        
        # Add "The final answer is option" to the prompt
        final_prompt += "The final answer is option"
        
        logger.debug(f"Final answer prompt ends with: ...{final_prompt[-50:]}")
        
        # Get logits for final answer
        logits = self.token_generator.get_logits_for_prompt(final_prompt)
        
        # Get answer token IDs using the original label-based approach
        answer_ids = self.token_resolver.get_answer_token_ids(options)

        # Calculate probabilities
        answer_probs = self._calculate_answer_probabilities(logits, answer_ids)

        # Sample using the label tokens (A/B/C/D)
        filter_tokens = set(answer_ids.values())
        pred_token_id = sampler.sample(logits, filter_tokens=filter_tokens)

        # Find which answer was selected
        selected_answer = None
        for answer_text, token_id in answer_ids.items():
            if token_id == pred_token_id:
                selected_answer = answer_text
                break

        # Extract final answer from generated text
        generated_text = self.tokenizer.decode(pred_token_id)
        final_answer = self._extract_final_answer(generated_text)

        # Parse and clean the reasoning response
        parsed_reasoning = self.cot_parser.parse_cot_response(reasoning_response)
        cleaned_reasoning = parsed_reasoning['reasoning']
        
        # Log if reasoning appears to be template text (debug level to reduce noise)
        if parsed_reasoning['is_template']:
            logger.debug(f"CoT response appears to be template text: {reasoning_response[:100]}...")
        elif not parsed_reasoning['is_valid']:
            logger.debug(f"CoT response quality is low: {reasoning_response[:100]}...")
        
        return {
            'selected_answer': final_answer or selected_answer,
            'reasoning': cleaned_reasoning,  # Use cleaned reasoning
            'raw_reasoning': reasoning_response,  # Keep original for debugging
            'reasoning_quality': parsed_reasoning,  # Include quality metrics
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
        
        # Use our flexible prompt formatter
        prompt = self._format_prompt(messages, add_generation_prompt=True)
        
        # Add the answer prompt to the assistant's message
        prompt += "The correct answer is option"
        
        logger.debug(f"Direct answer prompt: {prompt[-100:]}")
        
        # Get logits
        logits = self.token_generator.get_logits_for_prompt(prompt)
        
        # Get answer token IDs using the original label-based approach
        answer_ids = self.token_resolver.get_answer_token_ids(options)

        # Calculate probabilities
        answer_probs = self._calculate_answer_probabilities(logits, answer_ids)

        # Sample using the label tokens (A/B/C/D)
        filter_tokens = set(answer_ids.values())
        pred_token_id = sampler.sample(logits, filter_tokens=filter_tokens)

        # Find which answer was selected
        selected_answer = None
        for answer_text, token_id in answer_ids.items():
            if token_id == pred_token_id:
                selected_answer = answer_text
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
        import torch
        
        # Extract logits for only the answer tokens
        answer_token_ids = list(answer_ids.values())
        answer_logits = logits[answer_token_ids]
        
        # Apply softmax to only the answer tokens (so they sum to 1.0)
        answer_probs_tensor = torch.softmax(answer_logits, dim=-1)
        
        # Map back to option text
        answer_probs = {}
        for i, (option, token_id) in enumerate(answer_ids.items()):
            answer_probs[option] = answer_probs_tensor[i].item()
        
        return answer_probs
    
    def _answer_batch_with_reasoning(
        self,
        questions: List[str],
        options_list: List[List[str]],
        sampler
    ) -> List[Dict]:
        """Answer multiple questions with batch chain-of-thought reasoning."""
        logger.debug("Starting batch CoT reasoning")
        
        # STEP 1: Batch reasoning generation
        reasoning_messages_batch = []
        for question, options in zip(questions, options_list):
            reasoning_messages = self.message_generator.create_reasoning_messages(question, options)
            reasoning_messages_batch.append(reasoning_messages)
        
        # Generate reasoning responses in parallel
        # FIXED: Increased max_new_tokens for batch processing
        reasoning_responses, reasoning_caches = self.token_generator.generate_batch_with_kv_cache(
            reasoning_messages_batch, sampler, max_new_tokens=800
        )
        
        # STEP 2: Batch final answer generation
        final_messages_batch = []
        for reasoning_msgs, reasoning_resp in zip(reasoning_messages_batch, reasoning_responses):
            cleaned_reasoning = self.cot_parser.extract_reasoning(reasoning_resp)
            final_messages = self.message_generator.create_final_answer_messages(
                reasoning_msgs, cleaned_reasoning
            )
            final_messages_batch.append(final_messages)
        
        # Generate final answers using fresh batching (since we removed cached version)
        final_responses, _ = self.token_generator.generate_batch_with_kv_cache(
            final_messages_batch, sampler, max_new_tokens=50
        )
        
        # STEP 3: Process final answers and get probabilities
        results = []
        for i, (reasoning_resp, final_resp, options) in enumerate(
            zip(reasoning_responses, final_responses, options_list)
        ):
            # Create final prompt for logits calculation
            final_messages = final_messages_batch[i]
            final_prompt = self._format_prompt(final_messages, add_generation_prompt=True)
            final_prompt += "The final answer is option"
            
            # Get logits and probabilities
            logits = self.token_generator.get_logits_for_prompt(final_prompt)
            
            # Get answer token IDs using the original label-based approach
            answer_ids = self.token_resolver.get_answer_token_ids(options)

            # Calculate probabilities
            answer_probs = self._calculate_answer_probabilities(logits, answer_ids)

            # Sample using the label tokens (A/B/C/D)
            filter_tokens = set(answer_ids.values())
            pred_token_id = sampler.sample(logits, filter_tokens=filter_tokens)

            # Find which answer was selected
            selected_answer = None
            for answer_text, token_id in answer_ids.items():
                if token_id == pred_token_id:
                    selected_answer = answer_text
                    break

            # Extract final answer from generated text
            generated_text = self.tokenizer.decode(pred_token_id)
            final_answer = self._extract_final_answer(generated_text)

            # Parse and clean the reasoning response
            parsed_reasoning = self.cot_parser.parse_cot_response(reasoning_resp)
            cleaned_reasoning = parsed_reasoning['reasoning']
            
            # Log if reasoning appears to be template text
            if parsed_reasoning['is_template']:
                logger.warning(f"Batch CoT response appears to be template text: {reasoning_resp[:100]}...")
            elif not parsed_reasoning['is_valid']:
                logger.warning(f"Batch CoT response quality is low: {reasoning_resp[:100]}...")
            
            result = {
                'selected_answer': final_answer or selected_answer,
                'reasoning': cleaned_reasoning,  # Use cleaned reasoning
                'raw_reasoning': reasoning_resp,  # Keep original for debugging
                'reasoning_quality': parsed_reasoning,  # Include quality metrics
                'answer_probs': answer_probs,
                'pred_token_id': pred_token_id,
                'answer_ids': answer_ids,
                'filter_tokens': filter_tokens,
                'logits': logits
            }
            results.append(result)
        
        logger.debug(f"Batch CoT reasoning completed for {len(results)} questions")
        return results
    
    def _answer_batch_directly(
        self,
        questions: List[str],
        options_list: List[List[str]],
        sampler
    ) -> List[Dict]:
        """Answer multiple questions directly without reasoning."""
        logger.debug("Starting batch direct answering")
        
        results = []
        
        # Process each question individually for direct answers
        # (Batching direct answers is simpler as no multi-step reasoning needed)
        for question, options in zip(questions, options_list):
            result = self._answer_directly(question, options, sampler)
            results.append(result)
        
        logger.debug(f"Batch direct answering completed for {len(results)} questions")
        return results
    
    def _extract_final_answer(self, generated_text: str) -> Optional[str]:
        """
        Extract final answer from generated text.
        
        Args:
            generated_text: Text generated by the model
            
        Returns:
            Extracted answer or None if not found
        """
        # First try regex pattern matching for "The final answer is option X"
        match = re.search(r"The final answer is option ([A-D])", generated_text)
        if match:
            return match.group(1)
        
        # Fallback: if it's a single token, return it
        if generated_text and len(generated_text.strip()) <= 3:
            return generated_text.strip()
        
        return None

 