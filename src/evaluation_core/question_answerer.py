"""
Question answering orchestration.

This module combines message templates and token generation
to answer multiple choice questions with or without reasoning.
"""

import time
from typing import Dict, List, Optional, Tuple

import torch

from src.dcbs import SamplingContext, GreedySampler
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
        # Use greedy for reasoning generation to standardize context and avoid
        # degrading reasoning quality with experimental samplers
        self._reasoning_sampler = GreedySampler()
    
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
        options: Optional[List[str]],
        sampler,
        include_cot: bool = True,
    ) -> Dict:
        """Compatibility wrapper that executes a one-item batch.

        Uses the batched implementations under the hood to keep a single
        high-throughput code path, while preserving the old API for callers.
        """
        # Multiple-choice path via batch API
        if options is not None:
            results = self.answer_questions_batch([question], [options], sampler, include_cot)
            return results[0]

        # Open-ended path (single), still uses batched generator with batch size = 1
        messages_batch = [[{"role": "user", "content": question.strip() + "\nAnswer:"}]]
        responses, _, _ = self.token_generator.generate_batch_with_kv_cache(
            messages_batch, sampler, max_new_tokens=64
        )
        return {
            "generated_answer": responses[0].strip(),
            "pred_token_id": None,
            "answer_ids": None,
            "filter_tokens": None,
            "logits": None,
        }
    
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
        
        reasoning_responses, _, _ = self.token_generator.generate_batch_with_kv_cache(
            [reasoning_messages], self._reasoning_sampler, max_new_tokens=512
        )
        reasoning_response = reasoning_responses[0]

        # Step 2: Create final answer prompt with an explicit anchor
        final_messages = reasoning_messages + [
            {"role": "assistant", "content": reasoning_response},
            {"role": "user", "content": "What is your final answer? Respond with just the letter (A, B, C, or D)."}
        ]
        
        # Format the final prompt
        final_prompt = self._format_prompt(final_messages, add_generation_prompt=True)
        
        # Get logits for answer selection at the specific prompt position
        # Ensure add_generation_prompt positions the logits at the next assistant token (the final letter)
        logits = self.token_generator.get_logits_for_prompt(final_prompt)
        
        # Step 3: Sample from answer tokens
        answer_ids = self.token_resolver.get_answer_token_ids(options)
        answer_variants = self.token_resolver.get_answer_token_variants(options)
        # Sanity: enforce 4 distinct token IDs to avoid degenerate selection
        if len(set(answer_ids.values())) != len(answer_ids):
            logger.warning(f"Duplicate answer token IDs detected: {answer_ids}")
        answer_probs = self._calculate_answer_probabilities(logits, answer_ids)

        filter_tokens = set(answer_ids.values())
        # Expand filter to include plausible single-token variants for each option
        for ids in answer_variants.values():
            for tid in ids:
                filter_tokens.add(tid)
        pred_token_id = sampler.sample(logits, filter_tokens=filter_tokens)

        selected_answer = None
        canonical_pred_id = pred_token_id
        for answer_text, token_id in answer_ids.items():
            if token_id == pred_token_id:
                selected_answer = answer_text
                canonical_pred_id = token_id
                break
        if selected_answer is None:
            # Try to map variant token back to its canonical answer id
            for answer_text, ids in answer_variants.items():
                if pred_token_id in ids:
                    selected_answer = answer_text
                    canonical_pred_id = answer_ids[answer_text]
                    break
        
        generated_text = self.tokenizer.decode(pred_token_id)
        final_answer = self._extract_final_answer(generated_text)

        parsed_reasoning = self.cot_parser.parse_cot_response(reasoning_response)
        cleaned_reasoning = parsed_reasoning['reasoning']
        
        if parsed_reasoning['is_template']:
            logger.debug(f"CoT response appears to be template text: {reasoning_response[:100]}...")
        elif not parsed_reasoning['is_valid']:
            logger.debug(f"CoT response quality is low: {reasoning_response[:100]}...")
        
        return {
            'selected_answer': final_answer or selected_answer,
            'reasoning': cleaned_reasoning,
            'raw_reasoning': reasoning_response,
            'reasoning_quality': parsed_reasoning,
            'answer_probs': answer_probs,
            'pred_token_id': canonical_pred_id,
            'answer_ids': answer_ids,
            'answer_token_variants': answer_variants,
            'filter_tokens': filter_tokens,
            'logits': logits
        }
    
    # Removed single open-ended flow; use batched APIs instead

    # Removed direct single-question path; rely on batch APIs
    


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
        """Answer multiple questions with reasoning using batched generation and batched selection."""
        logger.debug(f"Starting batch reasoning for {len(questions)} questions (batched)")

        # Step 1: Batched reasoning generation
        reasoning_messages_batch = [
            self.message_generator.create_reasoning_messages(q, opts)
            for q, opts in zip(questions, options_list)
        ]
        gen_out = self.token_generator.generate_batch_with_kv_cache(
            reasoning_messages_batch, self._reasoning_sampler, max_new_tokens=512
        )
        # Back-compat: allow 2- or 3-tuple returns
        if isinstance(gen_out, tuple) and len(gen_out) == 3:
            reasoning_responses, _, _ = gen_out
        else:
            reasoning_responses, _ = gen_out

        # Step 2: Build final prompts and get batched logits
        final_prompts: List[str] = []
        for rr, msgs in zip(reasoning_responses, reasoning_messages_batch):
            final_messages = msgs + [
                {"role": "assistant", "content": rr},
                {"role": "user", "content": "What is your final answer? Respond with just the letter (A, B, C, or D)."},
            ]
            final_prompts.append(self._format_prompt(final_messages, add_generation_prompt=True))

        logits_batch = self.token_generator.get_logits_for_prompts_batch(final_prompts)

        # Step 3: Resolve answer tokens and select with batch sampling (or fallback)
        answer_ids_list = [self.token_resolver.get_answer_token_ids(opts) for opts in options_list]
        variants_list = [self.token_resolver.get_answer_token_variants(opts) for opts in options_list]
        filter_tokens_batch = []
        for ans_ids, variants in zip(answer_ids_list, variants_list):
            s = set(ans_ids.values())
            for ids in variants.values():
                for tid in ids:
                    s.add(tid)
            filter_tokens_batch.append(s)

        if hasattr(sampler, "sample_batch"):
            pred_ids = sampler.sample_batch(logits_batch, filter_tokens_batch)
        else:
            pred_ids = []
            for i in range(len(questions)):
                pred_ids.append(sampler.sample(logits_batch[i], filter_tokens=filter_tokens_batch[i]))

        # Step 4: Build results
        results: List[Dict] = []
        for i, (options, answer_ids, pred_id, rr) in enumerate(
            zip(options_list, answer_ids_list, pred_ids, reasoning_responses)
        ):
            # Probabilities per option (restricted softmax) for this row
            answer_probs = self._calculate_answer_probabilities(logits_batch[i], answer_ids)

            # Canonicalize predicted token to primary id, inferring option via variants if needed
            selected_answer = None
            canonical_pred_id = pred_id
            for answer_text, token_id in answer_ids.items():
                if token_id == pred_id:
                    selected_answer = answer_text
                    canonical_pred_id = token_id
                    break
            if selected_answer is None:
                for answer_text, id_list in variants_list[i].items():
                    if pred_id in id_list:
                        selected_answer = answer_text
                        canonical_pred_id = answer_ids[answer_text]
                        break

            parsed_reasoning = self.cot_parser.parse_cot_response(rr)
            results.append(
                {
                    "selected_answer": selected_answer,
                    "reasoning": parsed_reasoning["reasoning"],
                    "raw_reasoning": rr,
                    "reasoning_quality": parsed_reasoning,
                    "answer_probs": answer_probs,
                    "pred_token_id": canonical_pred_id,
                    "answer_ids": answer_ids,
                    "filter_tokens": filter_tokens_batch[i],
                    "logits": logits_batch[i],
                }
            )

        return results
    
    def _answer_batch_directly(
        self,
        questions: List[str],
        options_list: List[List[str]],
        sampler
    ) -> List[Dict]:
        """Answer multiple questions directly using batched logits and batched sampling."""
        logger.debug("Starting batch direct answering (batched)")

        # Build prompts
        prompts: List[str] = []
        for q, opts in zip(questions, options_list):
            msgs = self.message_generator.create_direct_answer_messages(q, opts)
            prompt = self._format_prompt(msgs, add_generation_prompt=True)
            prompts.append(prompt)

        # Batched logits
        logits_batch = self.token_generator.get_logits_for_prompts_batch(prompts)

        # Resolve answer tokens and build filter sets
        answer_ids_list = [self.token_resolver.get_answer_token_ids(opts) for opts in options_list]
        variants_list = [self.token_resolver.get_answer_token_variants(opts) for opts in options_list]
        filter_tokens_batch = []
        for ans_ids, variants in zip(answer_ids_list, variants_list):
            s = set(ans_ids.values())
            for ids in variants.values():
                for tid in ids:
                    s.add(tid)
            filter_tokens_batch.append(s)

        # Batch sampling (preferred), fallback to per-row if sampler lacks batch API
        if hasattr(sampler, "sample_batch"):
            pred_ids = sampler.sample_batch(logits_batch, filter_tokens_batch)
        else:
            pred_ids = []
            for i in range(len(questions)):
                pred_ids.append(sampler.sample(logits_batch[i], filter_tokens=filter_tokens_batch[i]))

        # Build results
        results: List[Dict] = []
        for i, (answer_ids, pred_id) in enumerate(zip(answer_ids_list, pred_ids)):
            # Calculate per-option probabilities from restricted logits
            answer_probs = self._calculate_answer_probabilities(logits_batch[i], answer_ids)

            # Canonicalize predicted token to primary id via variants
            selected_answer = None
            canonical_pred_id = pred_id
            for answer_text, token_id in answer_ids.items():
                if token_id == pred_id:
                    selected_answer = answer_text
                    canonical_pred_id = token_id
                    break
            if selected_answer is None:
                for answer_text, id_list in variants_list[i].items():
                    if pred_id in id_list:
                        selected_answer = answer_text
                        canonical_pred_id = answer_ids[answer_text]
                        break

            results.append(
                {
                    "selected_answer": selected_answer,
                    "reasoning": None,
                    "answer_probs": answer_probs,
                    "pred_token_id": canonical_pred_id,
                    "answer_ids": answer_ids,
                    "filter_tokens": filter_tokens_batch[i],
                    "logits": logits_batch[i],
                }
            )

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

 