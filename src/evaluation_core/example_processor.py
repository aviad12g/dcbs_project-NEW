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

    def _normalize_answer(self, ans: str) -> str:
        """Lower-case, strip and remove trailing punctuation for exact-match scoring."""
        return ans.strip().lower().rstrip(". ")

    # Back-compat wrapper that routes single example through the batched path
    def process_example(self, example: Dict, sampler, include_cot: bool = True) -> Dict:
        results = self.process_examples_batch([example], sampler, include_cot)
        return results[0] if results else {}

    def process_examples_batch(
        self, 
        examples: List[Dict], 
        sampler, 
        include_cot: bool = True
    ) -> List[Dict]:
        """
        Process multiple examples in parallel using batch processing.
        
        Args:
            examples: List of example data dictionaries
            sampler: Sampler to use for generation
            include_cot: Whether to include chain-of-thought reasoning
            
        Returns:
            List of processed example results
        """
        start_time = time.time()
        logger.debug(f"Starting batch processing for {len(examples)} examples")

        if not examples:
            return []

        # If dataset is open-ended, handle via batched prompt construction too
        if "options" not in examples[0]:
            questions = [ex["question"] for ex in examples]
            # Build messages batch
            messages_batch = [[{"role": "user", "content": q + "\nAnswer:"}] for q in questions]
            # Use batched generation
            gen_out = self.question_answerer.token_generator.generate_batch_with_kv_cache(
                messages_batch, sampler, max_new_tokens=64
            )
            if isinstance(gen_out, tuple) and len(gen_out) == 3:
                responses, _, _ = gen_out
            else:
                responses, _ = gen_out
            processed = []
            for ex, ans in zip(examples, responses):
                processed.append({
                    "id": ex.get("id", "unknown"),
                    "sentence": ex["question"],
                    "options": None,
                    "correct_answer": ex["answer"].strip(),
                    "generated_answer": ans,
                })
            return processed

        # STEP 1: Extract questions and options from all examples
        questions = []
        options_list = []
        example_metadata = []
        
        for example in examples:
            if "question" in example:
                sentence = example["question"]
                options = example["options"]
                # Robust parsing of correct option: supports 0-based, 1-based, or letter formats
                raw_co = example.get("correct_option", "1")
                correct_idx = None
                try:
                    if isinstance(raw_co, int):
                        # Prefer 0-based if in range, else treat as 1-based
                        if 0 <= raw_co < len(options):
                            correct_idx = raw_co
                        elif 1 <= raw_co <= len(options):
                            correct_idx = raw_co - 1
                    elif isinstance(raw_co, str):
                        s = raw_co.strip()
                        if s.isdigit():
                            v = int(s)
                            if 0 <= v < len(options):
                                correct_idx = v
                            elif 1 <= v <= len(options):
                                correct_idx = v - 1
                        elif len(s) == 1 and 'A' <= s.upper() <= chr(ord('A') + len(options) - 1):
                            correct_idx = ord(s.upper()) - ord('A')
                    # Fallback: use provided correct_answer text if available
                    if correct_idx is None:
                        ca = example.get("correct_answer")
                        if isinstance(ca, str) and ca in options:
                            correct_idx = options.index(ca)
                except Exception:
                    correct_idx = None

                if correct_idx is None:
                    raise ValueError(f"Unable to determine correct option index from value '{raw_co}' with {len(options)} choices")

                correct_answer = options[correct_idx]
                
                questions.append(sentence)
                options_list.append(options)
                example_metadata.append({
                    "id": example.get("id", "unknown"),
                    "sentence": sentence,
                    "options": options,
                    "correct_answer": correct_answer,
                    "correct_option": raw_co,
                    "correct_idx": correct_idx
                })
            else:
                raise ValueError("Example must have 'question' field")
        
        # STEP 2: Batch question answering
        answer_results = self.question_answerer.answer_questions_batch(
            questions, options_list, sampler, include_cot
        )
        
        # STEP 3: Combine results with example metadata
        processed_examples = []
        total_time = time.time() - start_time
        
        for metadata, answer_result in zip(example_metadata, answer_results):
            result = metadata.copy()
            
            # Extract results from batch answer
            result["cot_reasoning"] = answer_result.get("reasoning")
            result["answer_ids"] = answer_result["answer_ids"]
            result["filter_tokens"] = answer_result["filter_tokens"]
            result["correct_id"] = answer_result["answer_ids"][metadata["correct_answer"]]
            result["logits"] = answer_result["logits"]
            result["answer_probs"] = answer_result["answer_probs"]
            # Preserve token variants (if provided) for downstream debugging
            if "answer_token_variants" in answer_result:
                result["answer_token_variants"] = answer_result["answer_token_variants"]
            result["processing_time"] = answer_result.get("processing_time", total_time / len(examples))
            
            processed_examples.append(result)
        
        logger.debug(f"Batch processing completed for {len(processed_examples)} examples in {total_time:.2f}s")
        return processed_examples

    # Shims for tests expecting these helpers on ExampleProcessor
    def create_reasoning_messages(self, sentence: str, options: List[str]) -> List[Dict[str, str]]:
        return self.question_answerer.message_generator.create_reasoning_messages(sentence, options)

    def _get_answer_token_ids(self, options: List[str]) -> Dict[str, int]:
        return self.question_answerer.token_resolver.get_answer_token_ids(options)

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
        
        # Generate fresh logits / answers for each sampler independently using batch API
        sentence = processed_result["sentence"]
        options = processed_result["options"]  # None for open-ended datasets
        include_cot = processed_result.get("cot_reasoning") is not None and options is not None

        if options is None:
            # Open-ended single example via batched generation
            messages_batch = [[{"role": "user", "content": sentence + "\nAnswer:"}]]
            responses, _, _ = self.question_answerer.token_generator.generate_batch_with_kv_cache(
                messages_batch, sampler, max_new_tokens=64
            )
            pred_answer = responses[0]
            correct = self._normalize_answer(pred_answer) == self._normalize_answer(processed_result["correct_answer"])
            elapsed_ms = (time.time() - start_time) * 1000
            return {
                "sampler": sampler_name,
                "pred_id": None,
                "predicted_answer": pred_answer,
                "correct": correct,
                "elapsed_ms": elapsed_ms,
                "cluster_info": None,
            }

        # Multiple-choice: use batched direct/with-reasoning pipeline over one item
        batch_answers = self.question_answerer.answer_questions_batch(
            [sentence], [options], sampler, include_cot
        )
        answer_result = batch_answers[0]
        
        # Multiple-choice branch (existing logic)
        logits = answer_result["logits"]
        filter_tokens = answer_result["filter_tokens"]
        # Recompute correct_id against the current answer_ids to avoid tokenization drift
        current_answer_ids = answer_result.get("answer_ids", {})
        correct_answer_text = processed_result["correct_answer"]
        correct_id = current_answer_ids.get(correct_answer_text, processed_result["correct_id"])  # fallback to precomputed

        predicted_answer = answer_result.get("selected_answer")
        
        pred_id = answer_result.get("pred_token_id")
        if pred_id is None:
            logger.warning("pred_token_id is None, falling back to direct sampling")
            pred_id = sampler.sample(logits, filter_tokens=filter_tokens)

        cluster_info = None
        if hasattr(sampler, "get_cluster_history"):
            history = sampler.get_cluster_history()
            if history:
                cluster_info = history[-1]
        if hasattr(sampler, "clear_debug_data"):
            sampler.clear_debug_data()

        correct = (pred_id == correct_id)
        elapsed_ms = (time.time() - start_time) * 1000

        return {
            "sampler": sampler_name,
            "pred_id": pred_id,
            "predicted_answer": predicted_answer,
            "correct": correct,
            "elapsed_ms": elapsed_ms,
            "cluster_info": cluster_info,
        }

    def evaluate_batch_with_sampler(
        self, 
        processed_results: List[Dict], 
        sampler, 
        sampler_name: str
    ) -> List[Dict]:
        """
        Evaluate multiple processed examples with a specific sampler in batch.
        
        Args:
            processed_results: List of results from process_examples_batch
            sampler: Sampler to use
            sampler_name: Name for logging
            
        Returns:
            List of evaluation results, one per processed example
        """
        start_time = time.time()
        logger.debug(f"Starting batch evaluation with {sampler_name} for {len(processed_results)} examples")
        
        if not processed_results:
            return []

        # If open-ended dataset, evaluate one by one using evaluate_with_sampler
        if processed_results[0]["options"] is None:
            logger.debug("Open-ended batch detected – falling back to per-example evaluation for sampler batch.")
            return [self.evaluate_with_sampler(pr, sampler, sampler_name) for pr in processed_results]
        
        # Extract questions and options for batch processing
        questions = []
        options_list = []
        include_cot_flags = []
        
        for processed_result in processed_results:
            sentence = processed_result["sentence"]
            options = processed_result["options"]
            include_cot = processed_result.get("cot_reasoning") is not None
            
            questions.append(sentence)
            options_list.append(options)
            include_cot_flags.append(include_cot)
        
        # Check if all examples have the same CoT setting for efficient batching
        uniform_cot = all(flag == include_cot_flags[0] for flag in include_cot_flags)
        
        if uniform_cot:
            # All examples have same CoT setting - use efficient batch processing
            answer_results = self.question_answerer.answer_questions_batch(
                questions, options_list, sampler, include_cot=include_cot_flags[0]
            )
        else:
            # Mixed CoT settings - process individually for correctness
            answer_results = []
            for i, processed_result in enumerate(processed_results):
                sentence = processed_result["sentence"]
                options = processed_result["options"]
                include_cot = processed_result.get("cot_reasoning") is not None
                
                answer_result = self.question_answerer.answer_question(
                    sentence, options, sampler, include_cot
                )
                answer_results.append(answer_result)
        
        # Process results
        evaluation_results = []
        total_time = time.time() - start_time
        
        for processed_result, answer_result in zip(processed_results, answer_results):
            logits = answer_result["logits"]
            filter_tokens = answer_result["filter_tokens"]
            # Recompute correct_id using the same answer_ids used to produce pred_id
            current_answer_ids = answer_result.get("answer_ids", {})
            correct_answer_text = processed_result["correct_answer"]
            correct_id = current_answer_ids.get(correct_answer_text, processed_result["correct_id"])  # fallback

            predicted_answer = answer_result.get("selected_answer")
            pred_id = answer_result.get("pred_token_id")
            
            # Fallback: if pred_token_id is None, sample again (but this shouldn't happen)
            if pred_id is None:
                logger.warning("pred_token_id is None, falling back to direct sampling")
                pred_id = sampler.sample(logits, filter_tokens=filter_tokens)

            cluster_info = None
            if hasattr(sampler, "get_cluster_history"):
                history = sampler.get_cluster_history()
                if history:
                    cluster_info = history[-1]
            
            # Check correctness
            correct = (pred_id == correct_id)
            
            evaluation_result = {
                "sampler": sampler_name,
                "pred_id": pred_id,
                "predicted_answer": predicted_answer,
                "correct": correct,
                "elapsed_ms": (total_time / len(processed_results)) * 1000,  # Average per example
                "cluster_info": cluster_info,
            }
            evaluation_results.append(evaluation_result)
        
        # Clear debug data AFTER processing all examples
        if hasattr(sampler, "clear_debug_data"):
            sampler.clear_debug_data()
        
        logger.debug(f"Batch evaluation with {sampler_name} completed for {len(evaluation_results)} examples in {total_time:.2f}s")
        return evaluation_results
