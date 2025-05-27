"""
Improved example processing with proper conversation flow.

This module implements the correct two-step conversation flow:
1. LLM completes assistant reasoning response
2. LLM completes assistant final answer response

Key improvements:
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

from dcbs import DCBSSampler, SamplingContext
from src.errors import eval_logger as logger


class ImprovedExampleProcessor:
    """Improved example processor with correct conversation flow and KV caching."""

    def __init__(self, model, tokenizer, context: SamplingContext):
        self.model = model
        self.tokenizer = tokenizer
        self.context = context
        self.device = context.device

    def create_reasoning_messages(self, sentence: str, options: List[str]) -> List[Dict[str, str]]:
        """Create messages for the reasoning step."""
        options_str = self._format_options(options)
        
        return [
            {
                "role": "system", 
                "content": "You are an LLM that thinks step by step before answering."
            },
            {
                "role": "user",
                "content": f"{sentence}\n\n{options_str}"
            }
        ]

    def create_final_answer_messages(
        self, 
        reasoning_messages: List[Dict[str, str]], 
        reasoning_response: str
    ) -> List[Dict[str, str]]:
        """Create messages for the final answer step."""
        # Build on the previous conversation
        messages = reasoning_messages.copy()
        
        # Add the assistant's reasoning response
        messages.append({
            "role": "assistant",
            "content": reasoning_response
        })
        
        # Add the user's final question
        messages.append({
            "role": "user", 
            "content": "So what's the final answer?"
        })
        
        return messages

    def _format_options(self, options: List[str]) -> str:
        """Format options with letter labels."""
        formatted = ""
        for i, option in enumerate(options):
            label = chr(ord("A") + i)
            formatted += f"{label}. {option}\n"
        return formatted.strip()

    def _get_answer_token_ids(self, options: List[str]) -> Dict[str, int]:
        """Get token IDs for answer letters, handling prefix spaces correctly."""
        answer_ids = {}
        
        for i, option in enumerate(options):
            label = chr(ord("A") + i)
            
            # Try different tokenization strategies
            # LLMs are typically trained to output words with prefix space
            candidates = [
                f" {label}",  # Most common: space + letter
                label,        # Raw letter
                f"{label}.",  # Letter with period
                f" {label}.", # Space + letter + period
            ]
            
            best_token_id = None
            
            for candidate in candidates:
                tokens = self.tokenizer.encode(candidate, add_special_tokens=False)
                if len(tokens) == 1:
                    best_token_id = tokens[0]
                    break
            
            if best_token_id is None:
                # Fallback: use first token of " A" format
                tokens = self.tokenizer.encode(f" {label}", add_special_tokens=False)
                best_token_id = tokens[0] if tokens else 0
                logger.warning(f"Using fallback tokenization for option {label}")
            
            answer_ids[option] = best_token_id
            
        return answer_ids

    def generate_with_kv_cache(
        self, 
        messages: List[Dict[str, str]], 
        sampler, 
        max_new_tokens: int = 500,
        past_key_values: Optional[Cache] = None
    ) -> Tuple[str, Cache]:
        """
        Generate response using KV caching for efficiency.
        
        Args:
            messages: Chat messages
            sampler: Sampler to use for token generation
            max_new_tokens: Maximum tokens to generate
            past_key_values: Previous KV cache to continue from
            
        Returns:
            Tuple of (generated_text, new_cache)
        """
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Log the prompt for debugging
        logger.debug(f"Generated prompt:\n{prompt}")
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        # If we have past cache, we only need to process new tokens
        if past_key_values is not None:
            # Extract only the new tokens since last generation
            # This requires careful handling - for now, process full sequence
            # TODO: Implement proper incremental processing
            pass
        
        # Generate tokens one by one with caching
        generated_tokens = []
        current_cache = past_key_values
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Prepare input (full sequence for first step, last token for subsequent)
                if step == 0:
                    model_inputs = {
                        "input_ids": input_ids,
                        "past_key_values": current_cache,
                        "use_cache": True
                    }
                else:
                    # Only process the last generated token
                    last_token = torch.tensor([[generated_tokens[-1]]], device=self.device)
                    model_inputs = {
                        "input_ids": last_token,
                        "past_key_values": current_cache,
                        "use_cache": True
                    }
                
                # Forward pass
                outputs = self.model(**model_inputs)
                logits = outputs.logits[:, -1, :].squeeze(0)
                current_cache = outputs.past_key_values
                
                # Sample next token
                if isinstance(sampler, DCBSSampler):
                    next_token = sampler.sample(logits, context=self.context)
                else:
                    next_token = sampler.sample(logits)
                
                # Check for EOS
                if next_token == self.tokenizer.eos_token_id:
                    break
                    
                generated_tokens.append(next_token)
                
                # Update input_ids for next iteration
                if step == 0:
                    input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=self.device)], dim=1)
        
        # Decode generated tokens
        if generated_tokens:
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            generated_text = ""
            
        return generated_text.strip(), current_cache

    def process_example(self, example: Dict, sampler, include_cot: bool = True) -> Dict:
        """
        Process a single example using the improved two-step conversation flow.
        
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

        if include_cot:
            # Step 1: Generate reasoning
            reasoning_messages = self.create_reasoning_messages(sentence, options)
            reasoning_response, reasoning_cache = self.generate_with_kv_cache(
                reasoning_messages, sampler, max_new_tokens=500  # Increased from 200
            )
            
            result["cot_reasoning"] = reasoning_response
            
            # Step 2: Generate final answer
            final_messages = self.create_final_answer_messages(reasoning_messages, reasoning_response)
            
            # Log final chat for debugging
            logger.debug("Final chat messages:")
            for i, msg in enumerate(final_messages):
                logger.debug(f"  {i+1}. {msg['role']}: {msg['content'][:100]}...")
            
            # For final answer, we want to constrain to just the answer format
            # Create the final prompt manually to ensure proper format
            final_prompt = self.tokenizer.apply_chat_template(
                final_messages, tokenize=False, add_generation_prompt=True
            )
            final_prompt += "The final answer is option"  # No space after "option"
            
            logger.debug(f"Final answer prompt: {final_prompt}")
            
            # Get logits for final answer
            inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :].squeeze(0)
        else:
            # Direct answer without reasoning
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides direct answers."
                },
                {
                    "role": "user", 
                    "content": f"{sentence}\n\n{self._format_options(options)}\n\nThe final answer is option"
                }
            ]
            
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :].squeeze(0)
            
            result["cot_reasoning"] = None

        # Get answer token mappings
        answer_ids = self._get_answer_token_ids(options)
        
        # Calculate answer probabilities
        all_probs = torch.softmax(logits, dim=0)
        answer_probs = {
            option: all_probs[token_id].item()
            for option, token_id in answer_ids.items()
        }

        result.update({
            "answer_ids": answer_ids,
            "filter_tokens": set(answer_ids.values()),
            "correct_id": answer_ids[correct_answer],
            "logits": logits,
            "answer_probs": answer_probs,
            "processing_time": time.time() - start_time
        })

        return result

    def evaluate_with_sampler(
        self, processed_result: Dict, sampler, sampler_name: str
    ) -> Dict:
        """
        Evaluate a processed example with a specific sampler.
        
        Args:
            processed_result: Result from process_example
            sampler: Sampler to use
            sampler_name: Name for logging
            
        Returns:
            Evaluation result
        """
        start_time = time.time()
        
        logits = processed_result["logits"]
        filter_tokens = processed_result["filter_tokens"]
        correct_id = processed_result["correct_id"]
        
        # Sample using the specified sampler
        if isinstance(sampler, DCBSSampler):
            pred_id = sampler.sample(logits, filter_tokens=filter_tokens, context=self.context)
        else:
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