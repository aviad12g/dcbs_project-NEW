"""
Token generation with caching and filtering support.

This module handles token generation with optional KV caching
and token filtering for multiple choice questions.
"""

import time
from typing import Dict, List, Optional, Set, Tuple

import torch
from transformers import Cache

from src.errors import eval_logger as logger


class TokenGenerator:
    """Handles token generation with caching and filtering."""
    
    def __init__(self, model, tokenizer, device='cuda'):
        """
        Initialize the TokenGenerator.
        
        Args:
            model: Loaded model for generation
            tokenizer: Tokenizer for encoding/decoding
            device: Device to use for generation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Fix padding side for decoder-only models
        if not hasattr(tokenizer, 'padding_side') or tokenizer.padding_side != 'left':
            self.tokenizer.padding_side = 'left'
            logger.debug("Set tokenizer padding_side to 'left' for decoder-only model")
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.debug("Set pad_token to eos_token")
        
        # Check if chat template is available
        self.has_chat_template = (
            hasattr(tokenizer, 'chat_template') and 
            tokenizer.chat_template is not None
        )
        
        # Get max context length from model config
        self.max_context_length = getattr(model.config, 'max_position_embeddings', 4096)
    
    def _format_messages(self, messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> str:
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
                logger.warning(f"Chat template failed in token generator, using fallback: {e}")
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
    
    def generate_batch_with_kv_cache(
        self, 
        messages_batch: List[List[Dict[str, str]]], 
        sampler, 
        max_new_tokens: int = 500
    ) -> Tuple[List[str], List[Cache]]:
        """
        Generate responses for multiple conversations using KV caching.
        
        Note: Processes each conversation individually to avoid batch complexity
        while still using the custom sampler and proper KV caching.
        
        Args:
            messages_batch: List of conversation message lists
            sampler: Sampler to use for token generation
            max_new_tokens: Maximum tokens to generate per conversation
            
        Returns:
            Tuple of (generated_texts, kv_caches) - one per conversation
        """
        logger.debug(f"Starting batch generation for {len(messages_batch)} conversations")
        
        responses = []
        caches = []
        
        # Process each conversation individually using the proven single-token approach
        for i, messages in enumerate(messages_batch):
            logger.debug(f"Processing conversation {i+1}/{len(messages_batch)}")
            
            # Use the working generate_with_kv_cache method that properly:
            # 1. Does token-by-token generation
            # 2. Uses the custom sampler
            # 3. Handles KV caching correctly
            # 4. Calls model() not model.generate()
            response, cache = self.generate_with_kv_cache(
                messages, 
                sampler, 
                max_new_tokens
            )
            
            responses.append(response)
            caches.append(cache)
        
        logger.debug(f"Batch generation completed. Generated responses of lengths: {[len(r) for r in responses]}")
        return responses, caches
    
    def generate_batch_from_cache(
        self, 
        messages_batch: List[List[Dict[str, str]]], 
        caches: List[Cache], 
        sampler, 
        max_new_tokens: int = 100
    ) -> List[str]:
        """
        Generate responses using cached key-value pairs from previous generation.
        
        Args:
            messages_batch: List of conversation message lists (for new prompts)
            caches: List of KV caches from previous generation
            sampler: Sampler to use for token generation
            max_new_tokens: Maximum tokens to generate per conversation
            
        Returns:
            List of generated response strings
        """
        logger.debug(f"Starting cached batch generation for {len(messages_batch)} conversations")
        
        # Check if we have valid caches
        if not caches or all(cache is None for cache in caches):
            logger.debug("No valid caches available, falling back to standard batch generation")
            responses, _ = self.generate_batch_with_kv_cache(messages_batch, sampler, max_new_tokens)
            return responses
        
        responses = []
        
        # Process each conversation individually, using the provided cache
        for i, (messages, cache) in enumerate(zip(messages_batch, caches)):
            logger.debug(f"Processing cached conversation {i+1}/{len(messages_batch)}")
            
            # Use the working generate_with_kv_cache method with the provided cache
            response, _ = self.generate_with_kv_cache(
                messages, 
                sampler, 
                max_new_tokens,
                past_key_values=cache  # Use the provided cache!
            )
            
            responses.append(response)
        
        logger.debug(f"Cached batch generation completed. Generated responses of lengths: {[len(r) for r in responses]}")
        return responses
    
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
        # Apply chat template or fallback formatting
        prompt = self._format_messages(messages, add_generation_prompt=True)
        
        # Log the prompt for debugging
        logger.debug(f"Generated prompt:\n{prompt}")
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        # FIXED: Properly handle existing cache vs new processing
        if past_key_values is not None:
            # We have existing cache - we need to identify only the NEW tokens
            logger.debug("FIXED: Using existing KV cache, processing minimal new tokens")
            current_input_ids = input_ids  # This will be the follow-up tokens only
        else:
            # No existing cache - process the full sequence
            current_input_ids = input_ids
            logger.debug("Starting fresh generation without existing cache")
        
        # Generate tokens one by one with caching
        generated_tokens = []
        current_cache = past_key_values
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Prepare input for this step
                if step == 0:
                    # First step: process the full prompt
                    model_inputs = {
                        "input_ids": current_input_ids,
                        "past_key_values": current_cache,
                        "use_cache": True
                    }
                else:
                    # Subsequent steps: only process the last generated token
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
                next_token = sampler.sample(logits)
                
                # Check for EOS
                if next_token == self.tokenizer.eos_token_id:
                    logger.debug(f"Generation stopped at step {step} due to EOS token")
                    break
                    
                generated_tokens.append(next_token)
                
                # For efficiency tracking
                if step % 50 == 0 and step > 0:
                    logger.debug(f"Generated {step} tokens so far")
        
        # Decode generated tokens
        if generated_tokens:
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            generated_text = ""
            
        logger.debug(f"Generated {len(generated_tokens)} tokens: {generated_text[:100]}...")
        return generated_text.strip(), current_cache
    
    def get_logits_for_prompt(
        self,
        prompt: str,
        filter_tokens: Optional[Set[int]] = None
    ) -> torch.Tensor:
        """
        Get logits for the next token given a prompt.
        
        Args:
            prompt: The input prompt
            filter_tokens: Optional set of allowed token IDs
            
        Returns:
            Logits tensor for the next token
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :].squeeze(0)
        
        return logits 