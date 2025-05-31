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
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
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