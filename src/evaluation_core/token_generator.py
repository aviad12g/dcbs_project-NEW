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
        Generate responses for multiple conversations using TRUE PARALLEL BATCHING.
        
        This implementation processes all messages simultaneously in the model,
        using proper attention masking and parallel token generation.
        
        Args:
            messages_batch: List of conversation message lists
            sampler: Sampler to use for token generation
            max_new_tokens: Maximum tokens to generate per conversation
            
        Returns:
            Tuple of (generated_texts, kv_caches) - one per conversation
        """
        logger.debug(f"Starting TRUE batch generation for {len(messages_batch)} conversations")
        
        batch_size = len(messages_batch)
        if batch_size == 0:
            return [], []
        
        # STEP 1: Format and tokenize ALL messages together
        prompts = []
        for messages in messages_batch:
            prompt = self._format_messages(messages, add_generation_prompt=True)
            prompts.append(prompt)
        
        logger.debug(f"Formatted {len(prompts)} prompts for batch processing")
        
        # STEP 2: Tokenize with padding for batch processing
        batch_encoding = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_context_length
        ).to(self.device)
        
        input_ids = batch_encoding.input_ids  # [batch_size, seq_len]
        attention_mask = batch_encoding.attention_mask  # [batch_size, seq_len]
        
        logger.debug(f"Tokenized batch: input_ids shape {input_ids.shape}")
        
        # STEP 3: TRUE PARALLEL token-by-token generation
        generated_sequences = input_ids.clone()  # Start with input tokens
        current_attention_mask = attention_mask.clone()
        current_cache = None
        
        # Track which sequences are still generating (not finished)
        active_sequences = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Prepare model inputs for PARALLEL processing
                if step == 0:
                    # First step: process full prompts in parallel
                    model_inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "past_key_values": None,
                        "use_cache": True
                    }
                else:
                    # Subsequent steps: process last tokens for all active sequences
                    last_tokens = generated_sequences[:, -1:] # [batch_size, 1]
                    last_attention = torch.ones_like(last_tokens, dtype=torch.bool)
                    
                    model_inputs = {
                        "input_ids": last_tokens,
                        "attention_mask": last_attention,
                        "past_key_values": current_cache,
                        "use_cache": True
                    }
                
                # PARALLEL MODEL FORWARD PASS
                outputs = self.model(**model_inputs)
                batch_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
                current_cache = outputs.past_key_values
                
                # PARALLEL SAMPLING using custom sampler
                if hasattr(sampler, 'sample_batch'):
                    # Use batch sampling if available
                    next_tokens = sampler.sample_batch(
                        batch_logits,
                        filter_tokens_batch=[None] * batch_size,  # No filtering for now
                        context=getattr(sampler, 'context', None)
                    )
                    next_tokens_tensor = torch.tensor(next_tokens, device=self.device).unsqueeze(1)
                else:
                    # Fallback: sample each sequence individually but in parallel
                    next_tokens = []
                    for i in range(batch_size):
                        if active_sequences[i]:
                            token = sampler.sample(batch_logits[i])
                            next_tokens.append(token)
                        else:
                            next_tokens.append(self.tokenizer.eos_token_id)
                    next_tokens_tensor = torch.tensor(next_tokens, device=self.device).unsqueeze(1)
                
                # Update generated sequences
                generated_sequences = torch.cat([generated_sequences, next_tokens_tensor], dim=1)
                
                # Update attention mask
                new_attention = torch.ones(batch_size, 1, dtype=torch.bool, device=self.device)
                current_attention_mask = torch.cat([current_attention_mask, new_attention], dim=1)
                
                # Check for EOS tokens and update active sequences
                eos_mask = (next_tokens_tensor.squeeze(1) == self.tokenizer.eos_token_id)
                active_sequences = active_sequences & ~eos_mask
                
                # Stop if all sequences are done
                if not active_sequences.any():
                    logger.debug(f"All sequences finished at step {step}")
                    break
                
                if step % 20 == 0 and step > 0:
                    active_count = active_sequences.sum().item()
                    logger.debug(f"Step {step}: {active_count}/{batch_size} sequences still generating")
        
        # STEP 4: Decode generated sequences and extract individual caches
        responses = []
        caches = []
        
        for i in range(batch_size):
            # Extract tokens generated for this sequence (excluding input)
            input_length = attention_mask[i].sum().item()
            generated_tokens = generated_sequences[i, input_length:].tolist()
            
            # Remove EOS token if present
            if self.tokenizer.eos_token_id in generated_tokens:
                eos_idx = generated_tokens.index(self.tokenizer.eos_token_id)
                generated_tokens = generated_tokens[:eos_idx]
            
            # Decode to text
            if generated_tokens:
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            else:
                generated_text = ""
            
            responses.append(generated_text.strip())
            
            # NOTE: Individual caches are complex to extract from batched cache
            # For now, we'll return None and rely on the full context approach
            caches.append(None)
        
        logger.debug(f"TRUE batch generation completed. Generated {len(responses)} responses")
        logger.debug(f"Response lengths: {[len(r) for r in responses]}")
        
        return responses, caches

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