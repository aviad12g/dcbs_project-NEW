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
    ) -> Tuple[List[str], List[Optional[Cache]], torch.Tensor]:
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
        
        # STEP 3: Pre-computation to get initial logits
        with torch.no_grad():
            initial_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            batch_logits = initial_outputs.logits[:, -1, :]
            current_cache = initial_outputs.past_key_values

        # Initialize sequences and attention mask
        generated_sequences = input_ids.clone()
        current_attention_mask = attention_mask.clone()

        # Track which sequences are still generating and store final logits
        active_sequences = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        final_logits = torch.zeros_like(batch_logits)

        with torch.no_grad():
            for step in range(max_new_tokens):
                # In the first step (step=0), we use the pre-computed logits.
                # For subsequent steps, we compute them inside the loop.
                if step > 0:
                    last_tokens = generated_sequences[:, -1:]
                    model_inputs = {
                        "input_ids": last_tokens,
                        "attention_mask": current_attention_mask,
                        "past_key_values": current_cache,
                        "use_cache": True
                    }
                    outputs = self.model(**model_inputs)
                    batch_logits = outputs.logits[:, -1, :]
                    current_cache = outputs.past_key_values
                
                # PARALLEL SAMPLING
                next_tokens = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
                active_logits = batch_logits[active_sequences]
                
                if hasattr(sampler, 'sample_batch'):
                    sampled_tokens = sampler.sample_batch(active_logits)
                    next_tokens[active_sequences] = torch.tensor(
                        sampled_tokens, device=self.device
                    ).unsqueeze(1)
                else:
                    # Fallback for non-batch samplers
                    for i, logit_tensor in enumerate(active_logits):
                        active_idx = torch.where(active_sequences)[0][i]
                        token = sampler.sample(logit_tensor)
                        next_tokens[active_idx] = token
                
                # Update generated sequences and attention mask
                generated_sequences = torch.cat([generated_sequences, next_tokens], dim=1)
                new_attention = torch.ones(batch_size, 1, dtype=torch.long, device=self.device)
                current_attention_mask = torch.cat([current_attention_mask, new_attention], dim=1)
                
                # Store logits for sequences that just finished
                just_finished_mask = active_sequences & (next_tokens.squeeze(1) == self.tokenizer.eos_token_id)
                if just_finished_mask.any():
                    final_logits[just_finished_mask] = batch_logits[just_finished_mask]
                
                active_sequences &= ~just_finished_mask
                
                if not active_sequences.any():
                    logger.debug(f"All sequences finished at step {step}")
                    break
            
            # Store logits for any sequences that were still active when the loop ended
            if active_sequences.any():
                final_logits[active_sequences] = batch_logits[active_sequences]
        
        # STEP 4: Decode generated sequences
        responses = []
        for i in range(batch_size):
            input_length = attention_mask[i].sum().item()
            generated_ids = generated_sequences[i, input_length:].tolist()
            
            if self.tokenizer.eos_token_id in generated_ids:
                eos_idx = generated_ids.index(self.tokenizer.eos_token_id)
                generated_ids = generated_ids[:eos_idx]
            
            responses.append(self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
        
        logger.debug(f"TRUE batch generation completed. Generated {len(responses)} responses")
        
        return responses, [None] * batch_size, final_logits

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

    def get_logits_for_prompts_batch(
        self,
        prompts: List[str],
    ) -> torch.Tensor:
        """Get next-token logits for a batch of prompts.

        Args:
            prompts: List of input prompts

        Returns:
            Tensor of shape [batch_size, vocab_size] with next-token logits
        """
        batch = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_context_length
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**batch)
            logits = outputs.logits[:, -1, :]
        return logits