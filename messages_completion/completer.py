"""
Single entrypoint for message completion.

Provides MessageCompleter class that takes conversations and returns completions
with deterministic, batch-invariant behavior.
"""

from typing import List, Dict, Union, Optional
import copy

from .message_processor import MessageProcessor, MessageBatch
from .model_interface import ModelInterface
from .output_types import CompletionResult, BatchCompletionResult


class MessageCompleter:
    """
    Main class for converting conversations to completions.
    
    Provides deterministic, batch-invariant completion generation with
    optional token IDs and log probabilities.
    """
    
    def __init__(self, model: ModelInterface, max_new_tokens: int = 64):
        """
        Initialize message completer.
        
        Args:
            model: Model interface implementation
            max_new_tokens: Maximum tokens to generate per completion
        """
        self.model = model
        self.proc = MessageProcessor()
        self.max_new_tokens = max_new_tokens
    
    def complete(
        self, 
        conversations: List[List[Dict[str, str]]], 
        use_batching: bool = True, 
        return_logprobs: bool = False
    ) -> Union[CompletionResult, BatchCompletionResult]:
        """
        Complete conversations with deterministic, batch-invariant generation.
        
        Args:
            conversations: List of conversation sequences
            use_batching: Whether to use batch processing (should not affect results)
            return_logprobs: Whether to return log probabilities
            
        Returns:
            Single CompletionResult if one conversation, BatchCompletionResult otherwise
        """
        # Ensure we work with a copy to avoid modifying input
        conversations = copy.deepcopy(conversations)
        
        if use_batching:
            # Batch processing: single encode/generate call
            batch = MessageBatch(conversations)
            rendered = [self.proc.format_messages(seq) for seq in batch]  # preserves order
            
            # Tokenize inputs
            encoded = self.model.tokenize(rendered)
            
            # Generate completions - greedy by default for deterministic behavior
            ids, logps = self.model.generate(
                encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Greedy sampling for deterministic results
                return_logprobs=return_logprobs,
            )
        else:
            # Sequential processing: N times with batch=1
            ids = []
            logps = [] if return_logprobs else None
            
            for conversation in conversations:
                # Process single conversation
                rendered = [self.proc.format_messages(conversation)]
                encoded = self.model.tokenize(rendered)
                
                single_ids, single_logps = self.model.generate(
                    encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # Greedy sampling for deterministic results
                    return_logprobs=return_logprobs,
                )
                
                ids.extend(single_ids)
                if return_logprobs and single_logps:
                    logps.extend(single_logps)
        
        # Create completion results
        comps = []
        for i, token_ids in enumerate(ids):
            text = self.model.detokenize(token_ids)
            comps.append(CompletionResult(
                text=text,
                token_ids=token_ids,
                logprobs=(logps[i] if return_logprobs and logps is not None else None),
                model_name=self.model.model_name,
                sampling_method="greedy",
            ))
        
        # Return single result or batch result
        if len(comps) == 1:
            return comps[0]
        else:
            return BatchCompletionResult(
                completions=comps,
                batch_size=len(comps),
                model_name=self.model.model_name,
                sampling_method="greedy",
            )