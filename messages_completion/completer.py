"""
Message completer that orchestrates the completion pipeline.
"""

import logging
from typing import List, Dict, Union, Optional
from .config import CompletionConfig, SamplingMethod
from .processing import MessageProcessor
from .model import HuggingFaceModel
from .output_types import CompletionResult, BatchCompletionResult

logger = logging.getLogger(__name__)


class MessageCompleter:
    """
    Main interface for batched message completion.
    
    Takes a configuration and provides a clean interface for completing
    conversations with support for multiple sampling methods including DCBS.
    """
    
    def __init__(self, config: CompletionConfig):
        """
        Initialize the message completer.
        
        Args:
            config: CompletionConfig specifying model, sampling method, etc.
        """
        self.config = config
        
        # Initialize model
        self.model = HuggingFaceModel(
            model_name=config.model_name,
            device=config.device,
            load_in_4bit=config.load_in_4bit
        )
        
        # Set deterministic mode if requested
        if config.deterministic:
            self.model.set_seed(42)
        
        # Initialize message processor
        self.processor = MessageProcessor()
        
        # Initialize sampling interface based on method
        self._init_sampling()
        
        logger.info(f"MessageCompleter initialized: {config}")
    
    def _init_sampling(self):
        """Initialize the appropriate sampling interface."""
        method = self.config.sampling_method
        
        if method == SamplingMethod.DCBS:
            from .samplers.dcbs import DCBSSampler
            self.sampler = DCBSSampler(**self.config.sampling_params)
            logger.info("DCBS sampler initialized")
        else:
            self.sampler = None
    
    def complete(
        self,
        conversations: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        batch_size: Optional[int] = None
    ) -> Union[CompletionResult, BatchCompletionResult]:
        """
        Complete message conversations.
        
        Args:
            conversations: Single conversation or list of conversation message lists
            batch_size: Override config batch size if provided
            
        Returns:
            CompletionResult for single conversation, BatchCompletionResult for multiple
        """
        # Always treat as batch (single conversation = batch of 1)
        if isinstance(conversations, list) and len(conversations) > 0 and isinstance(conversations[0], dict):
            # Single conversation - convert to batch of 1
            conversations = [conversations]
            single_mode = True
        else:
            # Already a batch
            single_mode = False
        
        # Validate max_new_tokens and context length
        self._validate_token_limits(conversations)
        
        # Use provided batch_size or config default
        effective_batch_size = batch_size or self.config.batch_size
        
        # Process through single batched path
        results = self._complete_batch(conversations, effective_batch_size)
        
        # Return single result if input was single conversation
        if single_mode:
            return results.completions[0]
        else:
            return results
    
    def _validate_token_limits(self, conversations: List[List[Dict[str, str]]]):
        """Validate token limits for all conversations."""
        # Hard limit on max_new_tokens
        MAX_NEW_TOKENS = 4096
        if self.config.max_new_tokens > MAX_NEW_TOKENS:
            raise ValueError(
                f"max_new_tokens {self.config.max_new_tokens} exceeds maximum allowed {MAX_NEW_TOKENS}"
            )
        
        # Check context length for each conversation
        model_ctx_length = getattr(self.model, 'context_length', 8192)  # Default fallback
        
        for i, conversation in enumerate(conversations):
            formatted_prompt = self.processor.format_messages(conversation)
            input_tokens = len(self.model.tokenize([formatted_prompt])[0])
            
            total_tokens = input_tokens + self.config.max_new_tokens
            if total_tokens > model_ctx_length:
                raise ValueError(
                    f"Conversation {i}: total tokens ({total_tokens}) exceeds model context length ({model_ctx_length}). "
                    f"Input: {input_tokens}, max_new_tokens: {self.config.max_new_tokens}"
                )
    
    def _complete_batch(
        self,
        conversations: List[List[Dict[str, str]]],
        batch_size: Optional[int] = None
    ) -> BatchCompletionResult:
        """Complete multiple conversations in batches."""
        all_completions = []
        
        # Process in batches if batch_size specified
        if batch_size and len(conversations) > batch_size:
            for i in range(0, len(conversations), batch_size):
                batch_convs = conversations[i:i + batch_size]
                batch_completions = self._process_batch(batch_convs)
                all_completions.extend(batch_completions)
        else:
            all_completions = self._process_batch(conversations)
        
        return BatchCompletionResult(
            completions=all_completions,
            batch_size=len(all_completions),
            model_name=self.model.model_name,
            sampling_method=self.config.sampling_method.value
        )
    
    def _process_batch(self, conversations: List[List[Dict[str, str]]]) -> List[CompletionResult]:
        """Process a single batch of conversations."""
        # Format all prompts
        formatted_prompts = [
            self.processor.format_messages(conv) 
            for conv in conversations
        ]
        
        # Tokenize batch
        inputs = self.model.tokenize(formatted_prompts)
        
        # Generate based on sampling method
        if self.config.sampling_method == SamplingMethod.GREEDY:
            token_sequences, logprob_sequences = self.model.generate(
                inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                return_logprobs=self.config.return_logprobs
            )
        elif self.config.sampling_method == SamplingMethod.TOP_P:
            params = self.config.sampling_params
            token_sequences, logprob_sequences = self.model.generate(
                inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=params["temperature"],
                top_p=params["p"],
                return_logprobs=self.config.return_logprobs
            )
        elif self.config.sampling_method == SamplingMethod.DCBS:
            # Use DCBS sampling
            token_sequences, logprob_sequences = self.sampler.sample(
                self.model,
                inputs,
                max_new_tokens=self.config.max_new_tokens,
                return_logprobs=self.config.return_logprobs
            )
        
        # Create completions
        completions = []
        for i, (tokens, conversation, prompt) in enumerate(
            zip(token_sequences, conversations, formatted_prompts)
        ):
            completion_text = self.model.detokenize(tokens)
            logprobs = logprob_sequences[i] if logprob_sequences else None
            
            result = CompletionResult(
                text=completion_text,
                token_ids=tokens if self.config.return_token_ids else None,
                logprobs=logprobs,
                model_name=self.model.model_name,
                sampling_method=self.config.sampling_method.value,
                input_messages=conversation,
                formatted_prompt=prompt
            )
            completions.append(result)
        
        return completions