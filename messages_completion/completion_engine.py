"""
Core completion engine for the messages completion module.

Orchestrates message processing, model inference, and sampling to produce completions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Any, Set
import time
import logging
from .output_types import CompletionResult, BatchCompletionResult, TokenInfo, Messages
from .message_processor import MessageProcessor, MessageBatch
from .model_interface import ModelInterface, HuggingFaceModelInterface
from .sampling_interface import SamplingInterface, create_sampling_interface

logger = logging.getLogger(__name__)


@dataclass
class CompletionConfig:
    """Configuration for completion generation."""
    
    # Generation parameters
    max_new_tokens: int = 50
    stop_tokens: Optional[List[int]] = None
    stop_strings: Optional[List[str]] = None
    
    # Sampling configuration
    sampling_method: str = "greedy"
    sampling_params: Dict[str, Any] = field(default_factory=dict)
    
    # Output configuration
    include_logprobs: bool = False
    include_token_info: bool = False
    include_input_context: bool = True
    
    # Performance settings
    batch_size: Optional[int] = None
    enable_caching: bool = True
    
    # Model settings
    model_name: Optional[str] = None
    device: Optional[str] = None
    load_in_4bit: bool = False
    
    # Chat template settings
    custom_template: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.sampling_method not in ["greedy", "top_p", "dcbs", "random"]:
            logger.warning(f"Unknown sampling method: {self.sampling_method}")


class CompletionEngine:
    """Main engine for generating completions from messages."""
    
    def __init__(
        self,
        model_interface: Optional[ModelInterface] = None,
        config: Optional[CompletionConfig] = None,
        **kwargs
    ):
        """
        Initialize completion engine.
        
        Args:
            model_interface: Model interface instance
            config: Completion configuration
            **kwargs: Additional configuration parameters
        """
        # Setup configuration
        if config is None:
            config = CompletionConfig(**kwargs)
        self.config = config
        
        # Initialize model interface
        if model_interface is None:
            if self.config.model_name is None:
                raise ValueError("Either model_interface or model_name must be provided")
            
            logger.info(f"Initializing HuggingFace model: {self.config.model_name}")
            model_interface = HuggingFaceModelInterface(
                model_name=self.config.model_name,
                device=self.config.device,
                load_in_4bit=self.config.load_in_4bit
            )
        
        self.model_interface = model_interface
        
        # Initialize message processor
        self.message_processor = MessageProcessor(
            model_name=self.model_interface.model_name,
            custom_template=self.config.custom_template
        )
        
        # Initialize sampling interface
        self.sampling_interface = create_sampling_interface(
            self.config.sampling_method,
            **self.config.sampling_params
        )
        
        # Setup stop tokens
        self._setup_stop_tokens()
        
        logger.info(f"CompletionEngine initialized with {self.sampling_interface.method_name} sampling")
    
    def _setup_stop_tokens(self):
        """Setup stop tokens from configuration."""
        stop_tokens = set()
        
        # Add configured stop tokens
        if self.config.stop_tokens:
            stop_tokens.update(self.config.stop_tokens)
        
        # Add EOS token
        if hasattr(self.model_interface, 'tokenizer'):
            eos_token_id = self.model_interface.tokenizer.eos_token_id
            if eos_token_id is not None:
                stop_tokens.add(eos_token_id)
        
        # Convert stop strings to token IDs
        if self.config.stop_strings and hasattr(self.model_interface, 'encode_text'):
            for stop_string in self.config.stop_strings:
                try:
                    token_ids = self.model_interface.encode_text(stop_string)
                    if len(token_ids) == 1:  # Only single-token stops for now
                        stop_tokens.add(token_ids[0])
                except Exception as e:
                    logger.warning(f"Failed to encode stop string '{stop_string}': {e}")
        
        self.stop_tokens = stop_tokens
        logger.debug(f"Stop tokens: {self.stop_tokens}")
    
    def complete_messages(self, messages: Messages) -> CompletionResult:
        """
        Generate completion for a single message sequence.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            CompletionResult with generated text and metadata
        """
        start_time = time.time()
        
        # Validate and format messages
        self.message_processor.validate_messages(messages)
        formatted_prompt = self.message_processor.format_messages(messages)
        
        # Generate completion
        text, token_ids, logprobs = self._generate_completion(formatted_prompt)
        
        # Calculate generation time
        generation_time = time.time() - start_time
        
        # Create token info if requested
        token_info = None
        if self.config.include_token_info:
            token_info = self._create_token_info(token_ids, logprobs)
        
        # Create result
        result = CompletionResult(
            text=text,
            token_ids=token_ids,
            token_info=token_info,
            logprobs=logprobs if self.config.include_logprobs else None,
            model_name=self.model_interface.model_name,
            sampling_method=self.sampling_interface.method_name,
            generation_time=generation_time,
            input_messages=messages if self.config.include_input_context else None,
            formatted_prompt=formatted_prompt if self.config.include_input_context else None,
            metadata={
                "config": self.config.__dict__,
                "sampling_params": self.sampling_interface.get_parameters()
            }
        )
        
        return result
    
    def complete_batch(self, message_batch: Union[MessageBatch, List[Messages]]) -> BatchCompletionResult:
        """
        Generate completions for a batch of message sequences.
        
        Args:
            message_batch: Batch of message sequences
            
        Returns:
            BatchCompletionResult with all completions
        """
        start_time = time.time()
        
        # Convert to MessageBatch if needed
        if isinstance(message_batch, list):
            message_batch = MessageBatch.from_multiple_messages(message_batch)
        
        # Determine batch processing strategy
        batch_size = len(message_batch)
        effective_batch_size = self.config.batch_size or batch_size
        
        if batch_size <= effective_batch_size and hasattr(self.model_interface, 'generate_logits_batch'):
            # Use true batch processing
            completions = self._generate_batch_completions(message_batch)
        else:
            # Use sequential processing with chunking
            completions = self._generate_sequential_completions(message_batch, effective_batch_size)
        
        # Calculate total generation time
        total_generation_time = time.time() - start_time
        
        # Create batch result
        result = BatchCompletionResult(
            completions=completions,
            batch_size=batch_size,
            total_generation_time=total_generation_time,
            model_name=self.model_interface.model_name,
            sampling_method=self.sampling_interface.method_name,
            metadata={
                "config": self.config.__dict__,
                "sampling_params": self.sampling_interface.get_parameters(),
                "effective_batch_size": effective_batch_size
            }
        )
        
        return result
    
    def _generate_completion(self, formatted_prompt: str) -> tuple[str, List[int], List[float]]:
        """Generate completion for a single prompt."""
        generated_ids = []
        logprobs = []
        current_prompt = formatted_prompt
        
        # Get sampling context for DCBS
        context = None
        if hasattr(self.model_interface, 'get_embedding_layer'):
            try:
                # Try to import DCBS from parent project if available
                from src.dcbs import SamplingContext
                context = SamplingContext(
                    embedding_layer=self.model_interface.get_embedding_layer(),
                    tokenizer=getattr(self.model_interface, 'tokenizer', None),
                    device=self.model_interface.device
                )
            except ImportError:
                logger.warning("DCBS not available, using sampling without context")
        
        # Generate tokens one by one
        for _ in range(self.config.max_new_tokens):
            # Get logits for current prompt
            logits = self.model_interface.generate_logits(current_prompt)
            
            # Sample next token
            next_token_id = self.sampling_interface.sample_token(
                logits, 
                context=context
            )
            
            # Calculate log probability if requested
            logprob = None
            if self.config.include_logprobs or self.config.include_token_info:
                import torch
                probs = torch.softmax(logits, dim=-1)
                logprob = torch.log(probs[next_token_id]).item()
            
            generated_ids.append(next_token_id)
            if logprob is not None:
                logprobs.append(logprob)
            
            # Check for stop tokens
            if next_token_id in self.stop_tokens:
                break
            
            # Update current prompt
            next_token_text = self.model_interface.decode_tokens([next_token_id])
            current_prompt += next_token_text
        
        # Decode generated text
        generated_text = self.model_interface.decode_tokens(generated_ids)
        
        return generated_text, generated_ids, logprobs
    
    def _generate_batch_completions(self, message_batch: MessageBatch) -> List[CompletionResult]:
        """Generate completions using true batch processing."""
        # Format all prompts
        formatted_prompts = self.message_processor.format_batch(message_batch)
        
        completions = []
        
        # Initialize batch state
        batch_size = len(message_batch)
        current_prompts = formatted_prompts.copy()
        batch_generated_ids = [[] for _ in range(batch_size)]
        batch_logprobs = [[] for _ in range(batch_size)]
        active_indices = list(range(batch_size))
        
        # Get sampling context for DCBS
        context = None
        if hasattr(self.model_interface, 'get_embedding_layer'):
            try:
                # Try to import DCBS from parent project if available
                from src.dcbs import SamplingContext
                context = SamplingContext(
                    embedding_layer=self.model_interface.get_embedding_layer(),
                    tokenizer=getattr(self.model_interface, 'tokenizer', None),
                    device=self.model_interface.device
                )
            except ImportError:
                pass
        
        # Generate tokens
        for step in range(self.config.max_new_tokens):
            if not active_indices:
                break
            
            # Get logits for active prompts
            active_prompts = [current_prompts[i] for i in active_indices]
            logits_batch = self.model_interface.generate_logits_batch(active_prompts)
            
            # Sample next tokens
            next_token_ids = self.sampling_interface.sample_batch(
                logits_batch,
                context=context
            )
            
            # Process each active sequence
            new_active_indices = []
            for local_idx, global_idx in enumerate(active_indices):
                next_token_id = next_token_ids[local_idx]
                
                # Calculate log probability if requested
                if self.config.include_logprobs or self.config.include_token_info:
                    import torch
                    logits = logits_batch[local_idx]
                    probs = torch.softmax(logits, dim=-1)
                    logprob = torch.log(probs[next_token_id]).item()
                    batch_logprobs[global_idx].append(logprob)
                
                batch_generated_ids[global_idx].append(next_token_id)
                
                # Check for stop tokens
                if next_token_id not in self.stop_tokens:
                    # Update prompt and keep active
                    next_token_text = self.model_interface.decode_tokens([next_token_id])
                    current_prompts[global_idx] += next_token_text
                    new_active_indices.append(global_idx)
            
            active_indices = new_active_indices
        
        # Create completion results
        for i in range(batch_size):
            generated_text = self.model_interface.decode_tokens(batch_generated_ids[i])
            
            # Create token info if requested
            token_info = None
            if self.config.include_token_info:
                token_info = self._create_token_info(batch_generated_ids[i], batch_logprobs[i])
            
            completion = CompletionResult(
                text=generated_text,
                token_ids=batch_generated_ids[i],
                token_info=token_info,
                logprobs=batch_logprobs[i] if self.config.include_logprobs else None,
                model_name=self.model_interface.model_name,
                sampling_method=self.sampling_interface.method_name,
                input_messages=message_batch[i] if self.config.include_input_context else None,
                formatted_prompt=formatted_prompts[i] if self.config.include_input_context else None
            )
            
            completions.append(completion)
        
        return completions
    
    def _generate_sequential_completions(self, message_batch: MessageBatch, chunk_size: int) -> List[CompletionResult]:
        """Generate completions using sequential processing with chunking."""
        completions = []
        
        # Process in chunks
        for i in range(0, len(message_batch), chunk_size):
            chunk_end = min(i + chunk_size, len(message_batch))
            chunk_messages = message_batch.message_sequences[i:chunk_end]
            
            # Process each message in the chunk
            for messages in chunk_messages:
                completion = self.complete_messages(messages)
                completions.append(completion)
        
        return completions
    
    def _create_token_info(self, token_ids: List[int], logprobs: List[float]) -> List[TokenInfo]:
        """Create detailed token information."""
        token_info = []
        
        for i, token_id in enumerate(token_ids):
            token_text = self.model_interface.decode_tokens([token_id])
            logprob = logprobs[i] if i < len(logprobs) else None
            prob = None
            
            if logprob is not None:
                import math
                prob = math.exp(logprob)
            
            info = TokenInfo(
                token_id=token_id,
                token_text=token_text,
                logprob=logprob,
                prob=prob
            )
            
            token_info.append(info)
        
        return token_info
    
    def update_sampling_method(self, method: str, **params):
        """Update the sampling method and parameters."""
        self.config.sampling_method = method
        self.config.sampling_params = params
        self.sampling_interface = create_sampling_interface(method, **params)
        logger.info(f"Updated sampling method to {self.sampling_interface.method_name}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_interface.model_name,
            "vocab_size": self.model_interface.vocab_size,
            "device": str(self.model_interface.device),
            "sampling_method": self.sampling_interface.method_name,
            "sampling_params": self.sampling_interface.get_parameters()
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching statistics if available."""
        if hasattr(self.sampling_interface, 'get_cache_stats'):
            return self.sampling_interface.get_cache_stats()
        return {}
    
    def clear_caches(self):
        """Clear any caches."""
        if hasattr(self.sampling_interface, 'clear_cache'):
            self.sampling_interface.clear_cache()
    
    def __repr__(self) -> str:
        """String representation of the completion engine."""
        return (f"CompletionEngine(model='{self.model_interface.model_name}', "
                f"sampling='{self.sampling_interface.method_name}')")