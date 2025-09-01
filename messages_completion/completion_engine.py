"""
Core completion engine for the messages completion module.

Orchestrates message processing, model inference, and sampling to produce completions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Any, Set
import time
import logging
from .output_types import CompletionResult, BatchCompletionResult, TokenInfo, Messages
from .processing import MessageProcessor, MessageBatch
from .model import HuggingFaceModel
from .samplers import create_sampler

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
            model_interface = HuggingFaceModel(
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
        
        # Initialize sampler only for DCBS; standard methods use model.generate directly
        self.sampler = create_sampler(
            self.config.sampling_method,
            self.config.sampling_params
        )
        self._method_name = (
            self.sampler.method_name if self.sampler else self.config.sampling_method
        )
        
        # Setup stop tokens
        self._setup_stop_tokens()
        
        logger.info(f"CompletionEngine initialized with {self._method_name} sampling")
    
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
        
        # Generate completion (standard methods via model.generate; DCBS via sampler)
        text, token_ids, logprobs = self._generate_completion_via_model(formatted_prompt)
        
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
            sampling_method=self._method_name,
            generation_time=generation_time,
            input_messages=messages if self.config.include_input_context else None,
            formatted_prompt=formatted_prompt if self.config.include_input_context else None,
            metadata={
                "config": self.config.__dict__,
                "sampling_params": (self.sampler.get_parameters() if self.sampler else (self.config.sampling_params or {}))
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
        
        # Format prompts and use model.generate or DCBS sampler in batch
        batch_size = len(message_batch)
        formatted_prompts = self.message_processor.format_batch(message_batch)
        inputs = self.model_interface.tokenize(formatted_prompts)

        if self.sampler:
            token_ids_list, logprobs_list = self.sampler.generate(
                self.model_interface,
                inputs,
                max_new_tokens=self.config.max_new_tokens,
                return_logprobs=self.config.include_logprobs or self.config.include_token_info,
            )
        else:
            do_sample = self.config.sampling_method.lower() in ("top_p", "nucleus")
            params = self.config.sampling_params or {}
            temperature = params.get("temperature", 1.0)
            top_p = params.get("p", 1.0)
            token_ids_list, logprobs_list = self.model_interface.generate(
                inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                return_logprobs=self.config.include_logprobs or self.config.include_token_info,
            )

        completions: List[CompletionResult] = []
        for i, token_ids in enumerate(token_ids_list):
            text = self.model_interface.detokenize(token_ids)
            lp = (logprobs_list[i] if (logprobs_list and i < len(logprobs_list)) else None)
            completion = CompletionResult(
                text=text,
                token_ids=token_ids if self.config.include_token_info else None,
                token_info=None,
                logprobs=lp if self.config.include_logprobs else None,
                model_name=self.model_interface.model_name,
                sampling_method=self._method_name,
                input_messages=message_batch[i] if self.config.include_input_context else None,
                formatted_prompt=formatted_prompts[i] if self.config.include_input_context else None,
            )
            completions.append(completion)
        
        # Calculate total generation time
        total_generation_time = time.time() - start_time
        
        # Create batch result
        result = BatchCompletionResult(
            completions=completions,
            batch_size=batch_size,
            total_generation_time=total_generation_time,
            model_name=self.model_interface.model_name,
            sampling_method=self._method_name,
            metadata={
                "config": self.config.__dict__,
                "sampling_params": (self.sampler.get_parameters() if self.sampler else (self.config.sampling_params or {})),
                "effective_batch_size": self.config.batch_size or batch_size,
            }
        )
        
        return result
    
    def _generate_completion_via_model(self, formatted_prompt: str) -> tuple[str, List[int], List[float]]:
        """Generate completion for a single prompt via model.generate or DCBS sampler."""
        inputs = self.model_interface.tokenize([formatted_prompt])
        if self.sampler:
            token_ids_list, logprobs_list = self.sampler.generate(
                self.model_interface,
                inputs,
                max_new_tokens=self.config.max_new_tokens,
                return_logprobs=self.config.include_logprobs or self.config.include_token_info,
            )
        else:
            do_sample = self.config.sampling_method.lower() in ("top_p", "nucleus")
            params = self.config.sampling_params or {}
            temperature = params.get("temperature", 1.0)
            top_p = params.get("p", 1.0)
            token_ids_list, logprobs_list = self.model_interface.generate(
                inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                return_logprobs=self.config.include_logprobs or self.config.include_token_info,
            )

        token_ids = token_ids_list[0] if token_ids_list else []
        text = self.model_interface.detokenize(token_ids)
        logprobs = logprobs_list[0] if logprobs_list else []
        return text, token_ids, logprobs
    
    def _generate_batch_completions(self, message_batch: MessageBatch) -> List[CompletionResult]:
        """Legacy helper not used anymore (kept for backward compatibility)."""
        # Re-route through complete_batch to ensure a single code path
        result = self.complete_batch(message_batch)
        return list(result)
    
    def _generate_sequential_completions(self, message_batch: MessageBatch, chunk_size: int) -> List[CompletionResult]:
        """Legacy helper not used anymore; use complete_batch instead."""
        return list(self.complete_batch(message_batch))
    
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
        self.sampler = create_sampler(method, params)
        self._method_name = self.sampler.method_name if self.sampler else method
        logger.info(f"Updated sampling method to {self._method_name}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_interface.model_name,
            "vocab_size": self.model_interface.vocab_size,
            "device": str(self.model_interface.device),
            "sampling_method": self._method_name,
            "sampling_params": (self.sampler.get_parameters() if self.sampler else (self.config.sampling_params or {}))
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching statistics if available."""
        if self.sampler and hasattr(self.sampler, 'get_cache_stats'):
            return self.sampler.get_cache_stats()
        return {}
    
    def clear_caches(self):
        """Clear any caches."""
        if self.sampler and hasattr(self.sampler, 'clear_cache'):
            self.sampler.clear_cache()
    
    def __repr__(self) -> str:
        """String representation of the completion engine."""
        return (f"CompletionEngine(model='{self.model_interface.model_name}', "
                f"sampling='{self._method_name}')")
