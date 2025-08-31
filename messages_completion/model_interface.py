"""
Model interface abstractions for the completion module.

Provides a unified interface for different model backends.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any, Union
import torch
import logging


logger = logging.getLogger(__name__)


class ModelInterface(ABC):
    """Abstract interface for language models with batch-safe, deterministic generation."""
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name/identifier of the model."""
        pass
    
    @abstractmethod
    def set_seed(self, seed: int) -> None:
        """Set random seed for deterministic generation."""
        pass
    
    @abstractmethod
    def tokenize(self, texts: List[str]) -> Dict[str, Any]:
        """
        Tokenize input texts for batch generation.
        
        Args:
            texts: List of input texts
            
        Returns:
            Dictionary with tokenized inputs (preserves order)
        """
        pass
    
    @abstractmethod
    def detokenize(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        inputs: Dict[str, Any],
        max_new_tokens: int,
        do_sample: bool,
        temperature: float = 1.0,
        top_p: float = 1.0,
        return_logprobs: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate completions with order-preserving, deterministic behavior.
        
        Args:
            inputs: Tokenized inputs from tokenize()
            max_new_tokens: Maximum tokens to generate
            do_sample: If False, use greedy sampling (ignore temp/top_p)
            temperature: Sampling temperature (ignored if do_sample=False)
            top_p: Nucleus sampling threshold (ignored if do_sample=False)
            return_logprobs: Whether to return log probabilities
            
        Returns:
            Tuple of (token_ids_per_sample, logprobs_per_sample_or_None)
            Order must match input order (no sorting/reordering)
        """
        pass


class HuggingFaceModelInterface(ModelInterface):
    """HuggingFace Transformers model interface with batch-safe, deterministic generation."""
    
    def __init__(
        self, 
        model_name: str,
        device: Optional[Union[str, torch.device]] = None,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
        **model_kwargs
    ):
        """
        Initialize HuggingFace model interface.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to load model on
            load_in_4bit: Whether to use 4-bit quantization
            trust_remote_code: Whether to trust remote code
            **model_kwargs: Additional arguments for model loading
        """
        self._model_name = model_name
        self._device = self._setup_device(device)
        
        # Import HuggingFace libraries
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError as e:
            raise ImportError(f"HuggingFace transformers required: {e}")
        
        # Load tokenizer
        logger.info(f"Loading tokenizer for {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup model loading arguments
        model_loading_kwargs = {
            "trust_remote_code": trust_remote_code,
            **model_kwargs
        }
        
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_loading_kwargs["quantization_config"] = quantization_config
                logger.info("Using 4-bit quantization")
            except ImportError:
                logger.warning("BitsAndBytes not available, loading without quantization")
        else:
            model_loading_kwargs["torch_dtype"] = torch.float16
            model_loading_kwargs["device_map"] = "auto" if torch.cuda.is_available() else None
        
        # Setup GPU determinism for batch invariance
        self._setup_determinism()
        
        # Load model with deterministic settings
        logger.info(f"Loading model {model_name}")
        model_loading_kwargs["attn_implementation"] = "eager"  # Avoid flash/SDPA for determinism
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_loading_kwargs
        )
        
        # Fallback if transformers ignores the kwarg
        try:
            self.model.config._attn_implementation = "eager"
        except Exception:
            pass
        
        # Move to device if not using device_map
        if "device_map" not in model_loading_kwargs or model_loading_kwargs["device_map"] is None:
            self.model = self.model.to(self._device)
        
        self.model.eval()
        
        logger.info(f"Model loaded successfully on {self.model.device}")
    
    def _setup_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        """Setup and validate device."""
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        
        return device
    
    def _setup_determinism(self):
        """Setup deterministic behavior for batch invariance."""
        # Disable TF32 for determinism
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = False
        
        # Enable deterministic algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        # Set CUDA deterministic behavior
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def set_seed(self, seed: int) -> None:
        """Set random seed for deterministic generation."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def tokenize(self, texts: List[str]) -> Dict[str, Any]:
        """Tokenize input texts for batch generation."""
        if not texts:
            raise ValueError("texts cannot be empty")
        
        # Tokenize without reordering - preserve input order
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            pad_to_multiple_of=8  # Optimize for tensor cores
        ).to(self.model.device)
        
        return inputs
    
    def detokenize(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def generate(
        self,
        inputs: Dict[str, Any],
        max_new_tokens: int,
        do_sample: bool,
        temperature: float = 1.0,
        top_p: float = 1.0,
        return_logprobs: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """Generate completions with deterministic, order-preserving behavior."""
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "output_scores": return_logprobs,
            "return_dict_in_generate": True,
        }
        
        # Only add sampling parameters if do_sample=True
        if do_sample:
            generation_kwargs.update({
                "temperature": temperature,
                "top_p": top_p,
            })
        
        # Generate without reordering
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        
        # Extract generated token IDs (remove input tokens)
        input_length = inputs["input_ids"].shape[1]
        generated_sequences = outputs.sequences[:, input_length:]
        
        # Convert to list of lists
        token_ids_list = [seq.tolist() for seq in generated_sequences]
        
        # Extract logprobs if requested
        logprobs_list = None
        if return_logprobs and hasattr(outputs, 'scores') and outputs.scores:
            input_len = inputs["input_ids"].shape[1]
            gen_ids = outputs.sequences[:, input_len:]  # [B, T]
            
            logprobs_steps = []
            for t, scores in enumerate(outputs.scores):
                # scores: [B, V] logits for step t
                lp = torch.log_softmax(scores, dim=-1)       # [B, V]
                step_ids = gen_ids[:, t].unsqueeze(1)        # [B, 1]
                step_lp = lp.gather(1, step_ids).squeeze(1)  # [B]
                logprobs_steps.append(step_lp)
            
            # [T, B] -> [B, T]
            logprobs_list = torch.stack(logprobs_steps, dim=0).transpose(0, 1).tolist()
        
        return token_ids_list, logprobs_list
    
    @property
    def model_name(self) -> str:
        """Name of the model."""
        return self._model_name
    

    
    def __repr__(self) -> str:
        """String representation of the model interface."""
        return f"HuggingFaceModelInterface(model_name='{self.model_name}', device='{self.device}')"