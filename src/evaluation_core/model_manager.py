"""
Model loading and management functionality.

This module handles model loading, device management, and context creation
for evaluation runs.
"""

import os
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dcbs import SamplingContext
from src.errors import eval_logger as logger
from src.evaluation_core.template_manager import ChatTemplateManager


class ModelManager:
    """Handles model loading and management separately from evaluation logic."""

    def __init__(self, model_name: str, load_in_4bit: bool = False):
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        self.device = None
        self.context = None

    def load_model(self) -> Tuple[object, object, SamplingContext]:
        """
        Load the model, tokenizer, and create sampling context.
        
        Returns:
            Tuple of (model, tokenizer, sampling_context)
        """
        logger.info(f"Loading model: {self.model_name}")

        # Set dtype based on CUDA support
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

        model_kwargs = {
            "token": os.environ.get("HF_HUB_TOKEN"),
            "device_map": "auto",
            "torch_dtype": dtype,
        }

        # Add 4-bit quantization if requested
        if self.load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs
        )

        # CRITICAL: Set model to evaluation mode for inference
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, token=os.environ.get("HF_HUB_TOKEN")
        )

        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Set up the chat template using the template manager
        ChatTemplateManager.setup_chat_template(self.tokenizer, self.model_name)
        
        # Validate that the chat template is working correctly
        if not ChatTemplateManager.validate_template(self.tokenizer, self.model_name):
            logger.warning(
                f"Chat template validation failed for {self.model_name}. "
                "This may cause issues with prompt formatting."
            )

        self.device = next(self.model.parameters()).device

        # Create sampling context
        self.context = SamplingContext(
            embedding_layer=self.model.get_input_embeddings(),
            tokenizer=self.tokenizer,
            device=self.device,
        )

        logger.info(f"Model loaded on device: {self.device}")
        logger.info(f"Model dtype: {next(self.model.parameters()).dtype}")
        return self.model, self.tokenizer, self.context 