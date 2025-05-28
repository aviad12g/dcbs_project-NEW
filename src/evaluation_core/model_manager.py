"""
Model loading and management without ChatTemplateManager.

This module handles model loading using the default tokenizer.chat_template
for most relevant instruct models, as suggested in the code review.
"""

import os
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dcbs import SamplingContext
from src.errors import eval_logger as logger


class ModelManager:
    """Model manager using default chat templates."""

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
            
        # Use default chat template - most relevant instruct models have this
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
            logger.info("Using model's default chat template")
        else:
            logger.warning(f"Model {self.model_name} does not have a default chat template")
            # Add a fallback template for models without chat templates
            if "llama" in self.model_name.lower() and "3" in self.model_name:
                # Llama 3 style template
                fallback_template = (
                    "{% if messages[0]['role'] == 'system' %}"
                    "{% set loop_messages = messages[1:] %}"
                    "{% set system_message = messages[0]['content'] %}"
                    "{% else %}"
                    "{% set loop_messages = messages %}"
                    "{% set system_message = false %}"
                    "{% endif %}"
                    "{% for message in loop_messages %}"
                    "{% if loop.index0 == 0 and system_message %}"
                    "{{ '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n' + system_message + '<|eot_id|>' }}"
                    "{% endif %}"
                    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
                    "{% if loop.last and add_generation_prompt %}"
                    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
                    "{% endif %}"
                    "{% endfor %}"
                )
                self.tokenizer.chat_template = fallback_template
                logger.info("Applied Llama 3 fallback chat template")
            else:
                # Generic fallback template
                fallback_template = (
                    "{% for message in messages %}"
                    "{{ message['role'].title() + ': ' + message['content'] + '\\n' }}"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}"
                    "{{ 'Assistant: ' }}"
                    "{% endif %}"
                )
                self.tokenizer.chat_template = fallback_template
                logger.info("Applied generic fallback chat template")
            
        # Test the chat template to ensure it works
        try:
            test_messages = [{"role": "user", "content": "Hello"}]
            test_result = self.tokenizer.apply_chat_template(
                test_messages, tokenize=False, add_generation_prompt=True
            )
            logger.info("Chat template validation successful")
            logger.debug(f"Test template result: {test_result[:100]}...")
        except Exception as e:
            logger.error(f"Chat template validation failed: {e}")
            raise RuntimeError(f"Model {self.model_name} has incompatible chat template")

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