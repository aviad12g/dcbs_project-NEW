"""
Utility functions for loading language models and tokenizers.
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfFolder


def load_model_and_tokenizer(model_name=None, device="auto", load_in_4bit=False):
    """Load a model and tokenizer from Hugging Face."""
    try:
        # Default fallback only if model_name is None
        if model_name is None:
            model_name = "meta-llama/Llama-3.2-1B-Instruct"
            print(f"No model specified, using default: {model_name}")

        print(f"Loading model: {model_name}")

        # Get HuggingFace token from multiple sources
        hf_token = os.environ.get("HF_HUB_TOKEN") or HfFolder.get_token()
        if hf_token:
            print(f"Using HuggingFace token: {hf_token[:10]}...")
        else:
            print("Warning: No HuggingFace token found")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set dtype based on CUDA support
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

        model_kwargs = {"token": hf_token, "device_map": device, "torch_dtype": dtype}

        # Add 4-bit quantization if requested
        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

        print(f"Model loaded successfully: {model_name}")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")

        return model, tokenizer

    except Exception as e:
        print(f"Error loading model: {e}")
        raise
