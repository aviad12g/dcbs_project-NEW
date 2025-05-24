"""
Utility functions for loading language models and tokenizers.
"""

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_model_and_tokenizer(model_name=None, device="auto"):
    """Load a model and tokenizer from Hugging Face."""
    try:
        default_model = "meta-llama/Llama-3.2-1B-Instruct"
        model_name = model_name or default_model

        if model_name != default_model:
            print(f"Warning: Overriding model selection to use {default_model}")
            model_name = default_model

        print(f"Loading model: {model_name}")

        hf_token = os.environ.get("HF_HUB_TOKEN")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            model_name, token=hf_token, device_map=device
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

        print(f"Model loaded successfully: {model_name}")
        print(f"Model device: {next(model.parameters()).device}")

        return model, tokenizer

    except Exception as e:
        print(f"Error loading model: {e}")
        raise
