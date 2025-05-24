"""
Test to ensure that the model path is correctly set to meta-llama/Llama-3.2-1B-Instruct.
"""

import unittest
import sys
import torch
from pathlib import Path

# Add the parent directory to the path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.load_model import load_model_and_tokenizer


class TestModelPath(unittest.TestCase):
    """Test case for model path verification."""

    def test_default_model_path(self):
        """Test that the default model path is set to Llama-3.2-1B-Instruct."""
        # Mock the actual model loading to avoid trying to download the model
        original_from_pretrained = __import__(
            "transformers"
        ).AutoModelForCausalLM.from_pretrained

        model_name_used = None

        def mock_from_pretrained(model_name, **kwargs):
            nonlocal model_name_used
            model_name_used = model_name

            # Just return something to avoid errors
            class MockModel:
                def __init__(self):
                    self.mock_param = torch.nn.Parameter(torch.tensor([0.0]))
                    self.device = torch.device("cpu")

                def parameters(self):
                    yield self.mock_param

                def get_input_embeddings(self):
                    return torch.nn.Embedding(10, 10)

            return MockModel()

        # Also mock the tokenizer
        original_tokenizer = __import__("transformers").AutoTokenizer.from_pretrained

        def mock_tokenizer(model_name, **kwargs):
            class MockTokenizer:
                def __init__(self):
                    pass

            return MockTokenizer()

        # Patch both methods
        __import__("transformers").AutoModelForCausalLM.from_pretrained = (
            mock_from_pretrained
        )
        __import__("transformers").AutoTokenizer.from_pretrained = mock_tokenizer

        try:
            # Try to load with no model name specified
            model, tokenizer = load_model_and_tokenizer()

            # Check that the correct model path was used
            self.assertEqual(
                model_name_used,
                "meta-llama/Llama-3.2-1B-Instruct",
                "Default model path is not set to meta-llama/Llama-3.2-1B-Instruct",
            )

            # Try to load with a different model name
            model_name_used = None
            model, tokenizer = load_model_and_tokenizer("some-other-model")

            # Check that the correct model path was still used (should override)
            self.assertEqual(
                model_name_used,
                "meta-llama/Llama-3.2-1B-Instruct",
                "Model path override not working correctly",
            )
        finally:
            # Restore the original methods
            __import__("transformers").AutoModelForCausalLM.from_pretrained = (
                original_from_pretrained
            )
            __import__("transformers").AutoTokenizer.from_pretrained = (
                original_tokenizer
            )


if __name__ == "__main__":
    unittest.main()
