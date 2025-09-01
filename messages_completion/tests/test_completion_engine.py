"""Tests for the completion engine (simplified for direct model.generate path)."""

import unittest

from messages_completion.completion_engine import (
    CompletionEngine,
    CompletionConfig,
    CompletionResult,
    BatchCompletionResult,
    MessageBatch,
)


class MockModel:
    """Minimal mock model implementing tokenize/generate/detokenize."""

    def __init__(self):
        self._model_name = "test-model"
        self._device = "cpu"
        self._vocab_size = 1000

    def tokenize(self, texts):
        # Simulate HF batch inputs
        return {"input_ids": texts}

    def generate(self, inputs, max_new_tokens, do_sample, temperature=1.0, top_p=1.0, return_logprobs=False):
        batch = inputs["input_ids"]
        # Return deterministic 2-token output per prompt
        token_ids_list = [[1, 2] for _ in batch]
        logprobs = [[-0.1, -0.2] for _ in batch] if return_logprobs else None
        return token_ids_list, logprobs

    def detokenize(self, token_ids):
        return " ".join(f"tok{t}" for t in token_ids)

    @property
    def model_name(self):
        return self._model_name

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def device(self):
        return self._device


class TestCompletionConfig(unittest.TestCase):
    """Test completion configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CompletionConfig()
        
        self.assertEqual(config.max_new_tokens, 50)
        self.assertEqual(config.sampling_method, "greedy")
        self.assertFalse(config.include_logprobs)
        self.assertTrue(config.include_input_context)
        self.assertTrue(config.enable_caching)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CompletionConfig(
            max_new_tokens=100,
            sampling_method="dcbs",
            include_logprobs=True,
            model_name="test-model"
        )
        
        self.assertEqual(config.max_new_tokens, 100)
        self.assertEqual(config.sampling_method, "dcbs")
        self.assertTrue(config.include_logprobs)
        self.assertEqual(config.model_name, "test-model")
    
    def test_invalid_config(self):
        """Test invalid configuration values."""
        with self.assertRaises(ValueError):
            CompletionConfig(max_new_tokens=-1)
        
        with self.assertRaises(ValueError):
            CompletionConfig(batch_size=0)


class TestCompletionEngine(unittest.TestCase):
    """Test completion engine functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = MockModel()
        self.config = CompletionConfig(
            max_new_tokens=10,
            sampling_method="greedy",
            include_logprobs=True,
            include_token_info=True
        )
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = CompletionEngine(
            model_interface=self.mock_model,
            config=self.config
        )
        
        self.assertEqual(engine.model_interface, self.mock_model)
        self.assertEqual(engine.config, self.config)
        # For greedy defaults, no explicit sampler is used
        self.assertIsNone(engine.sampler)
        self.assertIsNotNone(engine.message_processor)
    
    def test_engine_initialization_with_model_name(self):
        """Test engine initialization with model name."""
        config = CompletionConfig(model_name="microsoft/DialoGPT-small")
        
        with patch('messages_completion.model.HuggingFaceModel') as mock_hf:
            mock_hf.return_value = self.mock_model
            engine = CompletionEngine(config=config)
            
            mock_hf.assert_called_once()
            self.assertEqual(engine.model_interface, self.mock_model)
    
    def test_single_completion(self):
        """Test single message completion."""
        engine = CompletionEngine(
            model_interface=self.mock_model,
            config=self.config
        )
        
        # Mock the sampling interface to return predictable tokens
        messages = [
            {"role": "user", "content": "Hello!"}
        ]
        
        result = engine.complete_messages(messages)
        
        self.assertIsInstance(result, CompletionResult)
        self.assertIsNotNone(result.text)
        self.assertIsInstance(result.token_ids, list)
        self.assertGreater(len(result.token_ids), 0)
        self.assertIsNotNone(result.generation_time)
        self.assertEqual(result.model_name, "test-model")
    
    def test_completion_with_stop_tokens(self):
        """Test completion with stop tokens."""
        config = CompletionConfig(
            max_new_tokens=100,
            stop_tokens=[42],  # Stop on token 42
            include_logprobs=True
        )
        
        engine = CompletionEngine(
            model_interface=self.mock_model,
            config=config
        )
        
        # Mock sampling to return stop token
        engine.sampling_interface = MockSamplingInterface()
        
        messages = [{"role": "user", "content": "Test"}]
        result = engine.complete_messages(messages)
        
        # Should stop after first token (42)
        self.assertEqual(len(result.token_ids), 1)
        self.assertEqual(result.token_ids[0], 42)
    
    def test_batch_completion(self):
        """Test batch completion."""
        engine = CompletionEngine(
            model_interface=self.mock_model,
            config=self.config
        )
        
        message_sequences = [
            [{"role": "user", "content": "Question 1"}],
            [{"role": "user", "content": "Question 2"}],
            [{"role": "user", "content": "Question 3"}]
        ]
        
        batch = MessageBatch.from_multiple_messages(message_sequences)
        results = engine.complete_batch(batch)
        
        self.assertIsInstance(results, BatchCompletionResult)
        self.assertEqual(len(results), 3)
        self.assertEqual(results.batch_size, 3)
        self.assertIsNotNone(results.total_generation_time)
        
        for completion in results:
            self.assertIsInstance(completion, CompletionResult)
            self.assertIsNotNone(completion.text)
    
    def test_batch_completion_from_list(self):
        """Test batch completion from list of message sequences."""
        engine = CompletionEngine(
            model_interface=self.mock_model,
            config=self.config
        )
        
        message_sequences = [
            [{"role": "user", "content": "Question 1"}],
            [{"role": "user", "content": "Question 2"}]
        ]
        
        results = engine.complete_batch(message_sequences)
        
        self.assertIsInstance(results, BatchCompletionResult)
        self.assertEqual(len(results), 2)
    
    def test_update_sampling_method(self):
        """Test updating sampling method."""
        engine = CompletionEngine(
            model_interface=self.mock_model,
            config=self.config
        )
        
        engine.update_sampling_method("top_p", p=0.9)
        self.assertEqual(engine._method_name, "top_p")
        self.assertEqual(engine.config.sampling_method, "top_p")
    
    def test_get_model_info(self):
        """Test getting model information."""
        engine = CompletionEngine(
            model_interface=self.mock_model,
            config=self.config
        )
        
        info = engine.get_model_info()
        
        self.assertIn("model_name", info)
        self.assertIn("vocab_size", info)
        self.assertIn("device", info)
        self.assertIn("sampling_method", info)
        
        self.assertEqual(info["model_name"], "test-model")
        self.assertEqual(info["vocab_size"], 1000)
    
    def test_invalid_messages(self):
        """Test handling of invalid messages."""
        engine = CompletionEngine(
            model_interface=self.mock_model,
            config=self.config
        )
        
        # Empty messages
        with self.assertRaises(ValueError):
            engine.complete_messages([])
        
        # Missing required keys
        with self.assertRaises(ValueError):
            engine.complete_messages([{"role": "user"}])  # Missing content
        
        with self.assertRaises(ValueError):
            engine.complete_messages([{"content": "Hello"}])  # Missing role
    
    def test_token_info_creation(self):
        """Test token info creation."""
        engine = CompletionEngine(
            model_interface=self.mock_model,
            config=self.config
        )
        
        messages = [{"role": "user", "content": "Test"}]
        result = engine.complete_messages(messages)
        
        # Should have token info since include_token_info=True
        self.assertIsNotNone(result.token_info)
        self.assertEqual(len(result.token_info), len(result.token_ids))
        
        for token_info in result.token_info:
            self.assertIsNotNone(token_info.token_id)
            self.assertIsNotNone(token_info.token_text)
            self.assertIsNotNone(token_info.logprob)
            self.assertIsNotNone(token_info.prob)
    
    def test_engine_repr(self):
        """Test engine string representation."""
        engine = CompletionEngine(
            model_interface=self.mock_model,
            config=self.config
        )
        
        repr_str = repr(engine)
        self.assertIn("CompletionEngine", repr_str)
        self.assertIn("test-model", repr_str)


if __name__ == "__main__":
    unittest.main()
