"""
Unit tests for sampler implementations.

Tests all sampler classes including GreedySampler, TopPSampler,
RandomSampler, and DCBSSampler.
"""

import unittest
from unittest.mock import Mock, patch

import torch

from dcbs.samplers import (
    GreedySampler,
    TopPSampler,
    RandomSampler,
    DCBSSampler,
    SamplingContext,
    Sampler,
)


class TestSamplingContext(unittest.TestCase):
    """Test SamplingContext functionality."""

    def test_sampling_context_creation(self):
        """Test creating a SamplingContext."""
        embedding = Mock()
        tokenizer = Mock()
        device = torch.device("cpu")
        
        context = SamplingContext(
            embedding_layer=embedding,
            tokenizer=tokenizer,
            device=device
        )
        
        self.assertEqual(context.embedding_layer, embedding)
        self.assertEqual(context.tokenizer, tokenizer)
        self.assertEqual(context.device, device)

    def test_sampling_context_defaults(self):
        """Test SamplingContext with default values."""
        context = SamplingContext()
        self.assertIsNone(context.embedding_layer)
        self.assertIsNone(context.tokenizer)
        self.assertIsNone(context.device)


class TestGreedySampler(unittest.TestCase):
    """Test GreedySampler functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.sampler = GreedySampler()

    def test_greedy_sampling_basic(self):
        """Test basic greedy sampling."""
        logits = torch.tensor([1.0, 3.0, 2.0, 0.5])
        result = self.sampler.sample(logits)
        self.assertEqual(result, 1)  # Index of maximum value

    def test_greedy_sampling_with_filter(self):
        """Test greedy sampling with token filtering."""
        logits = torch.tensor([1.0, 3.0, 2.0, 0.5])
        filter_tokens = {1}  # Exclude the maximum token
        result = self.sampler.sample(logits, filter_tokens=filter_tokens)
        self.assertEqual(result, 2)  # Index of second maximum

    def test_greedy_sampling_deterministic(self):
        """Test that greedy sampling is deterministic."""
        logits = torch.tensor([1.0, 3.0, 2.0, 0.5])
        result1 = self.sampler.sample(logits)
        result2 = self.sampler.sample(logits)
        self.assertEqual(result1, result2)

    def test_greedy_sampling_empty_filter(self):
        """Test greedy sampling with empty filter set."""
        logits = torch.tensor([1.0, 3.0, 2.0, 0.5])
        result = self.sampler.sample(logits, filter_tokens=set())
        self.assertEqual(result, 1)


class TestTopPSampler(unittest.TestCase):
    """Test TopPSampler functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.sampler = TopPSampler(p=0.9)

    def test_top_p_initialization(self):
        """Test TopPSampler initialization."""
        sampler = TopPSampler(p=0.8)
        self.assertEqual(sampler.p, 0.8)

    def test_top_p_default_initialization(self):
        """Test TopPSampler default initialization."""
        sampler = TopPSampler()
        self.assertEqual(sampler.p, 0.9)

    @patch('torch.multinomial')
    def test_top_p_sampling_basic(self, mock_multinomial):
        """Test basic top-p sampling."""
        mock_multinomial.return_value = torch.tensor([1])
        
        logits = torch.tensor([1.0, 3.0, 2.0, 0.5])
        result = self.sampler.sample(logits)
        
        # Verify multinomial was called
        mock_multinomial.assert_called_once()
        self.assertIsInstance(result, int)

    def test_top_p_sampling_with_filter(self):
        """Test top-p sampling with token filtering."""
        logits = torch.tensor([1.0, 3.0, 2.0, 0.5])
        filter_tokens = {0, 1}
        
        with patch('torch.multinomial', return_value=torch.tensor([0])):
            result = self.sampler.sample(logits, filter_tokens=filter_tokens)
            self.assertIsInstance(result, int)
            self.assertNotIn(result, filter_tokens)

    def test_top_p_nucleus_selection(self):
        """Test that top-p correctly selects nucleus tokens."""
        # Create logits where only top tokens sum to > p
        logits = torch.tensor([10.0, 9.0, 1.0, 0.1])  # High contrast
        
        with patch('torch.multinomial', return_value=torch.tensor([0])):
            result = self.sampler.sample(logits)
            self.assertIn(result, [0, 1])  # Should be one of the top tokens


class TestRandomSampler(unittest.TestCase):
    """Test RandomSampler functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.sampler = RandomSampler()

    @patch('random.choice')
    def test_random_sampling_basic(self, mock_choice):
        """Test basic random sampling."""
        mock_choice.return_value = 2
        
        logits = torch.tensor([1.0, 3.0, 2.0, 0.5])
        result = self.sampler.sample(logits)
        
        mock_choice.assert_called_once()
        self.assertEqual(result, 2)

    @patch('random.choice')
    def test_random_sampling_with_filter(self, mock_choice):
        """Test random sampling with token filtering."""
        mock_choice.return_value = 2
        
        logits = torch.tensor([1.0, 3.0, 2.0, 0.5])
        filter_tokens = {0, 1}
        result = self.sampler.sample(logits, filter_tokens=filter_tokens)
        
        # Verify choice was called with filtered tokens
        called_args = mock_choice.call_args[0][0]
        self.assertNotIn(0, called_args)
        self.assertNotIn(1, called_args)
        self.assertEqual(result, 2)

    def test_random_sampling_all_tokens_filtered(self):
        """Test random sampling when all tokens are filtered."""
        logits = torch.tensor([1.0, 3.0, 2.0, 0.5])
        filter_tokens = {0, 1, 2, 3}  # Filter all tokens
        
        with patch('random.choice', return_value=0):
            result = self.sampler.sample(logits, filter_tokens=filter_tokens)
            self.assertEqual(result, 0)  # Should fall back to first token


class TestDCBSSampler(unittest.TestCase):
    """Test DCBSSampler functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.sampler = DCBSSampler(k=2, top_n=4)
        
        # Create mock embedding layer
        self.mock_embedding = Mock()
        self.mock_embedding.weight = torch.randn(10, 8)  # 10 tokens, 8 dimensions
        
        # Create mock context
        self.context = SamplingContext(
            embedding_layer=self.mock_embedding,
            tokenizer=Mock(),
            device=torch.device("cpu")
        )

    def test_dcbs_initialization(self):
        """Test DCBSSampler initialization."""
        sampler = DCBSSampler(k=3, top_n=10)
        self.assertEqual(sampler.k, 3)
        self.assertEqual(sampler.top_n, 10)

    def test_dcbs_create_default(self):
        """Test DCBSSampler.create_default factory method."""
        sampler = DCBSSampler.create_default(k=4, top_n=20)
        self.assertEqual(sampler.k, 4)
        self.assertEqual(sampler.top_n, 20)

    @patch('dcbs.clustering.KMeansClusterer')
    @patch('dcbs.clustering.TopNCandidateSelector')
    def test_dcbs_sampling_basic(self, mock_selector, mock_clusterer):
        """Test basic DCBS sampling."""
        # Mock the candidate selector
        mock_selector_instance = Mock()
        mock_selector_instance.select_candidates.return_value = [0, 1, 2, 3]
        mock_selector.return_value = mock_selector_instance
        
        # Mock the clusterer
        mock_clusterer_instance = Mock()
        mock_clusterer_instance.cluster.return_value = {
            0: [0, 1],
            1: [2, 3]
        }
        mock_clusterer.return_value = mock_clusterer_instance
        
        logits = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        result = self.sampler.sample(logits, self.context)
        
        self.assertIsInstance(result, int)
        self.assertIn(result, [0, 1, 2, 3])  # Should be one of the candidates

    def test_dcbs_sampling_requires_context(self):
        """Test that DCBS sampling requires a context."""
        logits = torch.tensor([1.0, 2.0, 3.0])
        
        with self.assertRaises(ValueError):
            self.sampler.sample(logits, None)

    def test_dcbs_sampling_with_filter(self):
        """Test DCBS sampling with token filtering."""
        with patch('dcbs.clustering.TopNCandidateSelector') as mock_selector:
            with patch('dcbs.clustering.KMeansClusterer') as mock_clusterer:
                # Mock the candidate selector to respect filter
                mock_selector_instance = Mock()
                mock_selector_instance.select_candidates.return_value = [2, 3]  # Filtered candidates
                mock_selector.return_value = mock_selector_instance
                
                # Mock the clusterer
                mock_clusterer_instance = Mock()
                mock_clusterer_instance.cluster.return_value = {0: [2, 3]}
                mock_clusterer.return_value = mock_clusterer_instance
                
                logits = torch.tensor([3.0, 2.0, 1.0, 0.5])
                filter_tokens = {0, 1}
                
                result = self.sampler.sample(logits, self.context, filter_tokens=filter_tokens)
                
                self.assertIsInstance(result, int)
                self.assertNotIn(result, filter_tokens)


class TestSamplerInterface(unittest.TestCase):
    """Test the abstract Sampler interface."""

    def test_sampler_is_abstract(self):
        """Test that Sampler cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            Sampler()

    def test_all_samplers_implement_interface(self):
        """Test that all sampler classes implement the Sampler interface."""
        samplers = [GreedySampler(), TopPSampler(), RandomSampler()]
        
        for sampler in samplers:
            self.assertIsInstance(sampler, Sampler)
            self.assertTrue(hasattr(sampler, 'sample'))
            self.assertTrue(callable(getattr(sampler, 'sample')))


if __name__ == "__main__":
    unittest.main()
