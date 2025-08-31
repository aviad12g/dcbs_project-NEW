"""
Tests for sampling interfaces.
"""

import unittest
from unittest.mock import Mock, patch
import torch
from pathlib import Path
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from messages_completion.sampling_interface import (
    GreedySamplingInterface,
    TopPSamplingInterface,
    RandomSamplingInterface,
    create_sampling_interface,
    get_available_methods
)


class TestGreedySamplingInterface(unittest.TestCase):
    """Test greedy sampling interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sampler = GreedySamplingInterface()
    
    def test_method_name(self):
        """Test method name."""
        self.assertEqual(self.sampler.method_name, "greedy")
    
    def test_sample_token(self):
        """Test single token sampling."""
        logits = torch.tensor([1.0, 3.0, 2.0, 0.5])
        
        # Should select token with highest logit (index 1)
        token_id = self.sampler.sample_token(logits)
        self.assertEqual(token_id, 1)
    
    def test_sample_token_with_filter(self):
        """Test token sampling with filter."""
        logits = torch.tensor([1.0, 3.0, 2.0, 0.5])
        filter_tokens = {0, 2}  # Only allow tokens 0 and 2
        
        # Should select token 2 (highest among filtered)
        token_id = self.sampler.sample_token(logits, filter_tokens=filter_tokens)
        self.assertEqual(token_id, 2)
    
    def test_sample_batch(self):
        """Test batch token sampling."""
        logits_batch = torch.tensor([
            [1.0, 3.0, 2.0],
            [0.5, 1.0, 2.5],
            [2.0, 1.0, 0.5]
        ])
        
        token_ids = self.sampler.sample_batch(logits_batch)
        
        # Should select highest logit for each sequence
        expected = [1, 2, 0]
        self.assertEqual(token_ids, expected)
    
    def test_sample_batch_with_filters(self):
        """Test batch sampling with filters."""
        logits_batch = torch.tensor([
            [1.0, 3.0, 2.0],
            [0.5, 1.0, 2.5]
        ])
        
        filter_tokens_batch = [
            {0, 2},  # First sequence: only tokens 0, 2
            {1}      # Second sequence: only token 1
        ]
        
        token_ids = self.sampler.sample_batch(
            logits_batch, 
            filter_tokens_batch=filter_tokens_batch
        )
        
        expected = [2, 1]  # Best from filtered sets
        self.assertEqual(token_ids, expected)


class TestTopPSamplingInterface(unittest.TestCase):
    """Test top-p sampling interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sampler = TopPSamplingInterface(p=0.9, temperature=1.0)
    
    def test_method_name(self):
        """Test method name."""
        self.assertEqual(self.sampler.method_name, "top_p_0.9")
    
    def test_parameters(self):
        """Test parameter retrieval."""
        params = self.sampler.get_parameters()
        self.assertEqual(params["p"], 0.9)
        self.assertEqual(params["temperature"], 1.0)
    
    def test_sample_token(self):
        """Test single token sampling."""
        # Set seed for reproducible results
        torch.manual_seed(42)
        
        logits = torch.tensor([1.0, 3.0, 2.0, 0.1])
        token_id = self.sampler.sample_token(logits)
        
        # Should be a valid token ID
        self.assertIn(token_id, [0, 1, 2, 3])
    
    def test_sample_token_with_filter(self):
        """Test token sampling with filter."""
        torch.manual_seed(42)
        
        logits = torch.tensor([1.0, 3.0, 2.0, 0.1])
        filter_tokens = {1, 2}
        
        token_id = self.sampler.sample_token(logits, filter_tokens=filter_tokens)
        
        # Should only select from filtered tokens
        self.assertIn(token_id, filter_tokens)
    
    def test_sample_batch(self):
        """Test batch token sampling."""
        torch.manual_seed(42)
        
        logits_batch = torch.tensor([
            [1.0, 3.0, 2.0],
            [0.5, 1.0, 2.5]
        ])
        
        token_ids = self.sampler.sample_batch(logits_batch)
        
        self.assertEqual(len(token_ids), 2)
        for token_id in token_ids:
            self.assertIn(token_id, [0, 1, 2])
    
    def test_temperature_effect(self):
        """Test temperature effect on sampling."""
        # High temperature should make sampling more random
        high_temp_sampler = TopPSamplingInterface(p=0.9, temperature=2.0)
        
        # Low temperature should make sampling more deterministic
        low_temp_sampler = TopPSamplingInterface(p=0.9, temperature=0.1)
        
        logits = torch.tensor([1.0, 3.0, 2.0, 0.1])
        
        # With low temperature, should almost always pick highest logit
        torch.manual_seed(42)
        low_temp_results = [low_temp_sampler.sample_token(logits) for _ in range(10)]
        
        # Most results should be token 1 (highest logit)
        self.assertGreater(low_temp_results.count(1), 7)


class TestRandomSamplingInterface(unittest.TestCase):
    """Test random sampling interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sampler = RandomSamplingInterface(temperature=1.0)
    
    def test_method_name(self):
        """Test method name."""
        self.assertEqual(self.sampler.method_name, "random_temp1.0")
    
    def test_parameters(self):
        """Test parameter retrieval."""
        params = self.sampler.get_parameters()
        self.assertEqual(params["temperature"], 1.0)
    
    def test_sample_token(self):
        """Test single token sampling."""
        torch.manual_seed(42)
        
        logits = torch.tensor([1.0, 1.0, 1.0, 1.0])  # Uniform
        token_id = self.sampler.sample_token(logits)
        
        self.assertIn(token_id, [0, 1, 2, 3])
    
    def test_sample_token_with_filter(self):
        """Test token sampling with filter."""
        torch.manual_seed(42)
        
        logits = torch.tensor([1.0, 1.0, 1.0, 1.0])
        filter_tokens = {1, 3}
        
        token_id = self.sampler.sample_token(logits, filter_tokens=filter_tokens)
        
        self.assertIn(token_id, filter_tokens)
    
    def test_sample_batch(self):
        """Test batch token sampling."""
        torch.manual_seed(42)
        
        logits_batch = torch.tensor([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ])
        
        token_ids = self.sampler.sample_batch(logits_batch)
        
        self.assertEqual(len(token_ids), 2)
        for token_id in token_ids:
            self.assertIn(token_id, [0, 1, 2])


class TestDCBSSamplingInterface(unittest.TestCase):
    """Test DCBS sampling interface."""
    
    def test_dcbs_import_error(self):
        """Test DCBS interface when DCBS is not available."""
        with patch('messages_completion.sampling_interface.sys.path'):
            # Mock import error
            with patch.dict('sys.modules', {'src.dcbs': None}):
                from messages_completion.sampling_interface import DCBSSamplingInterface
                
                with self.assertRaises(ImportError):
                    DCBSSamplingInterface()
    
    @patch('messages_completion.sampling_interface.DCBSSamplerFactory')
    @patch('messages_completion.sampling_interface.SamplingContext')
    def test_dcbs_initialization(self, mock_context, mock_factory):
        """Test DCBS interface initialization."""
        from messages_completion.sampling_interface import DCBSSamplingInterface
        
        # Mock the factory
        mock_sampler = Mock()
        mock_factory.create_default.return_value = mock_sampler
        
        dcbs_interface = DCBSSamplingInterface(k=4, top_n=20)
        
        self.assertEqual(dcbs_interface.k, 4)
        self.assertEqual(dcbs_interface.top_n, 20)
        mock_factory.create_default.assert_called_once()
    
    @patch('messages_completion.sampling_interface.DCBSSamplerFactory')
    def test_dcbs_method_name(self, mock_factory):
        """Test DCBS method name."""
        from messages_completion.sampling_interface import DCBSSamplingInterface
        
        mock_factory.create_default.return_value = Mock()
        
        dcbs_interface = DCBSSamplingInterface(k=8, clustering_method="dbscan")
        
        self.assertEqual(dcbs_interface.method_name, "dcbs_dbscan_k8")
    
    @patch('messages_completion.sampling_interface.DCBSSamplerFactory')
    def test_dcbs_parameters(self, mock_factory):
        """Test DCBS parameter retrieval."""
        from messages_completion.sampling_interface import DCBSSamplingInterface
        
        mock_factory.create_default.return_value = Mock()
        
        dcbs_interface = DCBSSamplingInterface(
            k=8, 
            top_n=50, 
            clustering_method="dbscan",
            enable_caching=True
        )
        
        params = dcbs_interface.get_parameters()
        
        self.assertEqual(params["k"], 8)
        self.assertEqual(params["top_n"], 50)
        self.assertEqual(params["clustering_method"], "dbscan")
        self.assertTrue(params["enable_caching"])


class TestSamplingFactory(unittest.TestCase):
    """Test sampling interface factory functions."""
    
    def test_create_greedy_interface(self):
        """Test creating greedy sampling interface."""
        interface = create_sampling_interface("greedy")
        
        self.assertIsInstance(interface, GreedySamplingInterface)
        self.assertEqual(interface.method_name, "greedy")
    
    def test_create_top_p_interface(self):
        """Test creating top-p sampling interface."""
        interface = create_sampling_interface("top_p", p=0.8, temperature=0.9)
        
        self.assertIsInstance(interface, TopPSamplingInterface)
        self.assertEqual(interface.p, 0.8)
        self.assertEqual(interface.temperature, 0.9)
    
    def test_create_nucleus_interface(self):
        """Test creating nucleus sampling interface (alias for top_p)."""
        interface = create_sampling_interface("nucleus", p=0.7)
        
        self.assertIsInstance(interface, TopPSamplingInterface)
        self.assertEqual(interface.p, 0.7)
    
    def test_create_random_interface(self):
        """Test creating random sampling interface."""
        interface = create_sampling_interface("random", temperature=1.5)
        
        self.assertIsInstance(interface, RandomSamplingInterface)
        self.assertEqual(interface.temperature, 1.5)
    
    @patch('messages_completion.sampling_interface.DCBSSamplerFactory')
    def test_create_dcbs_interface(self, mock_factory):
        """Test creating DCBS sampling interface."""
        mock_factory.create_default.return_value = Mock()
        
        interface = create_sampling_interface("dcbs", k=6, top_n=30)
        
        from messages_completion.sampling_interface import DCBSSamplingInterface
        self.assertIsInstance(interface, DCBSSamplingInterface)
    
    def test_create_unknown_interface(self):
        """Test creating unknown sampling interface."""
        with self.assertRaises(ValueError):
            create_sampling_interface("unknown_method")
    
    def test_case_insensitive_creation(self):
        """Test case-insensitive interface creation."""
        interface1 = create_sampling_interface("GREEDY")
        interface2 = create_sampling_interface("Top_P", p=0.9)
        
        self.assertIsInstance(interface1, GreedySamplingInterface)
        self.assertIsInstance(interface2, TopPSamplingInterface)
    
    def test_get_available_methods(self):
        """Test getting available sampling methods."""
        methods = get_available_methods()
        
        expected_methods = ["greedy", "top_p", "dcbs", "random"]
        self.assertEqual(set(methods), set(expected_methods))


if __name__ == "__main__":
    unittest.main()