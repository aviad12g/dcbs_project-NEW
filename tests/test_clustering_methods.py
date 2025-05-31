"""
Tests for different clustering methods in DCBS.

This module tests that DCBS works correctly with KMeans, DBSCAN, 
and Hierarchical clustering methods.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

# Add the parent directory to the path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dcbs import (
    DCBSSampler,
    SamplingContext,
    KMeansClusterer,
    DBSCANClusterer,
    HierarchicalClusterer,
    TopNCandidateSelector,
    CategorySampler,
    GreedyCategorySelector,
    GreedyTokenSelector,
)


class TestClusteringMethods(unittest.TestCase):
    """Test cases for different clustering methods in DCBS."""

    def setUp(self):
        """Set up the test environment before each test."""
        # Create a mock embedding layer
        self.vocab_size = 100
        self.embed_dim = 50
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embed_dim)
        
        # Initialize with deterministic values
        torch.manual_seed(42)
        self.embedding.weight.data = torch.randn(self.vocab_size, self.embed_dim)
        
        # Create sampling context
        self.context = SamplingContext(
            embedding_layer=self.embedding,
            tokenizer=None,
            device=torch.device("cpu"),
        )
        
        # Create common components
        self.candidate_selector = TopNCandidateSelector(top_n=20)
        self.category_sampler = CategorySampler(
            category_selector=GreedyCategorySelector(),
            token_selector=GreedyTokenSelector()
        )
        
        # Create test logits with clear peaks
        self.logits = torch.ones(self.vocab_size) * -10  # Low baseline
        # Create some high probability tokens
        self.logits[10:15] = torch.tensor([5.0, 4.5, 4.0, 3.5, 3.0])
        self.logits[30:35] = torch.tensor([3.0, 2.5, 2.0, 1.5, 1.0])
        self.logits[50:55] = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0])

    def test_kmeans_clustering(self):
        """Test DCBS with KMeans clustering."""
        clusterer = KMeansClusterer(k=3)
        sampler = DCBSSampler(
            clusterer=clusterer,
            candidate_selector=self.candidate_selector,
            category_sampler=self.category_sampler,
            context=self.context,
            enable_caching=False
        )
        
        # Sample multiple times to ensure consistency
        samples = []
        for _ in range(5):
            token_id = sampler.sample(self.logits)
            samples.append(token_id)
        
        # All samples should be the same (deterministic)
        self.assertEqual(len(set(samples)), 1, "KMeans clustering should be deterministic")
        
        # Should select from high probability tokens
        self.assertIn(samples[0], range(self.vocab_size))
        
    def test_dbscan_clustering(self):
        """Test DCBS with DBSCAN clustering."""
        # Test with different epsilon values
        for eps in [0.1, 0.3, 0.5]:
            with self.subTest(eps=eps):
                clusterer = DBSCANClusterer(eps=eps, min_samples=2)
                sampler = DCBSSampler(
                    clusterer=clusterer,
                    candidate_selector=self.candidate_selector,
                    category_sampler=self.category_sampler,
                    context=self.context,
                    enable_caching=False
                )
                
                # Sample with DBSCAN
                token_id = sampler.sample(self.logits)
                
                # Verify it's a valid token
                self.assertGreaterEqual(token_id, 0)
                self.assertLess(token_id, self.vocab_size)
                
                # Verify consistency
                token_id2 = sampler.sample(self.logits)
                self.assertEqual(token_id, token_id2, f"DBSCAN with eps={eps} should be deterministic")
    
    def test_hierarchical_clustering(self):
        """Test DCBS with Hierarchical clustering."""
        # Test with different linkage methods
        for linkage in ["average", "complete", "single"]:
            with self.subTest(linkage=linkage):
                clusterer = HierarchicalClusterer(k=3, linkage=linkage)
                sampler = DCBSSampler(
                    clusterer=clusterer,
                    candidate_selector=self.candidate_selector,
                    category_sampler=self.category_sampler,
                    context=self.context,
                    enable_caching=False
                )
                
                # Sample with hierarchical clustering
                token_id = sampler.sample(self.logits)
                
                # Verify it's a valid token
                self.assertGreaterEqual(token_id, 0)
                self.assertLess(token_id, self.vocab_size)
                
                # Verify consistency
                token_id2 = sampler.sample(self.logits)
                self.assertEqual(token_id, token_id2, f"Hierarchical clustering with {linkage} linkage should be deterministic")
    
    def test_ward_linkage_hierarchical(self):
        """Test Hierarchical clustering with Ward linkage (requires euclidean metric)."""
        clusterer = HierarchicalClusterer(k=3, linkage="ward")
        sampler = DCBSSampler(
            clusterer=clusterer,
            candidate_selector=self.candidate_selector,
            category_sampler=self.category_sampler,
            context=self.context,
            enable_caching=False
        )
        
        # Should not raise an error despite metric mismatch (handled internally)
        token_id = sampler.sample(self.logits)
        self.assertGreaterEqual(token_id, 0)
        self.assertLess(token_id, self.vocab_size)
    
    def test_clustering_with_filtered_tokens(self):
        """Test different clustering methods with filtered tokens."""
        # Only allow tokens 10-15 and 30-35
        filter_tokens = set(list(range(10, 16)) + list(range(30, 36)))
        
        methods = [
            ("kmeans", KMeansClusterer(k=2)),
            ("dbscan", DBSCANClusterer(eps=0.3, min_samples=2)),
            ("hierarchical", HierarchicalClusterer(k=2, linkage="average"))
        ]
        
        for method_name, clusterer in methods:
            with self.subTest(method=method_name):
                sampler = DCBSSampler(
                    clusterer=clusterer,
                    candidate_selector=self.candidate_selector,
                    category_sampler=self.category_sampler,
                    context=self.context,
                    enable_caching=False
                )
                
                token_id = sampler.sample(self.logits, filter_tokens=filter_tokens)
                
                # Should only select from filtered tokens
                self.assertIn(token_id, filter_tokens, 
                            f"{method_name} should respect filter_tokens")
    
    def test_clustering_consistency_across_methods(self):
        """Test that all methods produce consistent results when called multiple times."""
        methods = [
            ("kmeans", KMeansClusterer(k=3)),
            ("dbscan", DBSCANClusterer(eps=0.3, min_samples=2)),
            ("hierarchical", HierarchicalClusterer(k=3, linkage="average"))
        ]
        
        for method_name, clusterer in methods:
            with self.subTest(method=method_name):
                sampler = DCBSSampler(
                    clusterer=clusterer,
                    candidate_selector=self.candidate_selector,
                    category_sampler=self.category_sampler,
                    context=self.context,
                    enable_caching=False
                )
                
                # Sample 10 times
                samples = [sampler.sample(self.logits) for _ in range(10)]
                
                # All samples should be identical (deterministic)
                unique_samples = set(samples)
                self.assertEqual(len(unique_samples), 1, 
                               f"{method_name} should produce deterministic results")
    
    def test_dbscan_dynamic_clusters(self):
        """Test that DBSCAN correctly tracks dynamic number of clusters."""
        clusterer = DBSCANClusterer(eps=0.3, min_samples=2)
        
        # Initial number of clusters should be 1
        self.assertEqual(clusterer.num_clusters, 1)
        
        # Create embeddings and cluster them
        embeddings = torch.randn(20, self.embed_dim)
        labels = clusterer.cluster(embeddings)
        
        # After clustering, num_clusters should reflect actual clusters found
        actual_clusters = len(np.unique(labels))
        self.assertEqual(clusterer.num_clusters, actual_clusters)
        self.assertGreater(actual_clusters, 0)
    
    def test_edge_case_single_candidate(self):
        """Test all clustering methods with a single candidate token."""
        # Create logits where only one token has reasonable probability
        edge_logits = torch.ones(self.vocab_size) * -100
        edge_logits[42] = 10.0
        
        methods = [
            ("kmeans", KMeansClusterer(k=3)),
            ("dbscan", DBSCANClusterer(eps=0.3, min_samples=2)),
            ("hierarchical", HierarchicalClusterer(k=3, linkage="average"))
        ]
        
        for method_name, clusterer in methods:
            with self.subTest(method=method_name):
                sampler = DCBSSampler(
                    clusterer=clusterer,
                    candidate_selector=TopNCandidateSelector(top_n=1),  # Only select top 1
                    category_sampler=self.category_sampler,
                    context=self.context,
                    enable_caching=False
                )
                
                token_id = sampler.sample(edge_logits)
                
                # Should select the only high probability token
                self.assertEqual(token_id, 42, 
                               f"{method_name} should handle single candidate correctly")


if __name__ == "__main__":
    unittest.main() 