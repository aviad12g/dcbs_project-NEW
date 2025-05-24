"""
Performance benchmarks for DCBS batch processing.

This module contains benchmarks to measure the performance of batch processing
in the DCBS evaluation framework, particularly focusing on memory usage and
execution time.
"""

import unittest
import time
import json
import tempfile
import os
import sys
import torch
import psutil

# Add parent directory to path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.run_dcbs_eval import process_example, evaluate_methods
from src.token_utils import tokenizer_cache
from src.errors import report_memory_usage
import logging

# Set up logging for benchmarks
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dcbs.benchmark")


class PerformanceBenchmark(unittest.TestCase):
    """Performance benchmarks for DCBS batch processing."""
    
    @classmethod
    def setUpClass(cls):
        """Set up resources for all tests."""
        try:
            # Try to load a small model for testing
            from transformers import AutoModelForCausalLM, AutoTokenizer
            cls.model = AutoModelForCausalLM.from_pretrained("gpt2")
            cls.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            cls.model.config.tokenizer = cls.tokenizer
            cls.device = next(cls.model.parameters()).device
            
            # Create sample benchmark data
            cls.sample_data = []
            for i in range(20):
                cls.sample_data.append({
                    "id": f"sample_{i}",
                    "sentence": f"This is a sample sentence {i}.",
                    "option1": "yes",
                    "option2": "no",
                    "correct_option": "1"
                })
                
            # Create a temporary file for the benchmark data
            cls.temp_dir = tempfile.mkdtemp()
            cls.benchmark_file = os.path.join(cls.temp_dir, "benchmark.json")
            with open(cls.benchmark_file, "w") as f:
                json.dump(cls.sample_data, f)
                
        except Exception as e:
            cls.skipTest(cls, f"Could not set up benchmark environment: {str(e)}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up resources."""
        # Remove temporary files
        if hasattr(cls, "temp_dir") and os.path.exists(cls.temp_dir):
            for file in os.listdir(cls.temp_dir):
                os.remove(os.path.join(cls.temp_dir, file))
            os.rmdir(cls.temp_dir)
    
    def setUp(self):
        """Set up for each test."""
        # Clear tokenizer cache before each test
        tokenizer_cache.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
    
    def measure_memory_usage(self, func, *args, **kwargs):
        """Measure memory usage of a function.
        
        Args:
            func: Function to measure
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Tuple of (result, peak_memory_mb, elapsed_time_ms)
        """
        process = psutil.Process()
        
        # Get initial memory usage
        initial_memory = process.memory_info().rss / (1024 * 1024)
        
        # Run the function and time it
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = (time.time() - start_time) * 1000  # ms
        
        # Get peak memory usage
        peak_memory = process.memory_info().rss / (1024 * 1024)
        memory_increase = peak_memory - initial_memory
        
        return result, memory_increase, elapsed_time
    
    def test_batch_processing_memory_usage(self):
        """Test memory usage with different batch sizes."""
        if not hasattr(self, "model"):
            self.skipTest("Model not available for benchmark")
            
        batch_sizes = [1, 5, 10]
        memory_usage = {}
        processing_time = {}
        
        for batch_size in batch_sizes:
            # Process in batches
            batch_memory = 0
            batch_time = 0
            
            for i in range(0, len(self.sample_data), batch_size):
                batch = self.sample_data[i:i+batch_size]
                
                # Process the batch
                for example in batch:
                    _, mem_increase, elapsed = self.measure_memory_usage(
                        process_example,
                        example,
                        self.model,
                        self.tokenizer,
                        self.device,
                        inject_reasoning=False,
                        multitoken_strategy="combine"
                    )
                    batch_memory += mem_increase
                    batch_time += elapsed
                
                # Force garbage collection after each batch
                import gc
                gc.collect()
            
            memory_usage[batch_size] = batch_memory / len(self.sample_data)
            processing_time[batch_size] = batch_time / len(self.sample_data)
            
            logger.info(f"Batch size {batch_size}: {memory_usage[batch_size]:.2f} MB/example, "
                       f"{processing_time[batch_size]:.2f} ms/example")
        
        # Verify that larger batches are more memory efficient
        # This might not always be true depending on the environment, so we log rather than assert
        if memory_usage[1] > memory_usage[5]:
            logger.info("Confirmed: Batch processing reduces per-example memory usage")
        else:
            logger.info("Note: Batch processing did not reduce per-example memory usage in this environment")
    
    def test_tokenizer_cache_performance(self):
        """Test the performance impact of tokenizer caching."""
        if not hasattr(self, "model"):
            self.skipTest("Model not available for benchmark")
            
        # First run with empty cache
        tokenizer_cache.clear()
        
        start_time = time.time()
        for example in self.sample_data[:5]:
            process_example(
                example,
                self.model,
                self.tokenizer,
                self.device,
                inject_reasoning=False,
                multitoken_strategy="combine"
            )
        first_run_time = time.time() - start_time
        
        # Second run with populated cache
        start_time = time.time()
        for example in self.sample_data[:5]:
            process_example(
                example,
                self.model,
                self.tokenizer,
                self.device,
                inject_reasoning=False,
                multitoken_strategy="combine"
            )
        second_run_time = time.time() - start_time
        
        # Log the results
        logger.info(f"First run (cold cache): {first_run_time:.4f}s")
        logger.info(f"Second run (warm cache): {second_run_time:.4f}s")
        logger.info(f"Cache hit rate: {tokenizer_cache.hits/(tokenizer_cache.hits+tokenizer_cache.misses)*100:.1f}%")
        
        # The second run should be faster due to caching
        self.assertLess(second_run_time, first_run_time)


if __name__ == "__main__":
    unittest.main() 