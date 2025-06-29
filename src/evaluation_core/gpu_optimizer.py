"""
GPU optimization utilities for dynamic batch sizing and multi-GPU support.

This module provides functionality to optimize GPU utilization by:
1. Automatically determining optimal batch sizes based on GPU memory
2. Supporting multi-GPU processing
3. Memory monitoring and adaptive batch adjustment
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import List, Tuple, Optional, Dict, Any
import gc
import psutil
import time
import os
from pathlib import Path

from src.errors import eval_logger as logger


class GPUInfo:
    """Information about a GPU device."""
    
    def __init__(self, device_id: int):
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
        self.properties = torch.cuda.get_device_properties(device_id)
        self.name = self.properties.name
        self.total_memory = self.properties.total_memory
        self.compute_capability = f"{self.properties.major}.{self.properties.minor}"
        
    def get_memory_info(self) -> Tuple[int, int, float]:
        """Get current memory usage info."""
        allocated = torch.cuda.memory_allocated(self.device_id)
        cached = torch.cuda.memory_reserved(self.device_id)
        utilization = (allocated / self.total_memory) * 100
        return allocated, cached, utilization
    
    def __str__(self) -> str:
        allocated, cached, util = self.get_memory_info()
        return (f"GPU {self.device_id}: {self.name} "
                f"({self.total_memory / 1e9:.1f}GB total, {util:.1f}% used)")


class GPUOptimizer:
    """Optimizes GPU usage for evaluation workloads."""
    
    def __init__(self, target_memory_utilization: float = 0.95, safety_margin: float = 0.90):
        """
        Initialize GPU optimizer.
        
        Args:
            target_memory_utilization: Target GPU memory utilization (0.0-1.0)
            safety_margin: Safety margin for batch size selection (0.0-1.0)
        """
        self.target_memory_utilization = target_memory_utilization
        self.safety_margin = safety_margin
        self.available_gpus = self._detect_gpus()
        self.optimal_batch_sizes = {}
        
        if not self.available_gpus:
            logger.warning("No CUDA GPUs detected, falling back to CPU")
        else:
            logger.info(f"Detected {len(self.available_gpus)} GPU(s):")
            for gpu in self.available_gpus:
                logger.info(f"  {gpu}")
    
    def _detect_gpus(self) -> List[GPUInfo]:
        """Detect available GPUs."""
        if not torch.cuda.is_available():
            return []
        
        gpus = []
        for i in range(torch.cuda.device_count()):
            try:
                gpu_info = GPUInfo(i)
                gpus.append(gpu_info)
            except Exception as e:
                logger.warning(f"Failed to initialize GPU {i}: {e}")
        
        return gpus
    
    def get_optimal_batch_size(self, 
                              model, 
                              tokenizer, 
                              sample_input_length: int = 512,
                              device_id: int = 0,
                              min_batch_size: int = 1,
                              max_batch_size: int = 512) -> int:
        """
        Determine optimal batch size for the given model and GPU.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            sample_input_length: Sample input length for testing
            device_id: GPU device ID
            min_batch_size: Minimum batch size to test
            max_batch_size: Maximum batch size to test
            
        Returns:
            Optimal batch size
        """
        cache_key = f"{device_id}_{sample_input_length}_{min_batch_size}_{max_batch_size}"
        if cache_key in self.optimal_batch_sizes:
            return self.optimal_batch_sizes[cache_key]
        
        if not self.available_gpus or device_id >= len(self.available_gpus):
            logger.warning(f"Invalid device_id {device_id} or no GPUs available, using batch size 1")
            return 1
        
        gpu = self.available_gpus[device_id]
        device = gpu.device
        
        # Clear cache before optimization
        torch.cuda.empty_cache()
        
        # Record initial memory state
        initial_allocated, _, initial_util = gpu.get_memory_info()
        logger.info(f"Starting batch size optimization on {gpu.name}")
        logger.info(f"Initial memory usage: {initial_util:.1f}%")
        
        # Create sample inputs - more realistic test data
        sample_texts = [
            "This is a sample question about mathematics and science that requires reasoning to solve correctly.",
            "Here is another example of a complex problem that involves multiple steps and logical thinking.",
            "Consider this scenario where we need to analyze data and draw conclusions based on evidence.",
            "This question tests understanding of concepts and the ability to apply knowledge effectively."
        ]
        
        optimal_batch_size = min_batch_size
        
        try:
            model.eval()
            with torch.no_grad():
                # Binary search for optimal batch size
                low, high = min_batch_size, max_batch_size
                last_successful_batch_size = min_batch_size
                
                while low <= high:
                    mid = (low + high) // 2
                    
                    try:
                        # Clear cache before each test
                        torch.cuda.empty_cache()
                        
                        # Create batch of inputs
                        batch_texts = []
                        for i in range(mid):
                            batch_texts.append(sample_texts[i % len(sample_texts)])
                        
                        # Tokenize batch
                        encoded = tokenizer(
                            batch_texts, 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True, 
                            max_length=sample_input_length
                        ).to(device)
                        
                        # Test forward pass
                        start_time = time.time()
                        outputs = model(**encoded)
                        
                        # Force computation to complete
                        if hasattr(outputs, 'logits'):
                            _ = outputs.logits.sum()
                        torch.cuda.synchronize()
                        
                        end_time = time.time()
                        
                        # Check memory utilization after forward pass
                        allocated, _, utilization = gpu.get_memory_info()
                        
                        # Calculate throughput
                        throughput = mid / (end_time - start_time)
                        
                        logger.debug(f"Batch size {mid}: {utilization:.1f}% memory, {throughput:.1f} samples/sec")
                        
                        # For high target utilization (>90%), be more aggressive
                        utilization_threshold = self.target_memory_utilization * 100
                        if self.target_memory_utilization >= 0.90:
                            # Allow slightly higher utilization for aggressive optimization
                            utilization_threshold = min(98.0, utilization_threshold + 5.0)
                        
                        if utilization <= utilization_threshold:
                            last_successful_batch_size = mid
                            low = mid + 1
                            logger.debug(f"  ✓ Batch size {mid} acceptable ({utilization:.1f}% ≤ {utilization_threshold:.1f}%)")
                        else:
                            high = mid - 1
                            logger.debug(f"  ✗ Batch size {mid} too high ({utilization:.1f}% > {utilization_threshold:.1f}%)")
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            logger.debug(f"Batch size {mid}: OOM error")
                            high = mid - 1
                        else:
                            logger.warning(f"Unexpected error at batch size {mid}: {e}")
                            high = mid - 1
                    except Exception as e:
                        logger.warning(f"Error testing batch size {mid}: {e}")
                        high = mid - 1
                
                # Apply safety margin to the last successful batch size
                optimal_batch_size = max(min_batch_size, int(last_successful_batch_size * self.safety_margin))
                
        except Exception as e:
            logger.warning(f"Failed to optimize batch size for GPU {device_id}: {e}")
            optimal_batch_size = min_batch_size
        
        finally:
            # Clean up
            torch.cuda.empty_cache()
            gc.collect()
        
        # Final memory check
        final_allocated, _, final_util = gpu.get_memory_info()
        
        self.optimal_batch_sizes[cache_key] = optimal_batch_size
        logger.info(f"Optimal batch size for GPU {device_id}: {optimal_batch_size}")
        logger.info(f"Expected memory utilization: ~{self.target_memory_utilization * 100:.1f}%")
        logger.info(f"Final memory usage: {final_util:.1f}%")
        
        return optimal_batch_size
    
    def get_best_device(self) -> torch.device:
        """Get the best available device (GPU with most free memory or CPU)."""
        if not self.available_gpus:
            return torch.device("cpu")
        
        best_gpu = None
        max_free_memory = 0
        
        for gpu in self.available_gpus:
            allocated, _, _ = gpu.get_memory_info()
            free_memory = gpu.total_memory - allocated
            
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_gpu = gpu
        
        return best_gpu.device if best_gpu else torch.device("cpu")
    
    def monitor_gpu_usage(self) -> Dict[int, Dict[str, float]]:
        """Monitor current GPU usage across all devices."""
        usage_info = {}
        
        for gpu in self.available_gpus:
            allocated, cached, utilization = gpu.get_memory_info()
            usage_info[gpu.device_id] = {
                "allocated_gb": allocated / 1e9,
                "cached_gb": cached / 1e9,
                "total_gb": gpu.total_memory / 1e9,
                "utilization_percent": utilization,
                "free_gb": (gpu.total_memory - allocated) / 1e9
            }
        
        return usage_info

    def adjust_batch_size_if_needed(self, current_batch_size: int, utilization_history: List[float], 
                                   model, tokenizer, device_id: int = 0) -> int:
        """
        Dynamically adjust batch size if GPU utilization is consistently low.
        
        Args:
            current_batch_size: Current batch size being used
            utilization_history: Recent GPU utilization percentages
            model: The language model
            tokenizer: The tokenizer
            device_id: GPU device ID
            
        Returns:
            New optimal batch size (may be same as current)
        """
        if not self.available_gpus or not utilization_history:
            return current_batch_size
            
        # Check if utilization is consistently low
        avg_utilization = sum(utilization_history) / len(utilization_history)
        target_utilization = self.target_memory_utilization * 100
        
        # If utilization is significantly below target, try to increase batch size
        if avg_utilization < target_utilization * 0.7:  # Less than 70% of target
            logger.info(f"Low GPU utilization detected ({avg_utilization:.1f}% vs target {target_utilization:.1f}%)")
            logger.info("Attempting to increase batch size for better utilization...")
            
            # Try a larger batch size range starting from current
            new_batch_size = self.get_optimal_batch_size(
                model, tokenizer, 
                device_id=device_id,
                min_batch_size=current_batch_size,
                max_batch_size=min(1024, current_batch_size * 4)  # Cap at 4x current or 1024
            )
            
            if new_batch_size > current_batch_size:
                logger.info(f"Increased batch size from {current_batch_size} to {new_batch_size}")
                return new_batch_size
            
        return current_batch_size


# Global optimizer instance
_gpu_optimizer = None

def get_gpu_optimizer() -> GPUOptimizer:
    """Get global GPU optimizer instance."""
    global _gpu_optimizer
    if _gpu_optimizer is None:
        _gpu_optimizer = GPUOptimizer()
    return _gpu_optimizer 