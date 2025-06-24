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
    
    def __init__(self, target_memory_utilization: float = 0.85):
        """
        Initialize GPU optimizer.
        
        Args:
            target_memory_utilization: Target GPU memory utilization (0.0-1.0)
        """
        self.target_memory_utilization = target_memory_utilization
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
                              device_id: int = 0) -> int:
        """
        Determine optimal batch size for the given model and GPU.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            sample_input_length: Sample input length for testing
            device_id: GPU device ID
            
        Returns:
            Optimal batch size
        """
        if device_id in self.optimal_batch_sizes:
            return self.optimal_batch_sizes[device_id]
        
        if device_id >= len(self.available_gpus):
            logger.warning(f"Invalid device_id {device_id}, using device 0")
            device_id = 0
        
        gpu = self.available_gpus[device_id]
        device = gpu.device
        
        # Start with a conservative batch size
        batch_size = 1
        max_batch_size = 1
        
        # Create sample inputs
        sample_text = "This is a sample text for batch size optimization." * 10
        
        try:
            model.eval()
            with torch.no_grad():
                # Binary search for optimal batch size
                low, high = 1, 64
                
                while low <= high:
                    mid = (low + high) // 2
                    
                    try:
                        # Clear cache
                        torch.cuda.empty_cache()
                        
                        # Test batch processing
                        inputs = [sample_text] * mid
                        encoded = tokenizer(
                            inputs, 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True, 
                            max_length=sample_input_length
                        ).to(device)
                        
                        # Forward pass
                        outputs = model(**encoded)
                        
                        # Check memory utilization
                        _, _, utilization = gpu.get_memory_info()
                        
                        if utilization <= self.target_memory_utilization * 100:
                            max_batch_size = mid
                            low = mid + 1
                        else:
                            high = mid - 1
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            high = mid - 1
                        else:
                            raise e
                
                # Use 80% of max found batch size for safety
                optimal_batch_size = max(1, int(max_batch_size * 0.8))
                
        except Exception as e:
            logger.warning(f"Failed to optimize batch size for GPU {device_id}: {e}")
            optimal_batch_size = 1
        
        finally:
            torch.cuda.empty_cache()
        
        self.optimal_batch_sizes[device_id] = optimal_batch_size
        logger.info(f"Optimal batch size for GPU {device_id}: {optimal_batch_size}")
        
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


# Global optimizer instance
_gpu_optimizer = None

def get_gpu_optimizer() -> GPUOptimizer:
    """Get global GPU optimizer instance."""
    global _gpu_optimizer
    if _gpu_optimizer is None:
        _gpu_optimizer = GPUOptimizer()
    return _gpu_optimizer 