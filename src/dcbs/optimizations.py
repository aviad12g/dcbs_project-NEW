"""Optimization utilities (legacy batch processor removed)."""

from dataclasses import dataclass

from .memory_efficient import MemoryEfficientDCBS


@dataclass
class OptimizationConfig:
    enable_parallel_processing: bool = True
    max_workers: int = 0
    use_gpu_clustering: bool = False
    use_mixed_precision: bool = False


__all__ = ["OptimizationConfig", "MemoryEfficientDCBS"]
