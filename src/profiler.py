"""
Performance profiling and optimization utilities.

This module provides tools for profiling DCBS evaluation performance,
identifying bottlenecks, and suggesting optimizations.
"""

import cProfile
import functools
import io
import pstats
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

import psutil
import torch

from src.errors import eval_logger as logger


class PerformanceProfiler:
    """Comprehensive performance profiler for DCBS evaluation."""

    def __init__(self, enable_gpu_profiling: bool = True):
        self.enable_gpu_profiling = enable_gpu_profiling and torch.cuda.is_available()
        self.profiles = {}
        self.timing_data = {}
        self.memory_data = {}
        self.gpu_data = {}

    @contextmanager
    def profile_section(self, section_name: str):
        """Context manager for profiling a code section."""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        start_gpu = self._get_gpu_usage() if self.enable_gpu_profiling else None

        # Start CPU profiling
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            yield
        finally:
            profiler.disable()
            
            # Record timing
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            # Record memory
            end_memory = self._get_memory_usage()
            memory_delta = end_memory - start_memory
            
            # Record GPU usage
            end_gpu = self._get_gpu_usage() if self.enable_gpu_profiling else None
            gpu_delta = None
            if start_gpu and end_gpu:
                gpu_delta = {
                    "memory_mb": end_gpu["memory_mb"] - start_gpu["memory_mb"],
                    "utilization": end_gpu["utilization"] - start_gpu["utilization"]
                }

            # Store profiling data
            self.profiles[section_name] = profiler
            self.timing_data[section_name] = duration
            self.memory_data[section_name] = memory_delta
            if gpu_delta:
                self.gpu_data[section_name] = gpu_delta

            logger.info(f"Section '{section_name}' completed in {duration:.3f}s, "
                       f"memory delta: {memory_delta:.1f}MB")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def _get_gpu_usage(self) -> Optional[Dict[str, float]]:
        """Get current GPU usage."""
        if not self.enable_gpu_profiling:
            return None
        
        try:
            memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
            utilization = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0
            
            return {
                "memory_mb": memory_allocated,
                "reserved_mb": memory_reserved,
                "utilization": utilization
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU usage: {e}")
            return None

    def get_profile_stats(self, section_name: str, sort_by: str = "cumulative") -> str:
        """Get formatted profiling statistics for a section."""
        if section_name not in self.profiles:
            return f"No profile data for section '{section_name}'"

        profiler = self.profiles[section_name]
        
        # Capture stats output
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats(sort_by)
        stats.print_stats(20)  # Top 20 functions
        
        return stats_stream.getvalue()

    def get_timing_summary(self) -> Dict[str, float]:
        """Get timing summary for all profiled sections."""
        return self.timing_data.copy()

    def get_memory_summary(self) -> Dict[str, float]:
        """Get memory usage summary for all profiled sections."""
        return self.memory_data.copy()

    def get_bottlenecks(self, threshold_seconds: float = 1.0) -> List[Tuple[str, float]]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        for section, duration in self.timing_data.items():
            if duration >= threshold_seconds:
                bottlenecks.append((section, duration))
        
        return sorted(bottlenecks, key=lambda x: x[1], reverse=True)

    def generate_report(self) -> str:
        """Generate a comprehensive performance report."""
        report = ["DCBS Performance Report", "=" * 50, ""]
        
        # Timing summary
        report.append("Timing Summary:")
        report.append("-" * 20)
        total_time = sum(self.timing_data.values())
        for section, duration in sorted(self.timing_data.items(), key=lambda x: x[1], reverse=True):
            percentage = (duration / total_time * 100) if total_time > 0 else 0
            report.append(f"{section:30} {duration:8.3f}s ({percentage:5.1f}%)")
        report.append(f"{'Total':30} {total_time:8.3f}s")
        report.append("")

        # Memory summary
        report.append("Memory Usage Summary:")
        report.append("-" * 25)
        for section, memory_delta in self.memory_data.items():
            report.append(f"{section:30} {memory_delta:+8.1f}MB")
        report.append("")

        # GPU summary (if available)
        if self.gpu_data:
            report.append("GPU Usage Summary:")
            report.append("-" * 20)
            for section, gpu_delta in self.gpu_data.items():
                report.append(f"{section:30} {gpu_delta['memory_mb']:+8.1f}MB GPU memory")
            report.append("")

        # Bottlenecks
        bottlenecks = self.get_bottlenecks()
        if bottlenecks:
            report.append("Performance Bottlenecks (>1s):")
            report.append("-" * 35)
            for section, duration in bottlenecks:
                report.append(f"{section:30} {duration:8.3f}s")
            report.append("")

        # Optimization suggestions
        report.extend(self._generate_optimization_suggestions())

        return "\n".join(report)

    def _generate_optimization_suggestions(self) -> List[str]:
        """Generate optimization suggestions based on profiling data."""
        suggestions = ["Optimization Suggestions:", "-" * 30]
        
        # Check for memory-intensive operations
        high_memory_sections = [
            (section, delta) for section, delta in self.memory_data.items() 
            if delta > 500  # > 500MB
        ]
        if high_memory_sections:
            suggestions.append("• High memory usage detected:")
            for section, delta in high_memory_sections:
                suggestions.append(f"  - {section}: {delta:.1f}MB")
            suggestions.append("  Consider: batch size reduction, model quantization, or caching optimization")
            suggestions.append("")

        # Check for slow sections
        slow_sections = self.get_bottlenecks(0.5)  # > 0.5s
        if slow_sections:
            suggestions.append("• Performance bottlenecks detected:")
            for section, duration in slow_sections[:3]:  # Top 3
                suggestions.append(f"  - {section}: {duration:.3f}s")
            suggestions.append("  Consider: algorithm optimization, parallel processing, or caching")
            suggestions.append("")

        # GPU-specific suggestions
        if self.gpu_data:
            high_gpu_usage = [
                (section, data) for section, data in self.gpu_data.items()
                if data.get("memory_mb", 0) > 1000  # > 1GB
            ]
            if high_gpu_usage:
                suggestions.append("• High GPU memory usage detected:")
                for section, data in high_gpu_usage:
                    suggestions.append(f"  - {section}: {data['memory_mb']:.1f}MB")
                suggestions.append("  Consider: gradient checkpointing, mixed precision, or model sharding")
                suggestions.append("")

        if len(suggestions) == 2:  # Only headers added
            suggestions.append("• No specific optimization suggestions at this time")
            suggestions.append("• Performance appears to be within normal ranges")

        return suggestions

    def clear_data(self):
        """Clear all profiling data."""
        self.profiles.clear()
        self.timing_data.clear()
        self.memory_data.clear()
        self.gpu_data.clear()


def profile_function(profiler: Optional[PerformanceProfiler] = None, section_name: Optional[str] = None):
    """Decorator for profiling individual functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal profiler, section_name
            
            if profiler is None:
                # Create a temporary profiler if none provided
                profiler = PerformanceProfiler()
            
            if section_name is None:
                section_name = f"{func.__module__}.{func.__name__}"
            
            with profiler.profile_section(section_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


class ModelProfiler:
    """Specialized profiler for model operations."""

    def __init__(self):
        self.forward_times = []
        self.generation_times = []
        self.memory_peaks = []

    @contextmanager
    def profile_forward_pass(self):
        """Profile a model forward pass."""
        start_time = time.perf_counter()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            self.forward_times.append(duration)
            
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated()
                self.memory_peaks.append(peak_memory)
                torch.cuda.reset_peak_memory_stats()

    @contextmanager
    def profile_generation(self):
        """Profile text generation."""
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            self.generation_times.append(duration)

    def get_stats(self) -> Dict[str, Any]:
        """Get profiling statistics."""
        stats = {}
        
        if self.forward_times:
            stats["forward_pass"] = {
                "count": len(self.forward_times),
                "mean_time": sum(self.forward_times) / len(self.forward_times),
                "min_time": min(self.forward_times),
                "max_time": max(self.forward_times),
                "total_time": sum(self.forward_times)
            }
        
        if self.generation_times:
            stats["generation"] = {
                "count": len(self.generation_times),
                "mean_time": sum(self.generation_times) / len(self.generation_times),
                "min_time": min(self.generation_times),
                "max_time": max(self.generation_times),
                "total_time": sum(self.generation_times)
            }
        
        if self.memory_peaks:
            stats["memory"] = {
                "peak_memory_mb": max(self.memory_peaks) / 1024 / 1024,
                "mean_memory_mb": sum(self.memory_peaks) / len(self.memory_peaks) / 1024 / 1024,
                "min_memory_mb": min(self.memory_peaks) / 1024 / 1024
            }
        
        return stats

    def clear(self):
        """Clear all profiling data."""
        self.forward_times.clear()
        self.generation_times.clear()
        self.memory_peaks.clear()


class CacheProfiler:
    """Profiler for caching operations."""

    def __init__(self):
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_times = []
        self.cache_sizes = []

    def record_hit(self, access_time: float):
        """Record a cache hit."""
        self.cache_hits += 1
        self.cache_times.append(access_time)

    def record_miss(self, computation_time: float):
        """Record a cache miss."""
        self.cache_misses += 1
        self.cache_times.append(computation_time)

    def record_size(self, size_bytes: int):
        """Record cache size."""
        self.cache_sizes.append(size_bytes)

    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache profiling statistics."""
        total_accesses = self.cache_hits + self.cache_misses
        
        stats = {
            "hit_rate": self.get_hit_rate(),
            "total_accesses": total_accesses,
            "hits": self.cache_hits,
            "misses": self.cache_misses
        }
        
        if self.cache_times:
            stats["timing"] = {
                "mean_access_time": sum(self.cache_times) / len(self.cache_times),
                "min_access_time": min(self.cache_times),
                "max_access_time": max(self.cache_times)
            }
        
        if self.cache_sizes:
            stats["size"] = {
                "current_size_mb": self.cache_sizes[-1] / 1024 / 1024 if self.cache_sizes else 0,
                "max_size_mb": max(self.cache_sizes) / 1024 / 1024,
                "mean_size_mb": sum(self.cache_sizes) / len(self.cache_sizes) / 1024 / 1024
            }
        
        return stats

    def clear(self):
        """Clear all profiling data."""
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_times.clear()
        self.cache_sizes.clear()


# Global profiler instance
global_profiler = PerformanceProfiler()


def get_global_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    return global_profiler


def profile_evaluation_run(func: Callable) -> Callable:
    """Decorator for profiling entire evaluation runs."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with global_profiler.profile_section("evaluation_run"):
            result = func(*args, **kwargs)
        
        # Generate and log performance report
        report = global_profiler.generate_report()
        logger.info("Performance Report:\n" + report)
        
        return result
    
    return wrapper 