#!/usr/bin/env python3
"""
Script to verify that caching actually improves performance.

This script runs the cache performance tests to measure whether
embedding and clustering caches provide measurable speedups.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run cache performance tests."""
    print("Running cache performance verification...")
    print("=" * 50)
    
    # Run the cache performance tests
    test_file = Path(__file__).parent / "tests" / "test_cache_performance.py"
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            str(test_file), 
            "-v", "-s",  # -s to show print statements
            "--tb=short"
        ], check=True, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        print("\nCache performance verification completed successfully!")
        print("Caching provides measurable performance improvements.")
        
    except subprocess.CalledProcessError as e:
        print("Cache performance test failed:")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        print("\nThis indicates that caching may not be providing expected benefits.")
        print("Consider reviewing cache implementation or disabling caching.")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: pytest not found. Please install pytest:")
        print("pip install pytest")
        sys.exit(1)

if __name__ == "__main__":
    main() 