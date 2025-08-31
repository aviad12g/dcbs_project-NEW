"""
Test runner for the messages completion module.
"""

import unittest
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_all_tests():
    """Run all tests in the messages completion module."""
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


def run_specific_test(test_module):
    """Run tests from a specific module."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_module)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests for messages completion module")
    parser.add_argument(
        "--module", 
        help="Specific test module to run (e.g., test_completion_engine)",
        default=None
    )
    
    args = parser.parse_args()
    
    if args.module:
        success = run_specific_test(args.module)
    else:
        success = run_all_tests()
    
    sys.exit(0 if success else 1)