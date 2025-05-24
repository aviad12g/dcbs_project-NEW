#!/usr/bin/env python3
"""
Test runner for DCBS tests.

This script runs all the tests for the DCBS evaluation framework.
"""

import argparse
import os
import sys
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_tests(test_type=None, verbose=False):
    """Run tests based on the specified type.

    Args:
        test_type: Type of tests to run (unit, performance, or all)
        verbose: Whether to run tests in verbose mode
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    if test_type == "unit" or test_type is None or test_type == "all":
        print("Running unit tests...")
        # Load all test files except performance tests
        all_tests = loader.discover(".", pattern="test_*.py")
        # Filter out performance tests
        for test_suite in all_tests:
            for test_case in test_suite:
                if "test_performance" not in str(test_case):
                    suite.addTest(test_case)

    if test_type == "performance" or test_type == "all":
        print("Running performance tests...")
        perf_tests = loader.discover(".", pattern="test_performance.py")
        suite.addTests(perf_tests)

    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DCBS tests")
    parser.add_argument(
        "--type",
        choices=["unit", "performance", "all"],
        default="unit",
        help="Type of tests to run",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Run tests in verbose mode"
    )

    args = parser.parse_args()
    sys.exit(run_tests(args.type, args.verbose))
