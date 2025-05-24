#!/usr/bin/env python3
"""
Unit tests for the --limit parameter in the DCBS evaluation framework.
Tests to ensure that the --limit parameter correctly limits the number of examples processed.
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Add the parent directory to the path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import shutil

import pandas as pd

from src.run_dcbs_eval import main


class TestLimit(unittest.TestCase):
    """Test case for the limit parameter in run_dcbs_eval.py."""

    def setUp(self):
        """
        Set up temporary files for testing.
        """
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()

        # Set up paths for test files
        self.output_csv = os.path.join(self.temp_dir, "test_output.csv")

    def tearDown(self):
        """
        Clean up temporary files after testing.
        """
        # Clean up temporary directory and files
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @unittest.skip("Skipping test_limit_parameter due to configuration differences")
    def test_limit_parameter(self):
        """Test that the --limit parameter correctly limits the number of examples."""
        # Skip test if it would take too long in CI
        if os.environ.get("CI") == "true":
            self.skipTest("Skipping limit test in CI environment")

        # Run with a small limit to test functionality
        limit = 3

        # Use subprocess to run the evaluation with the limit parameter
        try:
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "src.run_dcbs_eval",
                    "--config",
                    "configs/study_config.yaml",
                    "--limit",
                    str(limit),
                    "--out_csv",
                    self.output_csv,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            # Check if the output CSV was created
            self.assertTrue(
                os.path.exists(self.output_csv),
                f"Output CSV not created: {self.output_csv}",
            )

            # Load the CSV and check the number of examples
            df = pd.read_csv(self.output_csv)

            # There should be 4 rows per example (4 methods per example)
            expected_rows = limit * 4
            self.assertEqual(
                len(df),
                expected_rows,
                f"Expected {expected_rows} rows (4 methods × {limit} examples), got {len(df)}",
            )

            # Check that all four methods were used for each example
            methods = df["method"].unique()
            self.assertEqual(len(methods), 4, f"Expected 4 methods, got {len(methods)}")
            self.assertTrue(
                all(
                    method in methods
                    for method in ["greedy", "top-p", "dcbs", "random"]
                ),
                "Missing one or more expected methods",
            )

        except subprocess.CalledProcessError as e:
            self.fail(f"Subprocess failed: {e.stderr}")

    @unittest.skip(
        "Skipping test_script_wrapper_limit due to configuration differences"
    )
    def test_script_wrapper_limit(self):
        """Test that the top-level wrapper script properly passes the limit parameter."""
        # Skip test if it would take too long in CI
        if os.environ.get("CI") == "true":
            self.skipTest("Skipping limit test in CI environment")

        # Run with a small limit to test functionality
        limit = 2

        # Use subprocess to run the evaluation with the limit parameter through the wrapper
        try:
            result = subprocess.run(
                [
                    "python",
                    "run_dcbs_eval.py",
                    "--config",
                    "configs/study_config.yaml",
                    "--limit",
                    str(limit),
                    "--out_csv",
                    self.output_csv,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            # Check if the output CSV was created
            self.assertTrue(
                os.path.exists(self.output_csv),
                f"Output CSV not created: {self.output_csv}",
            )

            # Load the CSV and check the number of examples
            df = pd.read_csv(self.output_csv)

            # There should be 4 rows per example (4 methods per example)
            expected_rows = limit * 4
            self.assertEqual(
                len(df),
                expected_rows,
                f"Expected {expected_rows} rows (4 methods × {limit} examples), got {len(df)}",
            )

        except subprocess.CalledProcessError as e:
            self.fail(f"Subprocess failed: {e.stderr}")


if __name__ == "__main__":
    unittest.main()
