"""
Test to ensure that the Winogrande dataset has 1,267 examples.
"""

import json
import os
import sys
import unittest
from pathlib import Path

# Add the parent directory to the path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prepare_winogrande import prepare


class TestWinogrande(unittest.TestCase):
    """Test case for Winogrande dataset verification."""

    def setUp(self):
        """Create a temporary file for the test."""
        self.temp_file = os.path.join(os.path.dirname(__file__), "test_wino.json")

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)

    def test_winogrande_size(self):
        """Test that the Winogrande dataset has 1,267 examples."""
        # Only run this test if we can access the dataset
        try:
            from datasets import load_dataset

            load_dataset("winogrande", "winogrande_xl", split="validation")
        except Exception:
            self.skipTest("Couldn't access Winogrande dataset")

        # Prepare the dataset with a small sample for testing
        examples = prepare(num_examples=10, output_file=self.temp_file)

        # Check the structure of the examples
        self.assertEqual(len(examples), 10, "Should have 10 examples")
        for example in examples:
            self.assertIn("sentence", example)
            self.assertIn("option1", example)
            self.assertIn("option2", example)
            self.assertIn("correct_option", example)

        # Check if output file was created
        self.assertTrue(os.path.exists(self.temp_file), "Output file not created")

        # Load the file to verify content
        with open(self.temp_file, "r") as f:
            loaded_examples = json.load(f)

        self.assertEqual(len(loaded_examples), 10, "Should have 10 examples in file")

        # Now check full dataset size without actually downloading all examples
        from datasets import load_dataset

        dataset = load_dataset("winogrande", "winogrande_xl", split="validation")
        self.assertEqual(
            len(dataset), 1267, "Should have 1267 examples in full dataset"
        )


if __name__ == "__main__":
    unittest.main()
