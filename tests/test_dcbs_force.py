#!/usr/bin/env python3
"""
Unit tests for the force-include functionality in the DCBS implementation.
Tests to ensure that DCBS properly includes filtered tokens even when they have
extremely low probabilities.
"""

import unittest
import torch
import numpy as np
import sys
import os
import random
from pathlib import Path

# Set fixed seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Add the parent directory to the path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dcbs import category_sample, _top_n_with_force_include


class TestDCBSForceInclude(unittest.TestCase):
    """Test case for DCBS force-include functionality."""

    def test_top_n_with_force_include(self):
        """Test that _top_n_with_force_include properly includes forced tokens."""
        # Create logits with very low probability for id=3 (answer), high for others
        logits = torch.tensor([0.9, 0.8, 0.7, 1e-8, 0.6])
        force_ids = {3}  # The answer token has very low probability

        # Get top-3 tokens with forced include of id=3
        result = _top_n_with_force_include(logits, n=3, force_ids=force_ids)

        # Check that the force id is included
        self.assertIn(3, result, "Force token (id=3) not included in the result")

        # Top-3 tokens by probability should be 0, 1, 2
        self.assertIn(0, result, "Top token (id=0) not included")
        self.assertIn(1, result, "Top token (id=1) not included")

        # Should be at least 3 tokens in the result
        self.assertGreaterEqual(len(result), 3, "Result should have at least 3 tokens")

    def test_dcbs_force_include(self):
        """Test that DCBS properly includes forced tokens in clustering."""
        # Create logits with very low probability for id=3 (answer), high for others
        logits = torch.tensor([0.9, 0.8, 0.7, 1e-8, 0.6], dtype=torch.float)
        answer_ids = {3}  # The answer token has very low probability

        # Create a simple embedding table for testing (identity matrix)
        embed_table = torch.eye(5, dtype=torch.float)

        # Ensure we're consistently selecting forced tokens
        for _ in range(5):
            # Run DCBS with forced include of answer token
            pred = category_sample(
                logits=logits,
                k=3,
                top_n=3,
                filter_tokens=answer_ids,
                embed_table=embed_table,
            )

            # Check that the prediction is in the answer tokens
            self.assertIn(
                pred,
                answer_ids,
                f"DCBS failed to include forced token. Predicted {pred} instead of {answer_ids}",
            )

    def test_multiple_forced_tokens(self):
        """Test DCBS with multiple forced tokens."""
        # Create logits with very low probabilities for id=3,4 (answers), high for others
        logits = torch.tensor([0.9, 0.8, 0.7, 1e-8, 1e-9], dtype=torch.float)
        answer_ids = {3, 4}  # Two answer tokens with very low probabilities

        # Create a simple embedding table for testing
        embed_table = torch.eye(5, dtype=torch.float)

        # Run multiple trials of DCBS since it's probabilistic
        answer_count = 0
        trials = 10

        for _ in range(trials):
            pred = category_sample(
                logits=logits,
                k=3,
                top_n=3,
                filter_tokens=answer_ids,
                embed_table=embed_table,
            )

            # Check that the prediction is in the answer tokens
            if pred in answer_ids:
                answer_count += 1

        # Should predict an answer token in all trials
        self.assertEqual(
            answer_count,
            trials,
            f"DCBS only selected answer tokens in {answer_count}/{trials} trials",
        )


if __name__ == "__main__":
    unittest.main()
