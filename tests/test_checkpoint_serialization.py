"""
Tests for checkpoint serialization with numpy types.

This module tests that the checkpoint manager can properly handle
numpy types and other non-JSON-serializable objects in checkpoint data.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.evaluation_core.checkpoint import CheckpointManager, CheckpointState


class TestCheckpointSerialization(unittest.TestCase):
    """Test checkpoint serialization with various data types."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.temp_dir)

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_checkpoint_with_numpy_types(self):
        """Test checkpoint saving and loading with numpy types."""
        # Create checkpoint state with numpy types
        state = CheckpointState(
            run_id="test_run",
            timestamp="2025-01-01T00:00:00",
            total_examples=np.int64(100),
            completed_examples=np.int64(50),
            current_example_idx=np.int64(50),
            sampler_states={
                "dcbs": {
                    "k": np.int32(8),
                    "top_n": np.int32(50),
                    "accuracy": np.float64(0.75)
                }
            },
            results=[
                {
                    "id": "example_1",
                    "correct": np.bool_(True),
                    "score": np.float32(0.95),
                    "logits": np.array([1.0, 2.0, 3.0])
                }
            ],
            config={
                "batch_size": np.int64(32),
                "learning_rate": np.float64(0.001)
            }
        )

        # Save checkpoint
        self.checkpoint_manager.save_checkpoint(state)

        # Verify checkpoint file exists
        checkpoint_path = self.checkpoint_manager.get_checkpoint_path("test_run")
        self.assertTrue(checkpoint_path.exists())

        # Load and verify checkpoint
        loaded_state = self.checkpoint_manager.load_checkpoint("test_run")
        self.assertIsNotNone(loaded_state)
        
        # Verify basic fields
        self.assertEqual(loaded_state.run_id, "test_run")
        self.assertEqual(loaded_state.total_examples, 100)
        self.assertEqual(loaded_state.completed_examples, 50)
        
        # Verify numpy types were converted properly
        self.assertIsInstance(loaded_state.total_examples, int)
        self.assertIsInstance(loaded_state.completed_examples, int)
        
        # Verify nested numpy types in sampler_states
        dcbs_state = loaded_state.sampler_states["dcbs"]
        self.assertEqual(dcbs_state["k"], 8)
        self.assertEqual(dcbs_state["top_n"], 50)
        self.assertAlmostEqual(dcbs_state["accuracy"], 0.75)
        
        # Verify numpy types in results
        result = loaded_state.results[0]
        self.assertEqual(result["correct"], True)
        self.assertAlmostEqual(result["score"], 0.95, places=2)
        
        # Verify numpy array was serialized properly
        self.assertIn("logits", result)
        logits_data = result["logits"]
        if isinstance(logits_data, dict):
            # Array was serialized as dict with metadata
            self.assertEqual(logits_data["data"], [1.0, 2.0, 3.0])
        else:
            # Array was serialized as list
            self.assertEqual(logits_data, [1.0, 2.0, 3.0])

    def test_checkpoint_serialization_error_recovery(self):
        """Test checkpoint error recovery for unserializable objects."""
        # Create a problematic object that can't be serialized
        class UnserializableObject:
            def __str__(self):
                raise Exception("Cannot convert to string")
            
            @property
            def __dict__(self):
                raise Exception("Cannot access dict")

        # Create checkpoint state with problematic data
        state = CheckpointState(
            run_id="error_test",
            timestamp="2025-01-01T00:00:00",
            total_examples=100,
            completed_examples=50,
            current_example_idx=50,
            sampler_states={},
            results=[],
            config={"problematic": UnserializableObject()}
        )

        # Save checkpoint (should handle error gracefully)
        with patch('src.evaluation_core.checkpoint.logger') as mock_logger:
            self.checkpoint_manager.save_checkpoint(state)
            
            # Should have logged serialization error
            mock_logger.error.assert_called()
            error_calls = [call for call in mock_logger.error.call_args_list 
                          if 'serialize' in str(call)]
            self.assertTrue(len(error_calls) > 0)

        # Verify minimal checkpoint was saved
        checkpoint_path = self.checkpoint_manager.get_checkpoint_path("error_test")
        self.assertTrue(checkpoint_path.exists())

        # Load checkpoint and verify minimal data
        loaded_state = self.checkpoint_manager.load_checkpoint("error_test")
        self.assertIsNotNone(loaded_state)
        self.assertEqual(loaded_state.run_id, "error_test")
        self.assertEqual(loaded_state.completed_examples, 50)

    def test_checkpoint_file_format_validation(self):
        """Test that saved checkpoint files are valid JSON."""
        # Create simple checkpoint state
        state = CheckpointState(
            run_id="format_test",
            timestamp="2025-01-01T00:00:00",
            total_examples=np.int64(10),
            completed_examples=np.int64(5),
            current_example_idx=np.int64(5),
            sampler_states={},
            results=[],
            config={}
        )

        # Save checkpoint
        self.checkpoint_manager.save_checkpoint(state)

        # Verify file is valid JSON
        checkpoint_path = self.checkpoint_manager.get_checkpoint_path("format_test")
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)  # Should not raise exception

        # Verify structure
        self.assertIn("run_id", data)
        self.assertIn("total_examples", data)
        self.assertIn("completed_examples", data)
        
        # Verify numpy types were converted to native Python types
        self.assertIsInstance(data["total_examples"], int)
        self.assertIsInstance(data["completed_examples"], int)

    def test_checkpoint_with_torch_tensors(self):
        """Test checkpoint with torch tensors if available."""
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not available")

        # Create checkpoint state with torch tensor
        tensor = torch.tensor([1.0, 2.0, 3.0])
        state = CheckpointState(
            run_id="torch_test",
            timestamp="2025-01-01T00:00:00",
            total_examples=10,
            completed_examples=5,
            current_example_idx=5,
            sampler_states={},
            results=[{
                "tensor_data": tensor,
                "tensor_with_grad": torch.tensor([4.0, 5.0], requires_grad=True)
            }],
            config={}
        )

        # Save checkpoint
        self.checkpoint_manager.save_checkpoint(state)

        # Load and verify
        loaded_state = self.checkpoint_manager.load_checkpoint("torch_test")
        self.assertIsNotNone(loaded_state)
        
        result = loaded_state.results[0]
        
        # Verify tensor was serialized properly
        tensor_data = result["tensor_data"]
        if isinstance(tensor_data, dict):
            self.assertEqual(tensor_data["data"], [1.0, 2.0, 3.0])
            self.assertEqual(tensor_data["original_type"], "torch_tensor")
        
        # Verify tensor with grad was serialized properly
        tensor_grad_data = result["tensor_with_grad"]
        if isinstance(tensor_grad_data, dict):
            self.assertEqual(tensor_grad_data["data"], [4.0, 5.0])
            self.assertEqual(tensor_grad_data["requires_grad"], True)


if __name__ == '__main__':
    unittest.main()