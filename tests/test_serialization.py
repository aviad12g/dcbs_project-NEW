"""
Unit tests for enhanced JSON serialization utilities.

This module tests the SerializationUtils class and related functions
to ensure proper handling of numpy types, torch tensors, and other
non-JSON-serializable objects.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

# Import the module under test
from src.utils.serialization import (
    SerializationUtils,
    SerializationError,
    convert_numpy_types,
    serialize_evaluation_results,
    serialize_checkpoint_state,
)

# Optional torch import for testing
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestSerializationUtils(unittest.TestCase):
    """Test cases for SerializationUtils class."""

    def test_basic_types(self):
        """Test serialization of basic JSON-serializable types."""
        # Test basic types that should pass through unchanged
        test_cases = [
            None,
            True,
            False,
            42,
            3.14,
            "hello world",
            "",
        ]
        
        for value in test_cases:
            with self.subTest(value=value):
                result = SerializationUtils.convert_to_json_serializable(value)
                self.assertEqual(result, value)
                # Ensure result is JSON serializable
                json.dumps(result)

    def test_numpy_integer_types(self):
        """Test conversion of numpy integer types."""
        test_cases = [
            (np.int8(42), 42),
            (np.int16(42), 42),
            (np.int32(42), 42),
            (np.int64(42), 42),
            (np.uint8(42), 42),
            (np.uint16(42), 42),
            (np.uint32(42), 42),
            (np.uint64(42), 42),
        ]
        
        for numpy_val, expected in test_cases:
            with self.subTest(numpy_val=numpy_val):
                result = SerializationUtils.convert_to_json_serializable(numpy_val)
                self.assertEqual(result, expected)
                self.assertIsInstance(result, int)
                # Ensure result is JSON serializable
                json.dumps(result)

    def test_numpy_floating_types(self):
        """Test conversion of numpy floating point types."""
        test_cases = [
            (np.float16(3.14), 3.14),
            (np.float32(3.14), 3.14),
            (np.float64(3.14), 3.14),
        ]
        
        for numpy_val, expected in test_cases:
            with self.subTest(numpy_val=numpy_val):
                result = SerializationUtils.convert_to_json_serializable(numpy_val)
                self.assertAlmostEqual(result, expected, places=2)
                self.assertIsInstance(result, float)
                # Ensure result is JSON serializable
                json.dumps(result)

    def test_numpy_boolean_types(self):
        """Test conversion of numpy boolean types."""
        test_cases = [
            (np.bool_(True), True),
            (np.bool_(False), False),
        ]
        
        for numpy_val, expected in test_cases:
            with self.subTest(numpy_val=numpy_val):
                result = SerializationUtils.convert_to_json_serializable(numpy_val)
                self.assertEqual(result, expected)
                self.assertIsInstance(result, bool)
                # Ensure result is JSON serializable
                json.dumps(result)

    def test_numpy_arrays_small(self):
        """Test conversion of small numpy arrays."""
        # Test 1D array
        arr_1d = np.array([1, 2, 3, 4, 5])
        result = SerializationUtils.convert_to_json_serializable(arr_1d)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["type"], "numpy_array")
        self.assertEqual(result["shape"], [5])
        self.assertEqual(result["data"], [1, 2, 3, 4, 5])
        self.assertIn("dtype", result)
        
        # Ensure result is JSON serializable
        json.dumps(result)
        
        # Test 2D array
        arr_2d = np.array([[1, 2], [3, 4]])
        result = SerializationUtils.convert_to_json_serializable(arr_2d)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["type"], "numpy_array")
        self.assertEqual(result["shape"], [2, 2])
        self.assertEqual(result["data"], [[1, 2], [3, 4]])
        
        # Ensure result is JSON serializable
        json.dumps(result)

    def test_numpy_arrays_large(self):
        """Test conversion of large numpy arrays (summary format)."""
        # Create large array (> 1000 elements)
        large_arr = np.random.rand(50, 50)  # 2500 elements
        result = SerializationUtils.convert_to_json_serializable(large_arr)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["type"], "numpy_array_large")
        self.assertEqual(result["shape"], [50, 50])
        self.assertEqual(result["size"], 2500)
        self.assertIn("summary", result)
        self.assertIn("min", result["summary"])
        self.assertIn("max", result["summary"])
        self.assertIn("mean", result["summary"])
        
        # Ensure result is JSON serializable
        json.dumps(result)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_torch_tensors(self):
        """Test conversion of torch tensors."""
        # Test CPU tensor
        tensor_cpu = torch.tensor([1.0, 2.0, 3.0])
        result = SerializationUtils.convert_to_json_serializable(tensor_cpu)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["original_type"], "torch_tensor")
        self.assertEqual(result["data"], [1.0, 2.0, 3.0])
        self.assertIn("device", result)
        self.assertIn("requires_grad", result)
        
        # Ensure result is JSON serializable
        json.dumps(result)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_torch_tensors_with_grad(self):
        """Test conversion of torch tensors with gradients."""
        tensor_grad = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        # Verify the tensor actually requires grad before conversion
        self.assertTrue(tensor_grad.requires_grad)
        
        result = SerializationUtils.convert_to_json_serializable(tensor_grad)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["original_type"], "torch_tensor")
        self.assertEqual(result["requires_grad"], True)
        
        # Ensure result is JSON serializable
        json.dumps(result)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_torch_device_and_dtype(self):
        """Test conversion of torch device and dtype objects."""
        device = torch.device('cpu')
        result = SerializationUtils.convert_to_json_serializable(device)
        self.assertEqual(result, 'cpu')
        
        dtype = torch.float32
        result = SerializationUtils.convert_to_json_serializable(dtype)
        self.assertEqual(result, 'torch.float32')

    def test_dictionaries(self):
        """Test conversion of dictionaries with mixed types."""
        test_dict = {
            "string": "value",
            "int": 42,
            "numpy_int": np.int64(123),
            "numpy_float": np.float32(3.14),
            "numpy_array": np.array([1, 2, 3]),
            "nested": {
                "numpy_bool": np.bool_(True),
                "list": [np.int32(1), np.int32(2)]
            }
        }
        
        result = SerializationUtils.convert_to_json_serializable(test_dict)
        
        # Check basic types
        self.assertEqual(result["string"], "value")
        self.assertEqual(result["int"], 42)
        
        # Check numpy conversions
        self.assertEqual(result["numpy_int"], 123)
        self.assertAlmostEqual(result["numpy_float"], 3.14, places=2)
        self.assertIsInstance(result["numpy_array"], dict)
        
        # Check nested conversions
        self.assertEqual(result["nested"]["numpy_bool"], True)
        self.assertEqual(result["nested"]["list"], [1, 2])
        
        # Ensure result is JSON serializable
        json.dumps(result)

    def test_lists_and_tuples(self):
        """Test conversion of lists and tuples with mixed types."""
        test_list = [
            "string",
            42,
            np.int64(123),
            np.array([1, 2]),
            {"nested": np.float32(3.14)}
        ]
        
        result = SerializationUtils.convert_to_json_serializable(test_list)
        
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], "string")
        self.assertEqual(result[1], 42)
        self.assertEqual(result[2], 123)
        self.assertIsInstance(result[3], dict)  # numpy array
        self.assertAlmostEqual(result[4]["nested"], 3.14, places=2)
        
        # Test tuple (should become list)
        test_tuple = (np.int32(1), np.int32(2), np.int32(3))
        result = SerializationUtils.convert_to_json_serializable(test_tuple)
        self.assertEqual(result, [1, 2, 3])
        
        # Ensure result is JSON serializable
        json.dumps(result)

    def test_sets(self):
        """Test conversion of sets to lists."""
        test_set = {1, 2, 3, "string"}
        result = SerializationUtils.convert_to_json_serializable(test_set)
        
        self.assertIsInstance(result, list)
        self.assertEqual(set(result), test_set)
        
        # Ensure result is JSON serializable
        json.dumps(result)

    def test_dataclass_objects(self):
        """Test conversion of dataclass objects."""
        from dataclasses import dataclass
        
        @dataclass
        class TestDataclass:
            name: str
            value: int
            numpy_val: np.int64
        
        obj = TestDataclass("test", 42, np.int64(123))
        result = SerializationUtils.convert_to_json_serializable(obj)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["name"], "test")
        self.assertEqual(result["value"], 42)
        self.assertEqual(result["numpy_val"], 123)
        
        # Ensure result is JSON serializable
        json.dumps(result)

    def test_objects_with_dict(self):
        """Test conversion of objects with __dict__ attribute."""
        class TestObject:
            def __init__(self):
                self.name = "test"
                self.numpy_val = np.int64(42)
        
        obj = TestObject()
        result = SerializationUtils.convert_to_json_serializable(obj)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["name"], "test")
        self.assertEqual(result["numpy_val"], 42)
        
        # Ensure result is JSON serializable
        json.dumps(result)

    def test_unknown_types_fallback(self):
        """Test fallback to string conversion for unknown types."""
        class UnknownType:
            def __str__(self):
                return "unknown_object"
            
            # Override __dict__ to prevent dict serialization
            @property
            def __dict__(self):
                return None
        
        obj = UnknownType()
        
        with patch('src.utils.serialization.logger') as mock_logger:
            result = SerializationUtils.convert_to_json_serializable(obj)
            
            self.assertEqual(result, "unknown_object")
            mock_logger.warning.assert_called_once()
        
        # Ensure result is JSON serializable
        json.dumps(result)

    def test_serialization_error_handling(self):
        """Test error handling for objects that cannot be serialized."""
        class UnserializableObject:
            def __str__(self):
                raise Exception("Cannot convert to string")
            
            @property
            def __dict__(self):
                raise Exception("Cannot access dict")
        
        obj = UnserializableObject()
        
        with self.assertRaises(SerializationError) as context:
            SerializationUtils.convert_to_json_serializable(obj)
        
        self.assertIn("Unexpected error during serialization", str(context.exception))
        self.assertIn("UnserializableObject", str(context.exception))

    def test_path_tracking_in_errors(self):
        """Test that error paths are correctly tracked."""
        class UnserializableObject:
            def __str__(self):
                raise Exception("Cannot convert to string")
            
            @property
            def __dict__(self):
                raise Exception("Cannot access dict")
        
        problematic_dict = {
            "level1": {
                "level2": {
                    "problematic": UnserializableObject()
                }
            }
        }
        
        with self.assertRaises(SerializationError) as context:
            SerializationUtils.convert_to_json_serializable(problematic_dict)
        
        error_message = str(context.exception)
        self.assertIn("root.level1.level2.problematic", error_message)

    def test_safe_json_dump(self):
        """Test safe JSON file writing."""
        test_data = {
            "string": "value",
            "numpy_int": np.int64(42),
            "numpy_array": np.array([1, 2, 3])
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            SerializationUtils.safe_json_dump(test_data, temp_path, indent=2)
            
            # Verify file was written correctly
            with open(temp_path, 'r') as f:
                loaded_data = json.load(f)
            
            self.assertEqual(loaded_data["string"], "value")
            self.assertEqual(loaded_data["numpy_int"], 42)
            self.assertIsInstance(loaded_data["numpy_array"], dict)
            
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_safe_json_dumps(self):
        """Test safe JSON string conversion."""
        test_data = {
            "numpy_int": np.int64(42),
            "numpy_float": np.float32(3.14)
        }
        
        result_str = SerializationUtils.safe_json_dumps(test_data, indent=2)
        
        # Verify string can be parsed back
        loaded_data = json.loads(result_str)
        self.assertEqual(loaded_data["numpy_int"], 42)
        self.assertAlmostEqual(loaded_data["numpy_float"], 3.14, places=2)


class TestLegacyFunctions(unittest.TestCase):
    """Test legacy compatibility functions."""

    def test_convert_numpy_types_compatibility(self):
        """Test backward compatibility of convert_numpy_types function."""
        test_data = {
            "numpy_int": np.int64(42),
            "numpy_array": np.array([1, 2, 3]),
            "regular_data": "unchanged"
        }
        
        result = convert_numpy_types(test_data)
        
        self.assertEqual(result["numpy_int"], 42)
        self.assertIsInstance(result["numpy_array"], dict)
        self.assertEqual(result["regular_data"], "unchanged")
        
        # Ensure result is JSON serializable
        json.dumps(result)

    def test_serialize_evaluation_results(self):
        """Test evaluation results serialization."""
        results = {
            "statistics": {
                "dcbs": {
                    "accuracy": np.float64(78.5),
                    "correct": np.int64(39),
                    "total": np.int64(50)
                }
            },
            "config": {
                "k": np.int32(8),
                "top_n": np.int32(50)
            }
        }
        
        serialized = serialize_evaluation_results(results)
        
        self.assertEqual(serialized["statistics"]["dcbs"]["accuracy"], 78.5)
        self.assertEqual(serialized["statistics"]["dcbs"]["correct"], 39)
        self.assertEqual(serialized["config"]["k"], 8)
        
        # Ensure result is JSON serializable
        json.dumps(serialized)

    def test_serialize_checkpoint_state_success(self):
        """Test successful checkpoint state serialization."""
        from dataclasses import dataclass
        
        @dataclass
        class MockCheckpointState:
            run_id: str
            timestamp: str
            completed_examples: int
            numpy_data: np.int64
        
        state = MockCheckpointState(
            run_id="test_run",
            timestamp="2025-01-01",
            completed_examples=np.int64(42),
            numpy_data=np.int64(123)
        )
        
        result = serialize_checkpoint_state(state)
        
        self.assertEqual(result["run_id"], "test_run")
        self.assertEqual(result["completed_examples"], 42)
        self.assertEqual(result["numpy_data"], 123)
        
        # Ensure result is JSON serializable
        json.dumps(result)

    def test_serialize_checkpoint_state_error_recovery(self):
        """Test checkpoint state serialization error recovery."""
        class UnserializableObject:
            def __str__(self):
                raise Exception("Cannot convert to string")
            
            @property
            def __dict__(self):
                raise Exception("Cannot access dict")
        
        class ProblematicState:
            def __init__(self):
                self.run_id = "test_run"
                self.timestamp = "2025-01-01"
                self.problematic_data = UnserializableObject()
            
            def __str__(self):
                # Make the state itself unserializable
                raise Exception("Cannot convert state to string")
        
        state = ProblematicState()
        
        with patch('src.utils.serialization.logger') as mock_logger:
            result = serialize_checkpoint_state(state)
            
            # Should return minimal recovery state
            self.assertIn("serialization_error", result)
            self.assertEqual(result["run_id"], "test_run")
            self.assertEqual(result["timestamp"], "2025-01-01")
            
            mock_logger.error.assert_called_once()
        
        # Ensure result is JSON serializable
        json.dumps(result)


if __name__ == '__main__':
    unittest.main()