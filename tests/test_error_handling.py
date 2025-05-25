"""
Unit tests for error handling and recovery functionality.

Tests custom exception classes, ErrorRecoveryManager, and error recovery
decorators.
"""

import unittest
from unittest.mock import Mock, patch

from src.errors import (
    DCBSError,
    DCBSRuntimeError,
    EvaluationError,
    ConfigurationError,
    ModelLoadError,
    SamplingError,
    CacheError,
    ResourceError,
    ErrorRecoveryManager,
    with_error_recovery,
)


class TestCustomExceptions(unittest.TestCase):
    """Test custom exception classes."""

    def test_dcbs_error_basic(self):
        """Test basic DCBSError functionality."""
        error = DCBSError("Test error")
        self.assertEqual(error.message, "Test error")
        self.assertEqual(error.details, {})
        self.assertFalse(error.recoverable)

    def test_dcbs_error_with_details(self):
        """Test DCBSError with details and recovery flag."""
        details = {"key": "value", "number": 42}
        error = DCBSError("Test error", details=details, recoverable=True)
        
        self.assertEqual(error.message, "Test error")
        self.assertEqual(error.details, details)
        self.assertTrue(error.recoverable)

    def test_dcbs_error_to_dict(self):
        """Test DCBSError serialization to dictionary."""
        details = {"context": "test"}
        error = DCBSError("Test error", details=details, recoverable=True)
        
        error_dict = error.to_dict()
        expected = {
            "error_type": "DCBSError",
            "message": "Test error",
            "details": details,
            "recoverable": True,
        }
        
        self.assertEqual(error_dict, expected)

    def test_specific_error_types(self):
        """Test that specific error types inherit from DCBSError."""
        errors = [
            DCBSRuntimeError("Runtime error"),
            EvaluationError("Evaluation error"),
            ConfigurationError("Config error"),
            ModelLoadError("Model load error"),
            SamplingError("Sampling error"),
            CacheError("Cache error"),
            ResourceError("Resource error"),
        ]
        
        for error in errors:
            self.assertIsInstance(error, DCBSError)
            self.assertIsInstance(error.message, str)


class TestErrorRecoveryManager(unittest.TestCase):
    """Test ErrorRecoveryManager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.recovery_manager = ErrorRecoveryManager()

    def test_recovery_manager_initialization(self):
        """Test ErrorRecoveryManager initialization."""
        self.assertIsNotNone(self.recovery_manager.logger)
        self.assertIsInstance(self.recovery_manager.recovery_strategies, dict)
        self.assertIn(ModelLoadError, self.recovery_manager.recovery_strategies)

    def test_attempt_recovery_non_recoverable(self):
        """Test recovery attempt on non-recoverable error."""
        error = DCBSError("Non-recoverable error", recoverable=False)
        result = self.recovery_manager.attempt_recovery(error)
        self.assertFalse(result)

    def test_attempt_recovery_no_strategy(self):
        """Test recovery attempt for error type without strategy."""
        error = EvaluationError("Evaluation error", recoverable=True)
        result = self.recovery_manager.attempt_recovery(error)
        self.assertFalse(result)

    def test_model_load_recovery_4bit(self):
        """Test model load recovery with 4-bit quantization."""
        error = ModelLoadError("Model load failed", recoverable=True)
        context = {"load_in_4bit": False}
        
        result = self.recovery_manager.attempt_recovery(error, context)
        
        self.assertTrue(result)
        self.assertTrue(context["load_in_4bit"])

    def test_model_load_recovery_cpu_fallback(self):
        """Test model load recovery with CPU fallback."""
        error = ModelLoadError("Model load failed", recoverable=True)
        context = {"load_in_4bit": True, "device": "cuda"}
        
        result = self.recovery_manager.attempt_recovery(error, context)
        
        self.assertTrue(result)
        self.assertEqual(context["device"], "cpu")

    def test_model_load_recovery_exhausted(self):
        """Test model load recovery when all options are exhausted."""
        error = ModelLoadError("Model load failed", recoverable=True)
        context = {"load_in_4bit": True, "device": "cpu"}
        
        result = self.recovery_manager.attempt_recovery(error, context)
        
        self.assertFalse(result)

    @patch('shutil.rmtree')
    @patch('os.path.exists')
    def test_cache_recovery_success(self, mock_exists, mock_rmtree):
        """Test successful cache error recovery."""
        mock_exists.return_value = True
        
        error = CacheError("Cache error", recoverable=True)
        context = {"cache_dir": "/test/cache"}
        
        result = self.recovery_manager.attempt_recovery(error, context)
        
        self.assertTrue(result)
        self.assertFalse(context["enable_caching"])
        mock_rmtree.assert_called_once_with("/test/cache")

    @patch('shutil.rmtree')
    @patch('os.path.exists')
    def test_cache_recovery_failure(self, mock_exists, mock_rmtree):
        """Test cache error recovery failure."""
        mock_exists.return_value = True
        mock_rmtree.side_effect = Exception("Permission denied")
        
        error = CacheError("Cache error", recoverable=True)
        context = {"cache_dir": "/test/cache"}
        
        result = self.recovery_manager.attempt_recovery(error, context)
        
        self.assertFalse(result)

    @patch('gc.collect')
    def test_resource_recovery_batch_size(self, mock_gc):
        """Test resource error recovery by reducing batch size."""
        error = ResourceError("Out of memory", recoverable=True)
        context = {"batch_size": 8}
        
        result = self.recovery_manager.attempt_recovery(error, context)
        
        self.assertTrue(result)
        self.assertEqual(context["batch_size"], 4)

    @patch('gc.collect')
    def test_resource_recovery_gc_only(self, mock_gc):
        """Test resource error recovery with garbage collection only."""
        error = ResourceError("Out of memory", recoverable=True)
        context = {"batch_size": 1}
        
        result = self.recovery_manager.attempt_recovery(error, context)
        
        self.assertTrue(result)
        mock_gc.assert_called_once()

    def test_sampling_recovery(self):
        """Test sampling error recovery."""
        error = SamplingError("Sampling failed", recoverable=True)
        context = {"sampler_type": "dcbs"}
        
        result = self.recovery_manager.attempt_recovery(error, context)
        
        self.assertTrue(result)
        self.assertEqual(context["sampler_type"], "greedy")

    def test_sampling_recovery_already_greedy(self):
        """Test sampling error recovery when already using greedy."""
        error = SamplingError("Sampling failed", recoverable=True)
        context = {"sampler_type": "greedy"}
        
        result = self.recovery_manager.attempt_recovery(error, context)
        
        self.assertFalse(result)

    def test_recovery_strategy_exception(self):
        """Test recovery when strategy itself raises an exception."""
        error = ModelLoadError("Model load failed", recoverable=True)
        context = {}
        
        # Mock the recovery method to raise an exception
        original_method = self.recovery_manager._recover_model_load
        self.recovery_manager._recover_model_load = Mock(side_effect=Exception("Recovery failed"))
        
        result = self.recovery_manager.attempt_recovery(error, context)
        
        self.assertFalse(result)
        
        # Restore original method
        self.recovery_manager._recover_model_load = original_method


class TestErrorRecoveryDecorator(unittest.TestCase):
    """Test the with_error_recovery decorator."""

    def setUp(self):
        """Set up test fixtures."""
        self.recovery_manager = ErrorRecoveryManager()

    def test_decorator_success_no_retry(self):
        """Test decorator when function succeeds on first try."""
        @with_error_recovery(self.recovery_manager)
        def successful_function():
            return "success"
        
        result = successful_function()
        self.assertEqual(result, "success")

    def test_decorator_recoverable_error_success(self):
        """Test decorator with recoverable error that succeeds on retry."""
        call_count = 0
        
        @with_error_recovery(self.recovery_manager)
        def function_with_recoverable_error(**kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                raise ModelLoadError("Load failed", recoverable=True)
            return "success"
        
        # Mock the recovery to succeed
        with patch.object(self.recovery_manager, 'attempt_recovery', return_value=True):
            result = function_with_recoverable_error()
            self.assertEqual(result, "success")
            self.assertEqual(call_count, 2)

    def test_decorator_non_recoverable_error(self):
        """Test decorator with non-recoverable error."""
        @with_error_recovery(self.recovery_manager)
        def function_with_non_recoverable_error():
            raise ModelLoadError("Load failed", recoverable=False)
        
        with self.assertRaises(ModelLoadError):
            function_with_non_recoverable_error()

    def test_decorator_max_retries_exceeded(self):
        """Test decorator when max retries are exceeded."""
        @with_error_recovery(self.recovery_manager)
        def function_always_fails():
            raise ModelLoadError("Load failed", recoverable=True)
        
        with patch.object(self.recovery_manager, 'attempt_recovery', return_value=True):
            with self.assertRaises(ModelLoadError):
                function_always_fails(max_retries=2)

    def test_decorator_recovery_fails(self):
        """Test decorator when recovery fails."""
        @with_error_recovery(self.recovery_manager)
        def function_with_unrecoverable_failure():
            raise ModelLoadError("Load failed", recoverable=True)
        
        with patch.object(self.recovery_manager, 'attempt_recovery', return_value=False):
            with self.assertRaises(ModelLoadError):
                function_with_unrecoverable_failure()

    def test_decorator_non_dcbs_error(self):
        """Test decorator with non-DCBS exception."""
        @with_error_recovery(self.recovery_manager)
        def function_with_standard_error():
            raise ValueError("Standard error")
        
        with self.assertRaises(DCBSRuntimeError) as cm:
            function_with_standard_error()
        
        self.assertIn("Unexpected error", str(cm.exception))
        self.assertIsInstance(cm.exception.__cause__, ValueError)

    def test_decorator_default_recovery_manager(self):
        """Test decorator with default recovery manager."""
        @with_error_recovery()
        def function_with_default_manager():
            return "success"
        
        result = function_with_default_manager()
        self.assertEqual(result, "success")

    def test_decorator_kwargs_modification(self):
        """Test that decorator properly handles kwargs modification during recovery."""
        call_count = 0
        
        @with_error_recovery(self.recovery_manager)
        def function_with_kwargs(**kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                raise ModelLoadError("Load failed", recoverable=True)
            
            # Check that kwargs were modified by recovery
            self.assertTrue(kwargs.get("load_in_4bit", False))
            return "success"
        
        # Mock recovery to modify kwargs
        def mock_recovery(error, context):
            context["load_in_4bit"] = True
            return True
        
        with patch.object(self.recovery_manager, 'attempt_recovery', side_effect=mock_recovery):
            result = function_with_kwargs(load_in_4bit=False)
            self.assertEqual(result, "success")


if __name__ == "__main__":
    unittest.main() 