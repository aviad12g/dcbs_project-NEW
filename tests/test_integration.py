"""
Integration tests for the complete evaluation pipeline.

Tests the full workflow from data loading through visualization generation.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from dcbs import SamplingContext
from src.evaluation_core import (
    EvaluationConfig,
    EvaluationRunner,
    ExampleProcessor,
    ModelManager,
    SamplerFactory,
    load_benchmark_data,
)
from src.visualization import AccuracyVisualizer, generate_all_visualizations


class TestDataLoading:
    """Test benchmark data loading functionality."""

    def test_load_valid_benchmark_data(self):
        """Test loading valid benchmark data."""
        # Create temporary test data
        test_data = [
            {
                "id": "test1",
                "question": "The cat sat on the mat.",
                "options": ["A", "B"],
                "correct_option": "1",
            },
            {
                "id": "test2",
                "question": "The dog ran in the park.",
                "options": ["C", "D"],
                "correct_option": "2",
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            loaded_data = load_benchmark_data(temp_path)
            assert len(loaded_data) == 2
            assert loaded_data[0]["question"] == "The cat sat on the mat."
            assert loaded_data[1]["options"] == ["C", "D"]
        finally:
            os.unlink(temp_path)

    def test_load_invalid_benchmark_data(self):
        """Test loading invalid benchmark data raises appropriate errors."""
        # Test missing file
        with pytest.raises(FileNotFoundError):
            load_benchmark_data("nonexistent_file.json")

        # Test invalid JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name

        try:
            with pytest.raises(Exception):  # DataError
                load_benchmark_data(temp_path)
        finally:
            os.unlink(temp_path)

        # Test empty data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([], f)
            temp_path = f.name

        try:
            with pytest.raises(Exception):  # DataError
                load_benchmark_data(temp_path)
        finally:
            os.unlink(temp_path)


class TestEvaluationConfig:
    """Test evaluation configuration."""

    def test_config_creation(self):
        """Test that evaluation config can be created with required parameters."""
        config = EvaluationConfig(
            model_name="test-model",
            benchmark_path="test.json",
            output_dir="test_output",
        )

        assert config.model_name == "test-model"
        assert config.benchmark_path == "test.json"
        assert config.output_dir == "test_output"
        assert config.top_p == 0.9  # Default value
        assert config.include_cot is True  # Default value


class TestSamplerFactory:
    """Test sampler factory functionality."""

    def test_create_all_samplers(self):
        """Test that factory creates all expected samplers."""
        config = EvaluationConfig(
            model_name="test-model",
            benchmark_path="test.json",
            output_dir="test_output",
            top_p=0.8,
            k=5,
            top_n=20,
        )

        samplers = SamplerFactory.create_samplers(config)

        assert "greedy" in samplers
        assert "top_p" in samplers
        assert "dcbs" in samplers
        assert "random" in samplers

        # Test that configurations are applied
        assert samplers["top_p"].p == 0.8


class TestExampleProcessor:
    """Test example processing functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create mock model and tokenizer
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        self.mock_context = SamplingContext(device=Mock())

        # Setup mock tokenizer behaviors
        self.mock_tokenizer.encode.return_value = [1, 2, 3]
        self.mock_tokenizer.decode.return_value = "A"
        self.mock_tokenizer.apply_chat_template.return_value = "test prompt"

        # Setup mock model output
        mock_outputs = Mock()
        mock_outputs.logits = Mock()
        mock_outputs.logits.__getitem__ = Mock(return_value=Mock())
        mock_outputs.logits.__getitem__.return_value.squeeze.return_value = Mock()

        self.mock_model.return_value = mock_outputs

        self.processor = ExampleProcessor(
            self.mock_model, self.mock_tokenizer, self.mock_context
        )

    def test_create_reasoning_messages(self):
        """Test reasoning message creation."""
        sentence = "The cat sat on the mat."
        options = ["A", "B"]

        messages = self.processor.create_reasoning_messages(sentence, options)
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "step by step" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "A. A" in messages[1]["content"]
        assert "B. B" in messages[1]["content"]

    def test_get_answer_token_ids(self):
        """Test answer token ID extraction."""
        options = ["A", "B"]

        # Mock different tokenization scenarios
        def mock_encode(text, add_special_tokens=False):
            if text == "A":
                return [65]  # Single token
            elif text == " A":
                return [32, 65]  # Multiple tokens
            elif text == "B":
                return [66]
            elif text == " B":
                return [32, 66]
            return [1]  # Fallback

        self.mock_tokenizer.encode.side_effect = mock_encode

        answer_ids = self.processor._get_answer_token_ids(options)
        assert "A" in answer_ids
        assert "B" in answer_ids
        assert isinstance(answer_ids["A"], int)
        assert isinstance(answer_ids["B"], int)


class TestVisualization:
    """Test visualization functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.test_results = {
            "statistics": {
                "greedy": {
                    "accuracy": 75.0,
                    "correct": 15,
                    "total": 20,
                    "avg_time_ms": 500.0,
                    "confidence_interval": (55.0, 89.0),
                },
                "top_p": {
                    "accuracy": 70.0,
                    "correct": 14,
                    "total": 20,
                    "avg_time_ms": 520.0,
                    "confidence_interval": (50.0, 85.0),
                },
                "dcbs": {
                    "accuracy": 80.0,
                    "correct": 16,
                    "total": 20,
                    "avg_time_ms": 600.0,
                    "confidence_interval": (60.0, 92.0),
                },
                "random": {
                    "accuracy": 25.0,
                    "correct": 5,
                    "total": 20,
                    "avg_time_ms": 450.0,
                    "confidence_interval": (10.0, 45.0),
                },
            },
            "config": {
                "model": "test-model",
                "total_examples": 20,
                "methods": ["greedy", "top_p", "dcbs", "random"],
                "include_cot": True,
            },
        }

    def test_visualization_data_extraction(self):
        """Test that visualization can extract data correctly."""
        visualizer = AccuracyVisualizer()
        methods, accuracies, intervals, sample_sizes = visualizer._extract_data(
            self.test_results
        )

        assert len(methods) == 4
        assert "greedy" in methods
        assert "dcbs" in methods
        assert len(accuracies) == 4
        assert max(accuracies) > 75  # DCBS should be highest
        assert min(accuracies) < 55  # Random should be lowest

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_chart_generation(self, mock_close, mock_savefig):
        """Test that charts can be generated without errors."""
        visualizer = AccuracyVisualizer()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_chart.png")

            # This should not raise an exception
            visualizer.create_accuracy_chart(self.test_results, output_path)

            # Verify that matplotlib methods were called
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()

    def test_generate_all_visualizations(self):
        """Test that all visualizations can be generated."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock matplotlib to avoid actual file creation
            with (
                patch("matplotlib.pyplot.savefig"),
                patch("matplotlib.pyplot.close"),
                patch("builtins.open", create=True) as mock_open,
            ):

                # This should not raise an exception
                generate_all_visualizations(self.test_results, temp_dir)

                # Verify that files would be created
                assert mock_open.called


class TestEndToEndIntegration:
    """Test complete end-to-end evaluation pipeline."""

    def test_complete_pipeline_mock(self):
        """Test the complete pipeline with mocked components."""
        # Create test data
        test_data = [
            {
                "id": "test1",
                "question": "The cat sat on the mat.",
                "options": ["A", "B"],
                "correct_option": "1",
            }
        ]

        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save test data
            benchmark_path = os.path.join(temp_dir, "test_data.json")
            with open(benchmark_path, "w") as f:
                json.dump(test_data, f)

            # Create config
            config = EvaluationConfig(
                model_name="test-model",
                benchmark_path=benchmark_path,
                output_dir=temp_dir,
                limit=1,  # Just one example for testing
            )

            # Mock the model loading and evaluation
            with (
                patch("src.evaluation_core.model_manager.ModelManager.load_model") as mock_load,
                patch(
                    "src.evaluation_core.example_processor.ExampleProcessor.process_example"
                ) as mock_process,
                patch("matplotlib.pyplot.savefig"),
                patch("matplotlib.pyplot.close"),
            ):

                # Setup mocks
                mock_model = Mock()
                mock_tokenizer = Mock()
                mock_context = SamplingContext()
                mock_load.return_value = (mock_model, mock_tokenizer, mock_context)

                mock_process.return_value = {
                    "id": "test1",
                    "sentence": "Test sentence",
                    "options": ["A", "B"],
                    "correct_answer": "A",
                    "correct_option": "1",
                    "answer_ids": {"A": 65, "B": 66},
                    "filter_tokens": {65, 66},
                    "correct_id": 65,
                    "logits": Mock(),
                    "answer_probs": {"A": 0.7, "B": 0.3},
                    "processing_time": 0.1,
                }

                # Create and run evaluation
                runner = EvaluationRunner(config)

                # This should complete without errors
                results = runner.run_evaluation(test_data)

                # Verify results structure
                assert "statistics" in results
                assert "config" in results
                assert len(results["statistics"]) == 4  # All four methods


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
