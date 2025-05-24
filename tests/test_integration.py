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
    ChatTemplateManager,
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
                "sentence": "The cat sat on the mat.",
                "option1": "A",
                "option2": "B",
                "correct_option": "1",
            },
            {
                "id": "test2",
                "sentence": "The dog ran in the park.",
                "option1": "C",
                "option2": "D",
                "correct_option": "2",
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            loaded_data = load_benchmark_data(temp_path)
            assert len(loaded_data) == 2
            assert loaded_data[0]["sentence"] == "The cat sat on the mat."
            assert loaded_data[1]["option2"] == "D"
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
        assert "top-p" in samplers
        assert "dcbs" in samplers
        assert "random" in samplers

        # Test that configurations are applied
        assert samplers["top-p"].p == 0.8
        assert samplers["dcbs"].k == 5
        assert samplers["dcbs"].top_n == 20


class TestChatTemplateManager:
    """Test chat template management."""

    def test_template_selection(self):
        """Test that appropriate templates are selected based on model names."""
        mock_tokenizer = Mock()
        mock_tokenizer.chat_template = None

        # Test Llama template
        ChatTemplateManager.setup_chat_template(mock_tokenizer, "meta-llama/Llama-2-7b")
        assert mock_tokenizer.chat_template is not None
        assert (
            "llama" in mock_tokenizer.chat_template.lower()
            or "start_header_id" in mock_tokenizer.chat_template
        )

        # Test Mistral template
        mock_tokenizer.chat_template = None
        ChatTemplateManager.setup_chat_template(mock_tokenizer, "mistralai/Mistral-7B")
        assert mock_tokenizer.chat_template is not None
        assert "[INST]" in mock_tokenizer.chat_template

        # Test generic template
        mock_tokenizer.chat_template = None
        ChatTemplateManager.setup_chat_template(mock_tokenizer, "unknown-model")
        assert mock_tokenizer.chat_template is not None

    def test_template_validation(self):
        """Test template validation functionality."""
        # Mock successful validation
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "Test prompt"

        result = ChatTemplateManager.validate_template(mock_tokenizer, "test-model")
        assert result is True

        # Mock failed validation
        mock_tokenizer.apply_chat_template.side_effect = Exception("Template error")
        result = ChatTemplateManager.validate_template(mock_tokenizer, "test-model")
        assert result is False


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

        # Setup mock model output
        mock_outputs = Mock()
        mock_outputs.logits = Mock()
        mock_outputs.logits.__getitem__ = Mock(return_value=Mock())
        mock_outputs.logits.__getitem__.return_value.squeeze.return_value = Mock()

        self.mock_model.return_value = mock_outputs

        self.processor = ExampleProcessor(
            self.mock_model, self.mock_tokenizer, self.mock_context
        )

    def test_create_prompt(self):
        """Test prompt creation for examples."""
        sentence = "The cat sat on the mat."
        options = ["A", "B"]

        # Test with CoT
        prompt_cot = self.processor.create_prompt(sentence, options, include_cot=True)
        assert "Think step by step" in prompt_cot
        assert "A. A" in prompt_cot
        assert "B. B" in prompt_cot

        # Test without CoT
        prompt_no_cot = self.processor.create_prompt(
            sentence, options, include_cot=False
        )
        assert "Think step by step" not in prompt_no_cot
        assert "A. A" in prompt_no_cot

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

        answer_ids = self.processor.get_answer_token_ids(options)
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
                    "correct": 75,
                    "total": 100,
                    "avg_time_ms": 10.5,
                    "confidence_interval": (65.5, 84.5),
                },
                "top-p": {
                    "accuracy": 73.2,
                    "correct": 73,
                    "total": 100,
                    "avg_time_ms": 12.3,
                    "confidence_interval": (63.2, 83.2),
                },
                "dcbs": {
                    "accuracy": 77.8,
                    "correct": 78,
                    "total": 100,
                    "avg_time_ms": 15.1,
                    "confidence_interval": (68.1, 87.5),
                },
                "random": {
                    "accuracy": 49.5,
                    "correct": 49,
                    "total": 100,
                    "avg_time_ms": 8.9,
                    "confidence_interval": (39.2, 59.8),
                },
            },
            "config": {
                "model": "test-model",
                "total_examples": 100,
                "methods": ["greedy", "top-p", "dcbs", "random"],
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
                "sentence": "The cat sat on the mat.",
                "option1": "A",
                "option2": "B",
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
                patch("evaluation_core.ModelManager.load_model") as mock_load,
                patch(
                    "evaluation_core.ExampleProcessor.process_example"
                ) as mock_process,
                patch(
                    "evaluation_core.ExampleProcessor.get_answer_token_ids"
                ) as mock_get_ids,
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
