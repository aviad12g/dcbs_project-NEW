"""
Chat template management for different model families.

This module handles chat template setup and validation for various
language model architectures.
"""

from src.errors import eval_logger as logger


class ChatTemplateManager:
    """Manages chat templates for different model families."""

    @classmethod
    def setup_chat_template(cls, tokenizer, model_name: str) -> None:
        """
        Set up appropriate chat template for the given model.
        
        Args:
            tokenizer: The model tokenizer
            model_name: Name of the model to configure template for
        """
        # Check if model already has a template
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            logger.info("Using existing chat template")
            return

        # Use the chat_templates module for proper template handling
        try:
            from chat_templates import get_template_for_model

            tokenizer.chat_template = get_template_for_model(
                tokenizer.name_or_path or model_name
            )
            logger.info(f"Applied chat template for model: {model_name}")
        except ImportError:
            # Fallback to simple Llama 3 template if chat_templates module not available
            if "llama" in model_name.lower() and "3" in model_name:
                template = (
                    "{% if messages[0]['role'] == 'system' %}"
                    "{% set loop_messages = messages[1:] %}"
                    "{% set system_message = messages[0]['content'] %}"
                    "{% else %}"
                    "{% set loop_messages = messages %}"
                    "{% set system_message = false %}"
                    "{% endif %}"
                    "{% for message in loop_messages %}"
                    "{% if loop.index0 == 0 and system_message %}"
                    "{{ '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n' + system_message + '<|eot_id|>' }}"
                    "{% endif %}"
                    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
                    "{% if loop.last and add_generation_prompt %}"
                    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
                    "{% endif %}"
                    "{% endfor %}"
                )
                tokenizer.chat_template = template
                logger.info("Applied fallback Llama 3 chat template")
            else:
                # Generic fallback
                template = (
                    "{% for message in messages %}"
                    "{{ message['role'].title() + ': ' + message['content'] + '\\n' }}"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}"
                    "{{ 'Assistant: ' }}"
                    "{% endif %}"
                )
                tokenizer.chat_template = template
                logger.info("Applied generic fallback chat template")

    @staticmethod
    def validate_template(tokenizer, model_name: str) -> bool:
        """
        Validate that the chat template works correctly.
        
        Args:
            tokenizer: The tokenizer to validate
            model_name: Name of the model for logging
            
        Returns:
            True if template is valid, False otherwise
        """
        try:
            test_messages = [{"role": "user", "content": "Test message"}]
            result = tokenizer.apply_chat_template(
                test_messages, tokenize=False, add_generation_prompt=True
            )
            return len(result) > 0
        except Exception as e:
            logger.warning(f"Chat template validation failed: {e}")
            return False 