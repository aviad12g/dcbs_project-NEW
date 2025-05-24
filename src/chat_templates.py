"""
Comprehensive chat template library for different model families.

This module provides an extensive collection of chat templates for various
language model architectures and families, with automatic detection and
fallback mechanisms.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ModelInfo:
    """Information about a model for template selection."""

    name: str
    family: str
    version: Optional[str] = None
    special_tokens: Optional[Dict[str, str]] = None
    max_context: Optional[int] = None


class ChatTemplate(ABC):
    """Abstract base class for chat templates."""

    @abstractmethod
    def apply(
        self, messages: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """Apply the template to a list of messages."""
        pass

    @abstractmethod
    def get_special_tokens(self) -> Dict[str, str]:
        """Get special tokens used by this template."""
        pass

    @abstractmethod
    def validate_messages(self, messages: List[Dict[str, str]]) -> bool:
        """Validate that messages are compatible with this template."""
        pass


class LlamaTemplate(ChatTemplate):
    """Template for Llama family models (Llama 2, Llama 3, Code Llama)."""

    def __init__(self, version: str = "2"):
        self.version = version

    def apply(
        self, messages: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """Apply Llama template."""
        if not messages:
            return ""

        if self.version == "3":
            return self._apply_llama3(messages, add_generation_prompt)
        else:
            return self._apply_llama2(messages, add_generation_prompt)

    def _apply_llama2(
        self, messages: List[Dict[str, str]], add_generation_prompt: bool
    ) -> str:
        """Apply Llama 2 template."""
        result = ""
        system_message = None

        # Extract system message if present
        if messages and messages[0]["role"] == "system":
            system_message = messages[0]["content"]
            messages = messages[1:]

        for i, message in enumerate(messages):
            if message["role"] == "user":
                if i == 0 and system_message:
                    result += f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{message['content']} [/INST]"
                else:
                    result += f"<s>[INST] {message['content']} [/INST]"
            elif message["role"] == "assistant":
                result += f"{message['content']} </s>"

        return result

    def _apply_llama3(
        self, messages: List[Dict[str, str]], add_generation_prompt: bool
    ) -> str:
        """Apply Llama 3 template."""
        result = "<|begin_of_text|>"

        # Extract system message if present
        if messages and messages[0]["role"] == "system":
            result += f"<|start_header_id|>system<|end_header_id|>\n\n{messages[0]['content']}<|eot_id|>"
            messages = messages[1:]

        for message in messages:
            result += f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n{message['content']}<|eot_id|>"

        if add_generation_prompt:
            result += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        return result

    def get_special_tokens(self) -> Dict[str, str]:
        """Get Llama special tokens."""
        if self.version == "3":
            return {
                "bos_token": "<|begin_of_text|>",
                "eos_token": "<|eot_id|>",
                "system_start": "<|start_header_id|>system<|end_header_id|>",
                "user_start": "<|start_header_id|>user<|end_header_id|>",
                "assistant_start": "<|start_header_id|>assistant<|end_header_id|>",
            }
        else:
            return {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "inst_start": "[INST]",
                "inst_end": "[/INST]",
                "sys_start": "<<SYS>>",
                "sys_end": "<</SYS>>",
            }

    def validate_messages(self, messages: List[Dict[str, str]]) -> bool:
        """Validate messages for Llama template."""
        if not messages:
            return False

        valid_roles = {"system", "user", "assistant"}
        for message in messages:
            if "role" not in message or "content" not in message:
                return False
            if message["role"] not in valid_roles:
                return False

        return True


class MistralTemplate(ChatTemplate):
    """Template for Mistral family models."""

    def apply(
        self, messages: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """Apply Mistral template."""
        result = ""

        for i, message in enumerate(messages):
            if message["role"] == "system":
                # Mistral doesn't have explicit system role, prepend to first user message
                continue
            elif message["role"] == "user":
                # Check if there was a system message
                system_content = ""
                if i > 0 and messages[i - 1]["role"] == "system":
                    system_content = f"{messages[i-1]['content']}\n\n"
                elif i == 0 and len(messages) > 1 and messages[0]["role"] == "system":
                    system_content = f"{messages[0]['content']}\n\n"

                result += f"[INST] {system_content}{message['content']} [/INST]"
            elif message["role"] == "assistant":
                result += f"{message['content']}</s>"

        return result

    def get_special_tokens(self) -> Dict[str, str]:
        """Get Mistral special tokens."""
        return {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "inst_start": "[INST]",
            "inst_end": "[/INST]",
        }

    def validate_messages(self, messages: List[Dict[str, str]]) -> bool:
        """Validate messages for Mistral template."""
        if not messages:
            return False

        valid_roles = {"system", "user", "assistant"}
        for message in messages:
            if "role" not in message or "content" not in message:
                return False
            if message["role"] not in valid_roles:
                return False

        return True


class OpenAITemplate(ChatTemplate):
    """Template for OpenAI/GPT-style models."""

    def apply(
        self, messages: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """Apply OpenAI-style template."""
        result = ""

        for message in messages:
            role = message["role"].title()
            content = message["content"]
            result += f"{role}: {content}\n"

        if add_generation_prompt:
            result += "Assistant: "

        return result

    def get_special_tokens(self) -> Dict[str, str]:
        """Get OpenAI special tokens."""
        return {"bos_token": "", "eos_token": "", "separator": "\n"}

    def validate_messages(self, messages: List[Dict[str, str]]) -> bool:
        """Validate messages for OpenAI template."""
        if not messages:
            return False

        valid_roles = {"system", "user", "assistant"}
        for message in messages:
            if "role" not in message or "content" not in message:
                return False
            if message["role"] not in valid_roles:
                return False

        return True


class ChatMLTemplate(ChatTemplate):
    """Template for ChatML format (used by various models)."""

    def apply(
        self, messages: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """Apply ChatML template."""
        result = ""

        for message in messages:
            role = message["role"]
            content = message["content"]
            result += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        if add_generation_prompt:
            result += "<|im_start|>assistant\n"

        return result

    def get_special_tokens(self) -> Dict[str, str]:
        """Get ChatML special tokens."""
        return {
            "bos_token": "",
            "eos_token": "<|im_end|>",
            "start_token": "<|im_start|>",
            "end_token": "<|im_end|>",
        }

    def validate_messages(self, messages: List[Dict[str, str]]) -> bool:
        """Validate messages for ChatML template."""
        if not messages:
            return False

        valid_roles = {"system", "user", "assistant"}
        for message in messages:
            if "role" not in message or "content" not in message:
                return False
            if message["role"] not in valid_roles:
                return False

        return True


class AnthropicTemplate(ChatTemplate):
    """Template for Anthropic Claude models."""

    def apply(
        self, messages: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """Apply Anthropic template."""
        result = ""

        for message in messages:
            if message["role"] == "system":
                result += f"System: {message['content']}\n\n"
            elif message["role"] == "user":
                result += f"Human: {message['content']}\n\n"
            elif message["role"] == "assistant":
                result += f"Assistant: {message['content']}\n\n"

        if add_generation_prompt:
            result += "Assistant: "

        return result

    def get_special_tokens(self) -> Dict[str, str]:
        """Get Anthropic special tokens."""
        return {
            "bos_token": "",
            "eos_token": "",
            "human_prefix": "Human:",
            "assistant_prefix": "Assistant:",
            "system_prefix": "System:",
        }

    def validate_messages(self, messages: List[Dict[str, str]]) -> bool:
        """Validate messages for Anthropic template."""
        if not messages:
            return False

        valid_roles = {"system", "user", "assistant"}
        for message in messages:
            if "role" not in message or "content" not in message:
                return False
            if message["role"] not in valid_roles:
                return False

        return True


class GemmaTemplate(ChatTemplate):
    """Template for Google Gemma models."""

    def apply(
        self, messages: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """Apply Gemma template."""
        result = ""

        for message in messages:
            if message["role"] == "user":
                result += f"<start_of_turn>user\n{message['content']}<end_of_turn>\n"
            elif message["role"] == "assistant":
                result += f"<start_of_turn>model\n{message['content']}<end_of_turn>\n"
            elif message["role"] == "system":
                # Gemma typically handles system messages as part of user context
                result += f"<start_of_turn>user\n{message['content']}<end_of_turn>\n"

        if add_generation_prompt:
            result += "<start_of_turn>model\n"

        return result

    def get_special_tokens(self) -> Dict[str, str]:
        """Get Gemma special tokens."""
        return {
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "start_of_turn": "<start_of_turn>",
            "end_of_turn": "<end_of_turn>",
        }

    def validate_messages(self, messages: List[Dict[str, str]]) -> bool:
        """Validate messages for Gemma template."""
        if not messages:
            return False

        valid_roles = {"system", "user", "assistant"}
        for message in messages:
            if "role" not in message or "content" not in message:
                return False
            if message["role"] not in valid_roles:
                return False

        return True


class ChatTemplateManager:
    """Enhanced chat template manager with support for multiple model families."""

    # Model family patterns
    MODEL_PATTERNS = {
        "llama": [
            r"llama.*3",
            r"llama-3",
            r"meta-llama.*3",
        ],
        "llama2": [
            r"llama.*2",
            r"llama-2",
            r"meta-llama.*2",
            r"code.*llama",
        ],
        "mistral": [
            r"mistral",
            r"mixtral",
            r"mistralai",
        ],
        "openai": [
            r"gpt-3",
            r"gpt-4",
            r"text-davinci",
            r"openai",
        ],
        "chatml": [
            r"yi-",
            r"qwen",
            r"internlm",
            r"openchat",
        ],
        "anthropic": [
            r"claude",
            r"anthropic",
        ],
        "gemma": [
            r"gemma",
            r"google.*gemma",
        ],
    }

    def __init__(self):
        self.templates = {
            "llama": LlamaTemplate(version="3"),
            "llama2": LlamaTemplate(version="2"),
            "mistral": MistralTemplate(),
            "openai": OpenAITemplate(),
            "chatml": ChatMLTemplate(),
            "anthropic": AnthropicTemplate(),
            "gemma": GemmaTemplate(),
        }

    def detect_model_family(self, model_name: str) -> str:
        """Detect model family from model name."""
        model_name_lower = model_name.lower()

        for family, patterns in self.MODEL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, model_name_lower):
                    return family

        return "openai"  # Default fallback

    def get_template(self, model_name: str) -> ChatTemplate:
        """Get appropriate template for model."""
        family = self.detect_model_family(model_name)
        return self.templates.get(family, self.templates["openai"])

    def apply_template(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Apply appropriate template to messages."""
        template = self.get_template(model_name)

        if not template.validate_messages(messages):
            raise ValueError(f"Invalid messages for model {model_name}")

        return template.apply(messages, add_generation_prompt)

    def get_special_tokens(self, model_name: str) -> Dict[str, str]:
        """Get special tokens for model."""
        template = self.get_template(model_name)
        return template.get_special_tokens()

    def register_custom_template(self, family_name: str, template: ChatTemplate):
        """Register a custom template."""
        self.templates[family_name] = template

    def add_model_pattern(self, family_name: str, pattern: str):
        """Add a new model pattern for family detection."""
        if family_name not in self.MODEL_PATTERNS:
            self.MODEL_PATTERNS[family_name] = []
        self.MODEL_PATTERNS[family_name].append(pattern)

    def validate_template_compatibility(
        self, model_name: str, test_messages: Optional[List[Dict[str, str]]] = None
    ) -> bool:
        """Validate that template works with model."""
        if test_messages is None:
            test_messages = [{"role": "user", "content": "Hello, how are you?"}]

        try:
            template = self.get_template(model_name)
            if not template.validate_messages(test_messages):
                return False

            result = template.apply(test_messages, add_generation_prompt=True)
            return len(result) > 0

        except Exception:
            return False

    def get_model_info(self, model_name: str) -> ModelInfo:
        """Get detailed information about a model."""
        family = self.detect_model_family(model_name)
        template = self.templates[family]
        special_tokens = template.get_special_tokens()

        return ModelInfo(name=model_name, family=family, special_tokens=special_tokens)

    def list_supported_families(self) -> List[str]:
        """List all supported model families."""
        return list(self.templates.keys())

    def get_template_examples(self, family: str) -> Dict[str, str]:
        """Get example outputs for a template family."""
        if family not in self.templates:
            return {}

        template = self.templates[family]

        examples = {}

        # Simple conversation
        simple_messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        examples["simple_conversation"] = template.apply(simple_messages)

        # With system message
        system_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the capital of France?"},
        ]
        examples["with_system"] = template.apply(system_messages)

        return examples


# Global chat template manager instance
_chat_template_manager = ChatTemplateManager()


def get_template_for_model(model_name: str) -> str:
    """Get the appropriate chat template for a given model name.

    Args:
        model_name: The name or path of the model

    Returns:
        The chat template string for the model
    """
    template = _chat_template_manager.get_template(model_name)

    # Convert template object to Jinja2 template string
    if hasattr(template, "_apply_llama3") and "3" in model_name:
        # Llama 3 template
        return (
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
    elif hasattr(template, "_apply_llama2"):
        # Llama 2 template - simplified for Jinja2
        return (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<s>[INST] ' + message['content'] + ' [/INST]' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + ' </s>' }}"
            "{% endif %}"
            "{% endfor %}"
        )
    elif isinstance(template, MistralTemplate):
        # Mistral template
        return (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + '</s>' }}"
            "{% endif %}"
            "{% endfor %}"
        )
    else:
        # Generic fallback
        return (
            "{% for message in messages %}"
            "{{ message['role'].title() + ': ' + message['content'] + '\\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ 'Assistant: ' }}"
            "{% endif %}"
        )
