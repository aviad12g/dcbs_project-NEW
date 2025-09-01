"""
Message processing utilities for the completion module.

This module formats already-prepared message sequences into a prompt string.
It does not depend on external chat template managers and does not add
messages on its own; callers must prepare the desired messages beforehand.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Any
import logging

from .types import MessageBatch

logger = logging.getLogger(__name__)



class MessageProcessor:
    """Processes messages using simple, self-contained formatting."""
    
    def __init__(self, model_name: Optional[str] = None, custom_template: Optional[str] = None):
        """
        Initialize message processor.
        
        Args:
            model_name: Name of the model for template selection
            custom_template: Custom chat template string
        """
        self.model_name = model_name
        self.custom_template = custom_template
        
        # No external chat template integration inside messages_completion.
        # Formatting is intentionally lightweight and self-contained.
        self.template_manager = None
    
    def format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages using appropriate chat template.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            Formatted prompt string
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        # Validate message format
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"Message at index {i} must be a dictionary")
            if "role" not in message or "content" not in message:
                raise ValueError(f"Message at index {i} must have 'role' and 'content' keys")
        
        # Use custom template if provided
        if self.custom_template:
            return self._apply_custom_template(messages)
        
        # Simple formatting (no external template manager usage)
        return self._simple_format(messages)
    
    def format_batch(self, message_batch: MessageBatch) -> List[str]:
        """
        Format a batch of message sequences.
        
        Args:
            message_batch: Batch of message sequences
            
        Returns:
            List of formatted prompt strings
        """
        formatted_prompts = []
        
        for messages in message_batch:
            try:
                formatted_prompt = self.format_messages(messages)
                formatted_prompts.append(formatted_prompt)
            except Exception as e:
                logger.error(f"Failed to format messages: {e}")
                # Use fallback formatting for this sequence
                formatted_prompts.append(self._simple_format(messages))
        
        return formatted_prompts
    
    def _apply_custom_template(self, messages: List[Dict[str, str]]) -> str:
        """Apply custom template to messages."""
        # Simple template variable substitution
        formatted_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            # Replace template variables
            template_part = self.custom_template
            template_part = template_part.replace("{role}", role)
            template_part = template_part.replace("{content}", content)
            
            formatted_parts.append(template_part)
        
        return "".join(formatted_parts)
    
    def _simple_format(self, messages: List[Dict[str, str]]) -> str:
        """Simple fallback message formatting."""
        formatted_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted_parts.append(f"System: {content}\n")
            elif role == "user":
                formatted_parts.append(f"User: {content}\n")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}\n")
            else:
                formatted_parts.append(f"{role.title()}: {content}\n")
        
        return "".join(formatted_parts)
    
    def validate_messages(self, messages: List[Dict[str, str]]) -> bool:
        """
        Validate message format and content.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        valid_roles = {"system", "user", "assistant"}
        
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"Message at index {i} must be a dictionary")
            
            if "role" not in message:
                raise ValueError(f"Message at index {i} missing 'role' key")
            
            if "content" not in message:
                raise ValueError(f"Message at index {i} missing 'content' key")
            
            if message["role"] not in valid_roles:
                logger.warning(f"Message at index {i} has non-standard role: {message['role']}")
            
            if not isinstance(message["content"], str):
                raise ValueError(f"Message content at index {i} must be a string")
            
            if not message["content"].strip():
                logger.warning(f"Message at index {i} has empty content")
        
        return True
    
    def add_system_message(self, messages: List[Dict[str, str]], system_content: str) -> List[Dict[str, str]]:
        """
        Add or update system message at the beginning of the conversation.
        
        Args:
            messages: Existing message list
            system_content: System message content
            
        Returns:
            Updated message list with system message
        """
        messages_copy = messages.copy()
        
        # Check if first message is already a system message
        if messages_copy and messages_copy[0]["role"] == "system":
            # Update existing system message
            messages_copy[0]["content"] = system_content
        else:
            # Add new system message at the beginning
            system_message = {"role": "system", "content": system_content}
            messages_copy.insert(0, system_message)
        
        return messages_copy
    
    def get_conversation_length(self, messages: List[Dict[str, str]]) -> int:
        """Get the total character length of the conversation."""
        return sum(len(message["content"]) for message in messages)
    
    def truncate_conversation(self, messages: List[Dict[str, str]], max_length: int) -> List[Dict[str, str]]:
        """
        Truncate conversation to fit within max_length characters.
        
        Preserves system message and recent messages.
        """
        if self.get_conversation_length(messages) <= max_length:
            return messages
        
        # Always keep system message if present
        result = []
        remaining_length = max_length
        
        if messages and messages[0]["role"] == "system":
            result.append(messages[0])
            remaining_length -= len(messages[0]["content"])
            messages = messages[1:]
        
        # Add messages from the end until we hit the limit
        for message in reversed(messages):
            message_length = len(message["content"])
            if message_length <= remaining_length:
                result.insert(-1 if result and result[0]["role"] == "system" else 0, message)
                remaining_length -= message_length
            else:
                break
        
        return result
