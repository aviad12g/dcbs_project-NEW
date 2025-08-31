"""
Tests for message processing functionality.
"""

import unittest
from pathlib import Path
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from messages_completion.processing import MessageProcessor, MessageBatch


class TestMessageBatch(unittest.TestCase):
    """Test message batch functionality."""
    
    def test_create_from_single_messages(self):
        """Test creating batch from single message sequence."""
        messages = [
            {"role": "user", "content": "Hello!"}
        ]
        
        batch = MessageBatch.from_single_messages(messages)
        
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch.batch_size, 1)
        self.assertEqual(batch[0], messages)
    
    def test_create_from_multiple_messages(self):
        """Test creating batch from multiple message sequences."""
        message_sequences = [
            [{"role": "user", "content": "Question 1"}],
            [{"role": "user", "content": "Question 2"}],
            [{"role": "user", "content": "Question 3"}]
        ]
        
        batch = MessageBatch.from_multiple_messages(message_sequences)
        
        self.assertEqual(len(batch), 3)
        self.assertEqual(batch.batch_size, 3)
        
        for i, messages in enumerate(batch):
            self.assertEqual(messages, message_sequences[i])
    
    def test_batch_validation(self):
        """Test batch validation."""
        # Empty message sequences
        with self.assertRaises(ValueError):
            MessageBatch([])
        
        # Empty individual sequence
        with self.assertRaises(ValueError):
            MessageBatch([[]])
        
        # Invalid message format
        with self.assertRaises(ValueError):
            MessageBatch([["not a dict"]])
        
        # Missing required keys
        with self.assertRaises(ValueError):
            MessageBatch([[{"role": "user"}]])  # Missing content
        
        with self.assertRaises(ValueError):
            MessageBatch([[{"content": "Hello"}]])  # Missing role
    
    def test_batch_iteration(self):
        """Test batch iteration."""
        message_sequences = [
            [{"role": "user", "content": "Q1"}],
            [{"role": "user", "content": "Q2"}]
        ]
        
        batch = MessageBatch.from_multiple_messages(message_sequences)
        
        # Test iteration
        iterated_sequences = list(batch)
        self.assertEqual(iterated_sequences, message_sequences)
        
        # Test indexing
        self.assertEqual(batch[0], message_sequences[0])
        self.assertEqual(batch[1], message_sequences[1])


class TestMessageProcessor(unittest.TestCase):
    """Test message processor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = MessageProcessor()
    
    def test_simple_format(self):
        """Test simple message formatting."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        formatted = self.processor.format_messages(messages)
        
        self.assertIn("System: You are helpful.", formatted)
        self.assertIn("User: Hello!", formatted)
        self.assertIn("Assistant: Hi there!", formatted)
        self.assertIn("User: How are you?", formatted)
        self.assertTrue(formatted.endswith("Assistant:"))
    
    def test_custom_template(self):
        """Test custom template formatting."""
        custom_template = "<|{role}|> {content} <|end|>\n"
        processor = MessageProcessor(custom_template=custom_template)
        
        messages = [
            {"role": "user", "content": "Hello!"}
        ]
        
        formatted = processor.format_messages(messages)
        
        self.assertIn("<|user|> Hello! <|end|>", formatted)
    
    def test_format_batch(self):
        """Test batch message formatting."""
        message_sequences = [
            [{"role": "user", "content": "Question 1"}],
            [{"role": "user", "content": "Question 2"}]
        ]
        
        batch = MessageBatch.from_multiple_messages(message_sequences)
        formatted_prompts = self.processor.format_batch(batch)
        
        self.assertEqual(len(formatted_prompts), 2)
        self.assertIn("Question 1", formatted_prompts[0])
        self.assertIn("Question 2", formatted_prompts[1])
    
    def test_message_validation(self):
        """Test message validation."""
        # Valid messages
        valid_messages = [
            {"role": "user", "content": "Hello!"}
        ]
        self.assertTrue(self.processor.validate_messages(valid_messages))
        
        # Empty messages
        with self.assertRaises(ValueError):
            self.processor.validate_messages([])
        
        # Invalid message type
        with self.assertRaises(ValueError):
            self.processor.validate_messages(["not a dict"])
        
        # Missing role
        with self.assertRaises(ValueError):
            self.processor.validate_messages([{"content": "Hello"}])
        
        # Missing content
        with self.assertRaises(ValueError):
            self.processor.validate_messages([{"role": "user"}])
        
        # Non-string content
        with self.assertRaises(ValueError):
            self.processor.validate_messages([{"role": "user", "content": 123}])
    
    def test_add_system_message(self):
        """Test adding system message."""
        messages = [
            {"role": "user", "content": "Hello!"}
        ]
        
        # Add system message
        updated = self.processor.add_system_message(messages, "You are helpful.")
        
        self.assertEqual(len(updated), 2)
        self.assertEqual(updated[0]["role"], "system")
        self.assertEqual(updated[0]["content"], "You are helpful.")
        self.assertEqual(updated[1], messages[0])
        
        # Update existing system message
        messages_with_system = [
            {"role": "system", "content": "Old system message"},
            {"role": "user", "content": "Hello!"}
        ]
        
        updated = self.processor.add_system_message(messages_with_system, "New system message")
        
        self.assertEqual(len(updated), 2)
        self.assertEqual(updated[0]["content"], "New system message")
    
    def test_conversation_length(self):
        """Test conversation length calculation."""
        messages = [
            {"role": "user", "content": "Hello!"},  # 6 chars
            {"role": "assistant", "content": "Hi!"}  # 3 chars
        ]
        
        length = self.processor.get_conversation_length(messages)
        self.assertEqual(length, 9)
    
    def test_truncate_conversation(self):
        """Test conversation truncation."""
        messages = [
            {"role": "system", "content": "System"},  # 6 chars
            {"role": "user", "content": "First"},     # 5 chars
            {"role": "assistant", "content": "OK"},   # 2 chars
            {"role": "user", "content": "Second"}     # 6 chars
        ]
        
        # Truncate to 15 chars (should keep system + last 2 messages)
        truncated = self.processor.truncate_conversation(messages, 15)
        
        self.assertEqual(len(truncated), 3)
        self.assertEqual(truncated[0]["role"], "system")
        self.assertEqual(truncated[1]["content"], "OK")
        self.assertEqual(truncated[2]["content"], "Second")
    
    def test_truncate_conversation_no_system(self):
        """Test conversation truncation without system message."""
        messages = [
            {"role": "user", "content": "First"},     # 5 chars
            {"role": "assistant", "content": "OK"},   # 2 chars
            {"role": "user", "content": "Second"}     # 6 chars
        ]
        
        # Truncate to 10 chars (should keep last 2 messages)
        truncated = self.processor.truncate_conversation(messages, 10)
        
        self.assertEqual(len(truncated), 2)
        self.assertEqual(truncated[0]["content"], "OK")
        self.assertEqual(truncated[1]["content"], "Second")
    
    def test_empty_format_messages(self):
        """Test formatting empty messages."""
        with self.assertRaises(ValueError):
            self.processor.format_messages([])
    
    def test_non_standard_roles(self):
        """Test handling of non-standard roles."""
        messages = [
            {"role": "custom_role", "content": "Custom message"}
        ]
        
        # Should not raise error but may log warning
        formatted = self.processor.format_messages(messages)
        self.assertIn("Custom_Role: Custom message", formatted)
    
    def test_empty_content_warning(self):
        """Test handling of empty content."""
        messages = [
            {"role": "user", "content": ""},  # Empty content
            {"role": "user", "content": "   "}  # Whitespace only
        ]
        
        # Should validate but may log warnings
        self.assertTrue(self.processor.validate_messages(messages))


if __name__ == "__main__":
    unittest.main()