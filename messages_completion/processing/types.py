"""Message processing types."""

from typing import List, Dict


class MessageBatch:
    """
    Represents a batch of message sequences for processing.
    
    This class encapsulates multiple conversation sequences and provides
    validation and utility methods for batch processing.
    """
    
    def __init__(self, message_sequences: List[List[Dict[str, str]]]):
        """
        Initialize message batch.
        
        Args:
            message_sequences: List of message sequences (conversations)
        """
        if not message_sequences:
            raise ValueError("message_sequences cannot be empty")
        
        # Validate each message sequence
        for i, messages in enumerate(message_sequences):
            if not messages:
                raise ValueError(f"Message sequence {i} cannot be empty")
            
            # Validate message format
            for j, message in enumerate(messages):
                if not isinstance(message, dict):
                    raise ValueError(f"Message {j} in sequence {i} must be a dict")
                if "role" not in message or "content" not in message:
                    raise ValueError(f"Message {j} in sequence {i} must have 'role' and 'content' keys")
        
        self.message_sequences = message_sequences
    
    @classmethod
    def from_single_messages(cls, messages: List[Dict[str, str]]) -> "MessageBatch":
        """Create a batch from a single message sequence."""
        return cls([messages])
    
    @classmethod
    def from_multiple_messages(cls, message_sequences: List[List[Dict[str, str]]]) -> "MessageBatch":
        """Create a batch from multiple message sequences."""
        return cls(message_sequences)
    
    def __len__(self) -> int:
        """Return the number of message sequences in the batch."""
        return len(self.message_sequences)
    
    def __getitem__(self, index: int) -> List[Dict[str, str]]:
        """Get a message sequence by index."""
        return self.message_sequences[index]
    
    def __iter__(self):
        """Iterate over message sequences."""
        return iter(self.message_sequences)