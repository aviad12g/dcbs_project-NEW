"""
Token processing utilities for DCBS sampling.

Utilities for tokenized text including filtering, validation, and token probability manipulation.
"""

import logging
import random
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch


class TokenizerCache:
    """Memory-efficient tokenizer cache with LRU eviction.

    Provides caching to reduce redundant tokenization operations.
    """

    def __init__(self, max_size: int = 1000):
        """Initialize tokenizer cache.

        Args:
            max_size: Max cache entries
        """
        self.encode_cache: OrderedDict[str, List[int]] = OrderedDict()
        self.decode_cache: OrderedDict[Tuple[int, ...], str] = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.last_report_time = time.time()
        self.report_interval = 60  # Report stats every minute

    def encode(self, text: str, tokenizer: Any, **kwargs) -> List[int]:
        """Get token IDs with caching.

        Args:
            text: Text to encode
            tokenizer: Tokenizer for cache misses
            **kwargs: Tokenizer arguments

        Returns:
            Token IDs
        """
        # Create a cache key from the text and relevant kwargs
        key = (text, tuple(sorted(kwargs.items())))

        if key in self.encode_cache:
            self.hits += 1
            # Move item to end (most recently used position)
            self.encode_cache.move_to_end(key)
            return self.encode_cache[key]

        # Cache miss
        self.misses += 1
        tokens = tokenizer.encode(text, **kwargs)

        # Add to cache
        self.encode_cache[key] = tokens

        # Enforce cache size limit
        if len(self.encode_cache) > self.max_size:
            self.encode_cache.popitem(last=False)  # Remove least recently used

        self._maybe_report_stats()
        return tokens

    def decode(self, token_ids: List[int], tokenizer: Any, **kwargs) -> str:
        """Decode token IDs with caching.

        Args:
            token_ids: Token IDs to decode
            tokenizer: Tokenizer for cache misses
            **kwargs: Tokenizer arguments

        Returns:
            Decoded text
        """
        # Convert token_ids to hashable tuple for cache key
        key = (tuple(token_ids), tuple(sorted(kwargs.items())))

        if key in self.decode_cache:
            self.hits += 1
            # Move item to end (most recently used position)
            self.decode_cache.move_to_end(key)
            return self.decode_cache[key]

        # Cache miss
        self.misses += 1
        text = tokenizer.decode(token_ids, **kwargs)

        # Add to cache
        self.decode_cache[key] = text

        # Enforce cache size limit
        if len(self.decode_cache) > self.max_size:
            self.decode_cache.popitem(last=False)  # Remove least recently used

        self._maybe_report_stats()
        return text

    def clear(self) -> None:
        """Clear the cache."""
        self.encode_cache.clear()
        self.decode_cache.clear()

    def _maybe_report_stats(self) -> None:
        """Log cache stats periodically."""
        current_time = time.time()
        if current_time - self.last_report_time > self.report_interval:
            total = self.hits + self.misses
            hit_rate = (self.hits / total) * 100 if total > 0 else 0

            logging.getLogger("dcbs.algorithm").debug(
                f"TokenizerCache stats: {self.hits} hits, {self.misses} misses, "
                f"{hit_rate:.2f}% hit rate, encode cache: {len(self.encode_cache)}, "
                f"decode cache: {len(self.decode_cache)}"
            )

            self.last_report_time = current_time

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Stats dictionary
        """
        total = self.hits + self.misses
        hit_rate = (self.hits / total) * 100 if total > 0 else 0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "encode_cache_size": len(self.encode_cache),
            "decode_cache_size": len(self.decode_cache),
            "max_size": self.max_size,
        }


# Global tokenizer cache instance
tokenizer_cache = TokenizerCache(max_size=5000)


class AnswerTokenResolver:
    """Resolves answer choice letters to token IDs with fallback strategies."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.logger = logging.getLogger("dcbs.algorithm")
        
    def get_answer_token_ids(self, options: List[str]) -> Dict[str, int]:
        """
        Get token IDs for answer letters with robust fallback strategies.
        
        Args:
            options: List of answer options
            
        Returns:
            Dictionary mapping options to their token IDs
        """
        answer_ids = {}
        
        for i, option in enumerate(options):
            label = chr(ord("A") + i)
            token_id = self._resolve_single_token_id(label)
            answer_ids[option] = token_id
            
            # Verify the resolution
            decoded = tokenizer_cache.decode([token_id], self.tokenizer)
            self.logger.debug(f"Option {label} ({option}): token_id={token_id}, decoded='{decoded}'")
        
        # Check for conflicts
        self._check_for_duplicates(answer_ids)
        
        return answer_ids
    
    def _resolve_single_token_id(self, label: str) -> int:
        """Resolve a single answer label to its best token ID."""
        # Candidate tokenization strategies in order of preference
        candidates = [
            f" {label}",      # Space + letter (most common)
            label,            # Raw letter
            f"{label}.",      # Letter with period
            f" {label}.",     # Space + letter + period
        ]
        
        for candidate in candidates:
            tokens = tokenizer_cache.encode(candidate, self.tokenizer, add_special_tokens=False)
            
            if len(tokens) == 1:
                # Perfect single token match
                self.logger.debug(f"Found single token {tokens[0]} for '{candidate}'")
                return tokens[0]
        
        # Fallback: use last token from space + letter
        fallback_tokens = tokenizer_cache.encode(f" {label}", self.tokenizer, add_special_tokens=False)
        if fallback_tokens:
            token_id = fallback_tokens[-1]
            self.logger.debug(f"Using fallback token {token_id} for label {label}")
            return token_id
        
        # Ultimate fallback: ASCII value
        self.logger.warning(f"Using ASCII fallback for label {label}")
        return ord(label)
    
    def _check_for_duplicates(self, answer_ids: Dict[str, int]) -> None:
        """Check for and warn about duplicate token IDs."""
        token_counts = {}
        for option, token_id in answer_ids.items():
            if token_id in token_counts:
                token_counts[token_id].append(option)
            else:
                token_counts[token_id] = [option]
        
        duplicates = {tid: opts for tid, opts in token_counts.items() if len(opts) > 1}
        if duplicates:
            self.logger.warning(f"Found duplicate token IDs: {duplicates}")
            self.logger.warning("This may affect evaluation accuracy!")


def is_valid_token_prediction(
    pred_id: int, correct_id: Union[int, List[int]], correct_answer: str, tokenizer
) -> bool:
    """Checks if predicted token is correct.

    Args:
        pred_id: Predicted token ID
        correct_id: Correct token ID or list for multi-token
        correct_answer: String representation of answer
        tokenizer: Tokenizer for text conversion

    Returns:
        True if prediction is correct

    Note:
        Uses case-insensitive matching and supports multi-token answers.
    """
    logger = logging.getLogger("dcbs.algorithm")

    # Handle multi-token correct_id case
    if isinstance(correct_id, list):
        # Direct token ID match with any token in the correct answer
        if pred_id in correct_id:
            return True

        # Get the full concatenated representation of the correct answer using cache
        full_correct_text = (
            tokenizer_cache.decode(correct_id, tokenizer).lower().strip()
        )
        pred_str = tokenizer_cache.decode([pred_id], tokenizer).lower().strip()

        # Log detailed information about the token comparison
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Multi-token comparison: pred='{pred_str}' ({pred_id}) vs "
                f"full='{full_correct_text}' ({correct_id})"
            )

        # Check if the prediction is a reasonable match for the full answer
        if pred_str == full_correct_text:
            return True

        # Check if the prediction matches the normalized correct answer
        normalized_correct = correct_answer.lower().strip()
        if pred_str == normalized_correct:
            return True

        return False

    # Single token case - direct token ID match is always correct
    if pred_id == correct_id:
        return True

    # Otherwise, check if the decoded string matches after normalization
    # Ensure consistent normalization by lowercasing and stripping whitespace
    pred_str = tokenizer_cache.decode([pred_id], tokenizer).lower().strip()
    normalized_correct = correct_answer.lower().strip()

    # Log detailed information about the token comparison
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"Token comparison: pred='{pred_str}' ({pred_id}) vs "
            f"correct='{normalized_correct}' ({correct_id})"
        )

    return pred_str == normalized_correct


def filter_logits(
    logits: torch.Tensor, filter_tokens: Optional[Set[int]] = None
) -> torch.Tensor:
    """Filter logits to specified tokens.

    Args:
        logits: Original token logits
        filter_tokens: Allowed token IDs

    Returns:
        Filtered logits
    """
    if filter_tokens is None:
        return logits

    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[list(filter_tokens)] = False
    filtered = logits.clone()
    filtered[mask] = float("-inf")
    return filtered


def get_top_tokens(
    logits: torch.Tensor, top_n: int, force_include_ids: Optional[Set[int]] = None
) -> List[int]:
    """Get top-n token IDs with forced inclusions.

    Args:
        logits: Token logits
        top_n: Number of tokens to select
        force_include_ids: IDs to always include

    Returns:
        Selected token IDs
    """
    sorted_indices = torch.argsort(logits, descending=True)
    top_indices = set(sorted_indices[:top_n].cpu().tolist())

    if force_include_ids:
        for idx in force_include_ids:
            if idx not in top_indices:
                top_indices.add(idx)

    return list(top_indices)


def sample_token_from_logits(
    logits: torch.Tensor,
    temperature: float = None,  # Use None as default to indicate using global config
    filter_tokens: Optional[Set[int]] = None,
) -> int:
    """Sample token with temperature.

    Args:
        logits: Token logits
        temperature: Sampling temperature or None to use default
        filter_tokens: Allowed token IDs

    Returns:
        Sampled token ID
    """
    # Default to 1.0 if not specified (should be taken from config in production use)
    if temperature is None:
        temperature = 1.0

    if temperature != 1.0:
        logits = logits / temperature

    if filter_tokens:
        filtered_logits = filter_logits(logits, filter_tokens)
    else:
        filtered_logits = logits

    # Handle extreme cases
    if torch.isinf(filtered_logits).all() or torch.isnan(filtered_logits).any():
        if filter_tokens:
            return random.choice(list(filter_tokens))
        else:
            return random.randint(0, len(logits) - 1)

    probs = torch.softmax(filtered_logits, dim=0)
    return torch.multinomial(probs, 1).item()


def get_answer_token_ids(
    text: str, tokenizer, add_leading_space: bool = True
) -> List[int]:
    """Get token IDs for an answer option.

    Args:
        text: Answer text
        tokenizer: Tokenizer for conversion
        add_leading_space: Whether to add leading space

    Returns:
        Token IDs
    """
    text_to_encode = f" {text}" if add_leading_space else text
    return tokenizer_cache.encode(text_to_encode, tokenizer, add_special_tokens=False)
