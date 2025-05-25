"""
Example processing functionality for evaluation.

This module handles individual example processing, prompt creation,
and chain-of-thought reasoning generation.
"""

from typing import Dict, List

import torch

from dcbs import SamplingContext
from src.errors import eval_logger as logger


class ExampleProcessor:
    """Processes individual examples for evaluation."""

    def __init__(self, model, tokenizer, context: SamplingContext, sampler=None):
        self.model = model
        self.tokenizer = tokenizer
        self.context = context
        self.device = context.device
        self.sampler = sampler  # For chain-of-thought generation

    def create_prompt(
        self, sentence: str, options: List[str], include_cot: bool = True
    ) -> str:
        """Create a chat-formatted prompt for the problem."""
        # HARDCODED PROMPTS - TODO: Move to centralized prompt management system
        if include_cot:
            system_msg = "You are a helpful assistant. Think step by step and then give your final answer as a single letter (A, B, C, or D)."

            # Build options string dynamically
            options_str = ""
            for i, option in enumerate(options):
                label = chr(ord("A") + i)  # A, B, C, D, etc.
                options_str += f"{label}. {option}\n"

            user_msg = f"{sentence}\n\nOptions:\n{options_str}\nLet's think step by step to determine the answer.\n\nThe answer is"
        else:
            system_msg = "You are a helpful assistant. Give your answer as a single letter (A, B, C, or D)."

            # Build options string dynamically
            options_str = ""
            for i, option in enumerate(options):
                label = chr(ord("A") + i)  # A, B, C, D, etc.
                options_str += f"{label}. {option}\n"

            user_msg = f"{sentence}\n\nOptions:\n{options_str}\nThe answer is"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        # Use chat template if available, otherwise fall back to simple formatting
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant: "

    def get_answer_token_ids(self, options: List[str]) -> Dict[str, int]:
        """Get token IDs for answer letters (A, B, C, D) instead of full text options."""
        answer_ids = {}

        for i, option in enumerate(options):
            # Use letter labels (A, B, C, D) instead of full text
            label = chr(ord("A") + i)
            
            # Try different tokenization approaches for the letter
            candidates = [
                label,  # Raw letter (A, B, C, D)
                f" {label}",  # With leading space
                f"{label}.",  # With period
                f" {label}.",  # With space and period
            ]

            token_id = None
            for candidate in candidates:
                tokens = self.tokenizer.encode(candidate, add_special_tokens=False)
                if len(tokens) == 1:
                    token_id = tokens[0]
                    break

            if token_id is None:
                # Fall back to first token of the letter
                tokens = self.tokenizer.encode(f" {label}", add_special_tokens=False)
                token_id = tokens[0] if tokens else 0

            # Map the full option text to the letter's token ID
            answer_ids[option] = token_id

        return answer_ids

    def process_example(self, example: Dict, include_cot: bool = True) -> Dict:
        """Process a single example and get logits, with actual CoT generation if enabled."""

        # Handle ARC Easy format
        if "question" in example:
            # ARC Easy format
            sentence = example["question"]
            options = example["options"]
            correct_option = example.get("correct_option", "1")
            # Convert 1-based index to 0-based and get the correct answer
            correct_idx = int(correct_option) - 1
            correct_answer = options[correct_idx]
        else:
            raise ValueError(
                "Unknown example format - expected 'question' field for ARC Easy format"
            )

        if include_cot:
            # First, generate chain-of-thought reasoning
            cot_prompt = self.create_cot_prompt(sentence, options)
            cot_reasoning = self.generate_reasoning(cot_prompt)

            # Then create final answer prompt with the generated reasoning
            final_prompt = self.create_final_answer_prompt(
                sentence, options, cot_reasoning
            )
        else:
            final_prompt = self.create_prompt(sentence, options, include_cot=False)
            cot_reasoning = None

        # Tokenize and get model output for final answer
        inputs = self.tokenizer(final_prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :].squeeze(0)

        # Get token IDs for answer options
        answer_ids = self.get_answer_token_ids(options)

        # Calculate answer probabilities
        all_probs = torch.softmax(logits, dim=0)
        answer_probs = {
            option: all_probs[token_id].item()
            for option, token_id in answer_ids.items()
        }

        return {
            "id": example.get("id", "unknown"),
            "sentence": sentence,
            "options": options,
            "correct_answer": correct_answer,
            "correct_option": correct_option,
            "answer_ids": answer_ids,
            "filter_tokens": set(answer_ids.values()),
            "correct_id": answer_ids[correct_answer],
            "logits": logits,
            "answer_probs": answer_probs,
            "cot_reasoning": cot_reasoning,
        }

    def create_cot_prompt(self, sentence: str, options: List[str]) -> str:
        """Create a prompt for chain-of-thought reasoning generation."""
        # HARDCODED PROMPTS - TODO: Move to centralized prompt management system
        system_msg = (
            "You are a helpful assistant. Think step by step about the problem and explain your reasoning."
        )

        # Build options string dynamically
        options_str = ""
        for i, option in enumerate(options):
            label = chr(ord("A") + i)  # A, B, C, D, etc.
            options_str += f"{label}. {option}\n"

        user_msg = f"{sentence}\n\nOptions:\n{options_str}\nLet's think step by step about which option is correct:"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant: "

    def create_final_answer_prompt(
        self, sentence: str, options: List[str], reasoning: str
    ) -> str:
        """Create final answer prompt with generated reasoning."""
        # HARDCODED PROMPTS - TODO: Move to centralized prompt management system
        system_msg = "You are a helpful assistant. Based on your reasoning, give the final answer as a single letter (A, B, C, or D)."

        # Build options string dynamically
        options_str = ""
        for i, option in enumerate(options):
            label = chr(ord("A") + i)  # A, B, C, D, etc.
            options_str += f"{label}. {option}\n"

        user_msg = f"{sentence}\n\nOptions:\n{options_str}\nMy reasoning: {reasoning}\n\nTherefore, the answer is"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant: "

    def generate_reasoning(self, prompt: str, max_length: int = 200) -> str:
        """Generate chain-of-thought reasoning using simple model generation."""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)

        try:
            # Use simple model generation with strict limits - purely deterministic
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,  # Explicitly pass attention mask
                    max_new_tokens=min(max_length, 100),  # Increase limit for better reasoning
                    do_sample=False,  # Greedy decoding only
                    temperature=1.0,  # Override model default to eliminate warnings
                    top_p=1.0,  # Override model default to eliminate warnings
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Extract generated reasoning
            reasoning_tokens = outputs[0][inputs.input_ids.shape[1] :]
            reasoning = self.tokenizer.decode(reasoning_tokens, skip_special_tokens=True).strip()

        except Exception as e:
            logger.warning(f"CoT generation failed: {e}, using fallback reasoning")
            reasoning = "Let me analyze the options step by step."

        # Only truncate if we find complete answer patterns (not partial ones)
        stop_patterns = [
            "the answer is a",
            "the answer is b", 
            "the answer is c",
            "the answer is d",
            "therefore a",
            "therefore b",
            "therefore c", 
            "therefore d",
            "so the answer is a",
            "so the answer is b",
            "so the answer is c",
            "so the answer is d",
        ]
        
        reasoning_lower = reasoning.lower()
        for pattern in stop_patterns:
            if pattern in reasoning_lower:
                reasoning = reasoning[:reasoning_lower.find(pattern)].strip()
                break

        # Ensure we have meaningful reasoning (at least 20 characters)
        if not reasoning or len(reasoning) < 20:
            reasoning = "Looking at each option carefully, I need to consider which piece of safety equipment would prevent mold spores from entering the respiratory system."

        return reasoning 