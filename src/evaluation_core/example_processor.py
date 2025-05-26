"""
Example processing functionality for evaluation.

This module handles individual example processing, prompt creation,
and chain-of-thought reasoning generation.
"""

from typing import Dict, List

import torch

from dcbs import DCBSSampler, SamplingContext
from src.errors import eval_logger as logger
from src.evaluation_core.template_manager import ChatTemplateManager


class PromptManager:
    """Centralized prompt management system."""
    
    @staticmethod
    def get_cot_system_message() -> str:
        """Get system message for chain-of-thought reasoning."""
        return "You are a helpful assistant. Think step by step about the problem and explain your reasoning."
    
    @staticmethod
    def get_answer_system_message() -> str:
        """Get system message for final answer extraction."""
        return "You are a helpful assistant. Based on your reasoning, give the final answer as a single letter (A, B, C, or D)."
    
    @staticmethod
    def get_direct_system_message() -> str:
        """Get system message for direct answer extraction."""
        return "You are a helpful assistant. Give your answer as a single letter (A, B, C, or D)."
    
    @staticmethod
    def format_options(options: List[str]) -> str:
        """Format options as a string with letter labels."""
        options_str = ""
        for i, option in enumerate(options):
            label = chr(ord("A") + i)  # A, B, C, D, etc.
            options_str += f"{label}. {option}\n"
        return options_str
    
    @staticmethod
    def create_cot_messages(sentence: str, options: List[str]) -> List[Dict[str, str]]:
        """Create messages for chain-of-thought reasoning."""
        system_msg = PromptManager.get_cot_system_message()
        options_str = PromptManager.format_options(options)
        user_msg = f"{sentence}\n\nOptions:\n{options_str}\nLet's think step by step about which option is correct:"
        
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
    
    @staticmethod
    def create_final_answer_messages(sentence: str, options: List[str], reasoning: str) -> List[Dict[str, str]]:
        """Create messages for final answer extraction."""
        system_msg = PromptManager.get_answer_system_message()
        options_str = PromptManager.format_options(options)
        user_msg = f"{sentence}\n\nOptions:\n{options_str}\nMy reasoning: {reasoning}\n\nTherefore, the answer is"
        
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
    
    @staticmethod
    def create_direct_answer_messages(sentence: str, options: List[str]) -> List[Dict[str, str]]:
        """Create messages for direct answer extraction."""
        system_msg = PromptManager.get_direct_system_message()
        options_str = PromptManager.format_options(options)
        user_msg = f"{sentence}\n\nOptions:\n{options_str}\nThe answer is"
        
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]


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
        if include_cot:
            messages = PromptManager.create_cot_messages(sentence, options)
        else:
            messages = PromptManager.create_direct_answer_messages(sentence, options)

        # Use chat template if available, otherwise fall back to simple formatting
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}")
            system_msg = messages[0]["content"]
            user_msg = messages[1]["content"]
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
            # First, generate chain-of-thought reasoning using the provided sampler
            cot_messages = PromptManager.create_cot_messages(sentence, options)
            cot_prompt = self.tokenizer.apply_chat_template(
                cot_messages, tokenize=False, add_generation_prompt=True
            )
            cot_reasoning = self.generate_reasoning(cot_prompt, sampler=self.sampler)

            # Then create final answer prompt with the generated reasoning
            final_messages = PromptManager.create_final_answer_messages(
                sentence, options, cot_reasoning
            )
            final_prompt = self.tokenizer.apply_chat_template(
                final_messages, tokenize=False, add_generation_prompt=True
            )
        else:
            direct_messages = PromptManager.create_direct_answer_messages(sentence, options)
            final_prompt = self.tokenizer.apply_chat_template(
                direct_messages, tokenize=False, add_generation_prompt=True
            )
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
        messages = PromptManager.create_cot_messages(sentence, options)
        
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}")
            system_msg = messages[0]["content"]
            user_msg = messages[1]["content"]
            return f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant: "

    def create_final_answer_prompt(
        self, sentence: str, options: List[str], reasoning: str
    ) -> str:
        """Create final answer prompt with generated reasoning."""
        messages = PromptManager.create_final_answer_messages(sentence, options, reasoning)
        
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}")
            system_msg = messages[0]["content"]
            user_msg = messages[1]["content"]
            return f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant: "

    def generate_reasoning(self, prompt: str, max_length: int = 200, sampler=None) -> str:
        """Generate chain-of-thought reasoning using token-by-token generation with the provided sampler.
        
        Args:
            prompt: The prompt to generate reasoning from
            max_length: Maximum length of generated reasoning
            sampler: Sampler to use for generation (the same sampler being evaluated)
            
        Returns:
            Generated reasoning text
        """
        if sampler is None:
            raise ValueError("A sampler must be provided for CoT generation")
            
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        input_length = inputs.input_ids.shape[1]
        generated_tokens = inputs.input_ids[0].tolist()
        
        # Generate tokens one by one using the provided sampler
        for _ in range(max_length):
            # Get current input as tensor
            current_input = torch.tensor([generated_tokens], device=self.device)
            
            # Forward pass to get next token logits
            with torch.no_grad():
                outputs = self.model(current_input)
                next_token_logits = outputs.logits[0, -1, :]
            
            # Use the provided sampler to select the next token
            context = SamplingContext(
                embedding_layer=self.model.get_input_embeddings(),
                device=self.device
            )
            next_token = sampler.sample(next_token_logits, context=context)
            
            # Add token to generated sequence
            generated_tokens.append(next_token)
            
            # Stop if we reach the EOS token
            if next_token == self.tokenizer.eos_token_id:
                break
                
        # Decode the generated tokens
        reasoning = self.tokenizer.decode(generated_tokens[input_length:], skip_special_tokens=True).strip()
        
        # Clean up reasoning (remove premature answers)
        stop_patterns = [
            "the answer is a", "the answer is b", "the answer is c", "the answer is d",
            "therefore a", "therefore b", "therefore c", "therefore d",
        ]
        
        reasoning_lower = reasoning.lower()
        for pattern in stop_patterns:
            if pattern in reasoning_lower:
                reasoning = reasoning[:reasoning_lower.find(pattern)].strip()
                break

        # Ensure meaningful reasoning
        if not reasoning or len(reasoning) < 20:
            reasoning = "Looking at each option carefully, I need to consider which is most supported by the evidence."

        return reasoning 