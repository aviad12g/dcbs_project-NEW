"""
Disagreement tracking for DCBS vs Greedy sampling.

This module integrates with the existing evaluation system to track
when DCBS and greedy sampling choose different tokens.
"""

import time
from typing import Dict, List, Optional, Set, Tuple

import torch

from src.dcbs import GreedySampler, DCBSSampler, SamplingContext
from src.logger import DisagreementLogger


class DisagreementTracker:
    """Tracks disagreements between DCBS and greedy sampling during evaluation."""
    
    def __init__(
        self, 
        logger: DisagreementLogger,
        dcbs_sampler: DCBSSampler,
        greedy_sampler: GreedySampler,
        tokenizer,
        context: SamplingContext,
    ):
        """
        Initialize disagreement tracker.
        
        Args:
            logger: DisagreementLogger instance
            dcbs_sampler: DCBS sampler instance
            greedy_sampler: Greedy sampler instance  
            tokenizer: Tokenizer for text conversion
            context: Sampling context
        """
        self.logger = logger
        self.dcbs_sampler = dcbs_sampler
        self.greedy_sampler = greedy_sampler
        self.tokenizer = tokenizer
        self.context = context
        
        # Track current sequence
        self.current_sequence_id: Optional[str] = None
        self.current_disagreement_count = 0
        self.current_timestep = 0
    
    def start_sequence(self, sequence_id: str) -> None:
        """Start tracking a new sequence."""
        self.current_sequence_id = sequence_id
        self.current_disagreement_count = 0
        self.current_timestep = 0
    
    def check_for_disagreement(
        self,
        logits: torch.Tensor,
        filter_tokens: Optional[Set[int]] = None,
        context_tokens: Optional[List[int]] = None,
        context_text: Optional[str] = None,
    ) -> Tuple[int, int, bool]:
        """
        Check if DCBS and greedy sampling disagree on token choice.
        
        Args:
            logits: Token logits from model
            filter_tokens: Optional set of allowed tokens
            context_tokens: Full context as token IDs
            context_text: Human-readable context
            
        Returns:
            Tuple of (greedy_token, dcbs_token, disagreement_occurred)
        """
        if self.current_sequence_id is None:
            raise ValueError("Must call start_sequence() before checking disagreements")
        
        # Get greedy choice
        greedy_token = self.greedy_sampler.sample(logits, filter_tokens, self.context)
        
        # Get DCBS choice
        dcbs_token = self.dcbs_sampler.sample(logits, filter_tokens, self.context)
        
        # Check for disagreement
        disagreement_occurred = greedy_token != dcbs_token
        
        if disagreement_occurred:
            self._log_disagreement(
                logits=logits,
                filter_tokens=filter_tokens,
                context_tokens=context_tokens or [],
                context_text=context_text or "",
                greedy_token=greedy_token,
                dcbs_token=dcbs_token,
            )
            self.current_disagreement_count += 1
        
        self.current_timestep += 1
        return greedy_token, dcbs_token, disagreement_occurred
    
    def _log_disagreement(
        self,
        logits: torch.Tensor,
        filter_tokens: Optional[Set[int]],
        context_tokens: List[int],
        context_text: str,
        greedy_token: int,
        dcbs_token: int,
    ) -> None:
        """Log the disagreement details."""
        # Calculate probabilities
        probs = torch.softmax(logits, dim=-1)
        greedy_prob = probs[greedy_token].item()
        dcbs_prob = probs[dcbs_token].item()
        
        # Get top-k probabilities for context
        top_k_values, top_k_indices = torch.topk(probs, k=min(10, len(probs)))
        top_k_probs = {
            idx.item(): prob.item() 
            for idx, prob in zip(top_k_indices, top_k_values)
        }
        
        # Try to get cluster information from DCBS sampler
        cluster_id = None
        cluster_centroid = None
        
        # Get debug info if available
        if hasattr(self.dcbs_sampler, 'get_debug_stats'):
            debug_stats = self.dcbs_sampler.get_debug_stats()
            # Extract cluster info from debug stats if available
        
        self.logger.log_disagreement(
            sequence_id=self.current_sequence_id,
            timestep=self.current_timestep,
            context_tokens=context_tokens,
            context_text=context_text,
            greedy_token=greedy_token,
            greedy_prob=greedy_prob,
            dcbs_token=dcbs_token,
            dcbs_prob=dcbs_prob,
            cluster_id=cluster_id,
            cluster_centroid=cluster_centroid,
            top_k_probs=top_k_probs,
            metadata={
                "filter_tokens_count": len(filter_tokens) if filter_tokens else None,
                "total_vocab_size": len(logits),
            }
        )
    
    def end_sequence(
        self,
        greedy_answer: str,
        dcbs_answer: str,
        correct_answer: str,
        greedy_correct: bool,
        dcbs_correct: bool,
        dataset: Optional[str] = None,
        question: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Log the end of sequence with final comparison."""
        if self.current_sequence_id is None:
            raise ValueError("No active sequence to end")
        
        self.logger.log_sequence_end(
            sequence_id=self.current_sequence_id,
            greedy_answer=greedy_answer,
            dcbs_answer=dcbs_answer,
            correct_answer=correct_answer,
            greedy_correct=greedy_correct,
            dcbs_correct=dcbs_correct,
            total_disagreements=self.current_disagreement_count,
            dataset=dataset,
            question=question,
            metadata=metadata,
        )
        
        # Reset for next sequence
        self.current_sequence_id = None
        self.current_disagreement_count = 0
        self.current_timestep = 0


class DisagreementAwareQuestionAnswerer:
    """Question answerer that tracks disagreements between samplers."""
    
    def __init__(
        self,
        model,
        tokenizer,
        context: SamplingContext,
        disagreement_tracker: DisagreementTracker,
    ):
        """
        Initialize disagreement-aware question answerer.
        
        Args:
            model: Language model
            tokenizer: Model tokenizer
            context: Sampling context
            disagreement_tracker: Disagreement tracker instance
        """
        self.model = model
        self.tokenizer = tokenizer
        self.context = context
        self.tracker = disagreement_tracker
    
    def answer_question_with_tracking(
        self,
        sequence_id: str,
        question: str,
        options: List[str],
        include_cot: bool = True,
        dataset: Optional[str] = None,
    ) -> Dict:
        """
        Answer question while tracking disagreements between samplers.
        
        Args:
            sequence_id: Unique identifier for this sequence
            question: Question text
            options: Answer options
            include_cot: Whether to use chain-of-thought
            dataset: Dataset name
            
        Returns:
            Dictionary with both answers and disagreement info
        """
        self.tracker.start_sequence(sequence_id)
        
        try:
            # Generate the question prompt
            if include_cot:
                # For CoT, we track disagreements during reasoning generation
                result = self._answer_with_cot_tracking(question, options, dataset)
            else:
                # For direct answering, track on final answer token
                result = self._answer_direct_with_tracking(question, options, dataset)
            
            return result
            
        except Exception as e:
            # Log error and end sequence
            self.tracker.end_sequence(
                greedy_answer="ERROR",
                dcbs_answer="ERROR", 
                correct_answer="UNKNOWN",
                greedy_correct=False,
                dcbs_correct=False,
                dataset=dataset,
                question=question,
                metadata={"error": str(e)}
            )
            raise
    
    def _answer_with_cot_tracking(
        self, 
        question: str, 
        options: List[str],
        dataset: Optional[str],
    ) -> Dict:
        """Answer with chain-of-thought while tracking disagreements."""
        # This is a simplified version - in practice you'd integrate with
        # the existing QuestionAnswerer from evaluation_core
        
        # For now, simulate the process by tracking on final answer selection
        return self._answer_direct_with_tracking(question, options, dataset)
    
    def _answer_direct_with_tracking(
        self, 
        question: str, 
        options: List[str],
        dataset: Optional[str],
    ) -> Dict:
        """Answer directly while tracking disagreements on final token."""
        # Create prompt for direct answering
        prompt = self._create_answer_prompt(question, options)
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        context_tokens = input_ids[0].tolist()
        context_text = self.tokenizer.decode(context_tokens)
        
        # Get logits for answer selection
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :]  # Last token logits
        
        # Get answer token IDs
        answer_tokens = self._get_answer_token_ids(options)
        filter_tokens = set(answer_tokens.values())
        
        # Check for disagreement
        greedy_token, dcbs_token, disagreed = self.tracker.check_for_disagreement(
            logits=logits,
            filter_tokens=filter_tokens,
            context_tokens=context_tokens,
            context_text=context_text,
        )
        
        # Convert tokens back to answers
        greedy_answer = self._token_to_answer(greedy_token, answer_tokens)
        dcbs_answer = self._token_to_answer(dcbs_token, answer_tokens)
        
        return {
            "greedy_answer": greedy_answer,
            "dcbs_answer": dcbs_answer,
            "greedy_token": greedy_token,
            "dcbs_token": dcbs_token,
            "disagreed": disagreed,
            "answer_tokens": answer_tokens,
        }
    
    def _create_answer_prompt(self, question: str, options: List[str]) -> str:
        """Create prompt for answer selection."""
        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        return f"Question: {question}\n\nOptions:\n{options_text}\n\nThe answer is:"
    
    def _get_answer_token_ids(self, options: List[str]) -> Dict[str, int]:
        """Get token IDs for answer options."""
        answer_tokens = {}
        for i, option in enumerate(options):
            letter = chr(65 + i)  # A, B, C, D
            # Try different tokenization strategies
            candidates = [f" {letter}", letter, f"{letter}."]
            
            for candidate in candidates:
                tokens = self.tokenizer.encode(candidate, add_special_tokens=False)
                if len(tokens) == 1:
                    answer_tokens[option] = tokens[0]
                    break
            else:
                # Fallback
                tokens = self.tokenizer.encode(f" {letter}", add_special_tokens=False)
                answer_tokens[option] = tokens[-1]
        
        return answer_tokens
    
    def _token_to_answer(self, token_id: int, answer_tokens: Dict[str, int]) -> str:
        """Convert token ID back to answer option."""
        for option, tid in answer_tokens.items():
            if tid == token_id:
                return option
        return "UNKNOWN" 