"""DCBS sampling implementation aligned with core DCBS sampler."""

from typing import Dict, Any, List, Optional, Set
import logging
import torch

from ..dcbs.clustering import (
    KMeansClusterer,
    DBSCANClusterer,
    HierarchicalClusterer,
    TopNCandidateSelector,
)
from ..dcbs.category_sampling import (
    CategorySampler,
    ConfidenceAwareCategorySelector,
    GreedyTokenSelector,
)
from ..dcbs.samplers.dcbs_sampler import DCBSSampler as CoreDCBSSampler
from ..dcbs.samplers.base import SamplingContext

logger = logging.getLogger(__name__)


class DCBSSampler(CoreDCBSSampler):
    """Messages-completion DCBS sampler mirroring core DCBS behavior.

    Exposes sample() and sample_batch() identical to the core sampler; adds a
    convenience generate() for sequence decoding by iteratively calling
    sample_batch().
    """

    def __init__(
        self,
        k: int = 8,
        top_n: int = 50,
        clustering_method: str = "kmeans",
        enable_caching: bool = True,
        context: Optional[SamplingContext] = None,
        debug_mode: Optional[bool] = None,
        enable_cluster_history: Optional[bool] = None,
        cache_config: Optional[dict] = None,
    ) -> None:
        # Build clusterer per requested method
        if clustering_method == "kmeans":
            clusterer = KMeansClusterer(k=k)
        elif clustering_method == "dbscan":
            clusterer = DBSCANClusterer()
        elif clustering_method == "hierarchical":
            clusterer = HierarchicalClusterer(k=k)
        else:
            raise ValueError(f"Unknown clustering method: {clustering_method}")

        candidate_selector = TopNCandidateSelector(top_n=top_n)
        category_sampler = CategorySampler(
            category_selector=ConfidenceAwareCategorySelector(),
            token_selector=GreedyTokenSelector(),
        )
        super().__init__(
            clusterer=clusterer,
            candidate_selector=candidate_selector,
            category_sampler=category_sampler,
            context=context,
            cache_config=cache_config,
            enable_caching=enable_caching,
            debug_mode=debug_mode,
            enable_cluster_history=enable_cluster_history,
        )
        self.k = k
        self.top_n = top_n
        self.clustering_method = clustering_method

    def generate(
        self,
        model,
        inputs,
        max_new_tokens: int,
        return_logprobs: bool = False,
    ):
        """Iterative DCBS decoding via sample_batch over steps (deterministic)."""
        batch_size = inputs["input_ids"].shape[0]
        # Build sampling context from model
        embedding_layer = model.get_embedding_layer()
        device = (
            embedding_layer.weight.device
            if hasattr(embedding_layer, "weight")
            else getattr(model, "device", "cpu")
        )
        context = SamplingContext(
            embedding_layer=embedding_layer,
            tokenizer=getattr(model, "tokenizer", None),
            device=device,
        )

        generated_ids: list[list[int]] = [[] for _ in range(batch_size)]
        logprob_seqs: list[list[float]] = [[] for _ in range(batch_size)] if return_logprobs else []

        for _ in range(max_new_tokens):
            logits_batch = model.next_token_logits(inputs)
            next_ids = self.sample_batch(logits_batch, context=context)

            if return_logprobs:
                with torch.no_grad():
                    lp = torch.log_softmax(logits_batch, dim=-1)
                    batch_indices = torch.arange(logits_batch.size(0), device=lp.device)
                    gathered = lp[batch_indices, torch.tensor(next_ids, device=lp.device)]
                    for i in range(batch_size):
                        logprob_seqs[i].append(float(gathered[i].item()))

            for i in range(batch_size):
                generated_ids[i].append(int(next_ids[i]))
            inputs = model.append_tokens(inputs, next_ids)

        return generated_ids, (logprob_seqs if return_logprobs else None)

    # Backward-compat convenience for external callers
    def sample_token(
        self,
        logits: torch.Tensor,
        context: Optional[SamplingContext] = None,
        filter_tokens: Optional[Set[int]] = None,
    ) -> int:
        return self.sample(logits, filter_tokens=filter_tokens, context=context)

    @property
    def method_name(self) -> str:
        return f"dcbs_{self.clustering_method}_k{self.k}"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "top_n": self.top_n,
            "clustering_method": self.clustering_method,
            "enable_caching": getattr(self, "enable_caching", True),
        }
