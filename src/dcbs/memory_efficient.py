"""Memory efficient DCBS utilities."""

from typing import List, Optional

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from .samplers.base import SamplingContext


class MemoryEfficientDCBS:
    """Memory-efficient DCBS implementation for large-scale processing."""

    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample_with_memory_limit(
        self,
        logits: torch.Tensor,
        filter_tokens: Optional[set],
        context: SamplingContext,
        k: int = 8,
        top_n: int = 50,
    ) -> int:
        """Sample with memory usage constraints."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            _ = torch.cuda.memory_allocated()

        adaptive_top_n = self._compute_adaptive_top_n(logits, top_n)
        if filter_tokens:
            candidates = list(filter_tokens)
        else:
            candidates = self._get_top_n_memory_efficient(logits, adaptive_top_n)

        if len(candidates) <= 3:
            if len(candidates) == 0:
                # No candidates available, fallback to greedy selection from all logits
                return torch.argmax(logits).item()
            candidate_logits = logits[candidates]
            best_idx = torch.argmax(candidate_logits).item()
            return candidates[best_idx]

        embeddings = self._get_embeddings_memory_efficient(candidates, context.embedding_layer)
        labels = self._cluster_memory_efficient(embeddings, k)
        return self._select_token_from_clusters(logits, candidates, labels, filter_tokens)

    def _compute_adaptive_top_n(self, logits: torch.Tensor, target_top_n: int) -> int:
        """Compute adaptive top_n based on available memory."""
        vocab_size = logits.shape[0]
        if torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(0).total_memory
            used_memory = torch.cuda.memory_allocated()
            free_memory = available_memory - used_memory
            memory_per_token = 512 * 4
            max_tokens_by_memory = free_memory // (memory_per_token * 4)
            return min(target_top_n, max_tokens_by_memory, vocab_size)
        return min(target_top_n, vocab_size)

    def _get_top_n_memory_efficient(self, logits: torch.Tensor, top_n: int) -> List[int]:
        """Get top-n tokens with minimal memory overhead."""
        _, indices = torch.topk(logits, top_n, largest=True, sorted=False)
        return indices.cpu().tolist()

    def _get_embeddings_memory_efficient(
        self, candidates: List[int], embedding_layer: torch.nn.Embedding
    ) -> torch.Tensor:
        """Get embeddings with memory constraints."""
        max_batch_size = min(100, len(candidates))
        embeddings_list = []
        for i in range(0, len(candidates), max_batch_size):
            batch_candidates = candidates[i : i + max_batch_size]
            batch_tensor = torch.tensor(batch_candidates, device=self.device)
            with torch.no_grad():
                batch_embeddings = embedding_layer(batch_tensor)
                embeddings_list.append(batch_embeddings.cpu())
        return torch.cat(embeddings_list, dim=0).to(self.device)

    def _cluster_memory_efficient(self, embeddings: torch.Tensor, k: int) -> np.ndarray:
        """Memory-efficient clustering."""
        embeddings_cpu = embeddings.cpu().numpy()
        effective_k = min(k, len(embeddings_cpu))
        batch_size = min(32, len(embeddings_cpu))
        kmeans = MiniBatchKMeans(
            n_clusters=effective_k,
            batch_size=batch_size,
            max_iter=5,
            random_state=42,
        )
        labels = kmeans.fit_predict(embeddings_cpu)
        return labels

    def _select_token_from_clusters(
        self,
        logits: torch.Tensor,
        candidates: List[int],
        labels: np.ndarray,
        filter_tokens: Optional[set],
    ) -> int:
        """Standard token selection from clusters."""
        candidate_logits = logits[candidates]
        candidate_probs = torch.softmax(candidate_logits, dim=-1)
        k = len(np.unique(labels))
        clusters = [[] for _ in range(k)]
        for i, label in enumerate(labels):
            clusters[label].append(i)
        cluster_probs = [
            candidate_probs[cluster].sum().item() if cluster else 0.0 for cluster in clusters
        ]
        if sum(cluster_probs) == 0:
            if len(candidates) == 0:
                # No candidates available, fallback to greedy selection
                return torch.argmax(logits).item()
            return candidates[torch.argmax(candidate_probs).item()]
        best_cluster_idx = np.argmax(cluster_probs)
        cluster_indices = clusters[best_cluster_idx]
        if filter_tokens:
            valid_indices = [i for i in cluster_indices if candidates[i] in filter_tokens]
            if valid_indices:
                cluster_indices = valid_indices
        cluster_probs_tensor = candidate_probs[cluster_indices]
        best_in_cluster = torch.argmax(cluster_probs_tensor).item()
        return candidates[cluster_indices[best_in_cluster]]
