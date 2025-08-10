from __future__ import annotations

"""Deterministic hierarchical clustering‐tightening sampler.

This sampler starts with the top-N candidate tokens, clusters them with DBSCAN,
then repeatedly tightens the clustering parameters (eps ↓, min_samples ↑) on
only the winning cluster until a single token remains or no further split is
possible.  Every decision is fully deterministic; if ties occur the sampler
falls back to (cluster mass → max single-token probability → smallest token id).

The algorithm is intended for chain-of-thought decoding steps where diversity
inside the top-N matters but the final choice must stay reproducible.
"""

from typing import Optional, Set, List

import torch

from .base import Sampler, SamplingContext
from ..clustering import TopNCandidateSelector, DBSCANClusterer, TokenClusterer
from ..embedding_ops import EmbeddingOperations
from ..constants import DEFAULT_TOP_N, PROB_EPSILON


class DeterministicHierLoopSampler(Sampler):
    """Hierarchical deterministic sampler with progressive DBSCAN tightening."""

    def __init__(
        self,
        top_n: int = DEFAULT_TOP_N,
        initial_eps: float = 0.3,
        eps_decay: float = 0.5,
        initial_min_samples: int = 2,
        min_samples_step: int = 1,
        max_iters: int = 4,
        context: Optional[SamplingContext] = None,
        enable_caching: bool = True,
        debug_mode: bool = False,
        clusterer: Optional[TokenClusterer] = None,
    ) -> None:
        self.top_n = top_n
        self.initial_eps = initial_eps
        self.eps_decay = eps_decay
        self.initial_min_samples = initial_min_samples
        self.min_samples_step = min_samples_step
        self.max_iters = max_iters
        self.context = context

        # Injected clusterer (can be DBSCANClusterer or KMeansClusterer)
        self.clusterer = clusterer

        # Reuse existing utilities
        self.candidate_selector = TopNCandidateSelector(top_n=top_n)
        self.embedding_ops = EmbeddingOperations(None if not enable_caching else None)  # no cache manager for now

        # Optional debug storage
        self._iter_logs: List[dict] = [] if debug_mode else None

    # ---------------------------------------------------------------------
    # Sampler API
    # ---------------------------------------------------------------------
    def sample(
        self,
        logits: torch.Tensor,
        filter_tokens: Optional[Set[int]] = None,
        context: Optional[SamplingContext] = None,
    ) -> int:
        ctx = context or self.context
        if ctx is None or ctx.embedding_layer is None:
            raise ValueError("SamplingContext with embedding_layer is required.")

        # 1. Candidate tokens & probabilities
        candidate_ids: List[int] = self.candidate_selector.select_candidates(logits, filter_tokens)
        if not candidate_ids:
            raise ValueError("No candidate tokens available for sampling.")

        device = logits.device
        cand_tensor = torch.tensor(candidate_ids, device=device)
        cand_logits = logits[cand_tensor]
        cand_probs = torch.softmax(cand_logits, dim=-1)

        # 2. Normalised embeddings
        embeddings = self.embedding_ops.get_normalized_embeddings(cand_tensor, ctx.embedding_layer)
        # Ensure embeddings are float32; NumPy lacks full bfloat16 support.
        if embeddings.dtype != torch.float32:
            embeddings = embeddings.to(torch.float32)

        # 3. Hierarchical loop
        working_indices = list(range(len(candidate_ids)))  # indices relative to cand_tensor
        eps = self.initial_eps
        min_samples = self.initial_min_samples
        last_working_indices: List[int] | None = None

        for iteration in range(self.max_iters):
            if len(working_indices) == 1:
                break  # single token left

            # Cluster current subset using the provided clusterer (DBSCAN or KMeans)
            subset_embeddings = embeddings[working_indices]

            if self.clusterer is None:
                # Default to DBSCAN semantics if no clusterer was provided
                effective_clusterer: TokenClusterer = DBSCANClusterer(
                    eps=eps, min_samples=min_samples, metric="cosine"
                )
            else:
                # Use the injected clusterer; if it's DBSCAN, construct a fresh instance with updated params
                base_clusterer = self.clusterer
                if isinstance(base_clusterer, DBSCANClusterer):
                    effective_clusterer = DBSCANClusterer(
                        eps=eps,
                        min_samples=min_samples,
                        metric=base_clusterer.metric,
                        n_jobs=base_clusterer.n_jobs,
                    )
                else:
                    effective_clusterer = base_clusterer

            # Optional weighting by candidate probabilities if supported (only when enabled on sampler)
            if hasattr(effective_clusterer, "set_sample_weights") and getattr(self, 'cluster_weighting', 'none') == 'prob':
                try:
                    weights_np = cand_probs[working_indices].detach().cpu().numpy()
                    effective_clusterer.set_sample_weights(weights_np)
                except Exception:
                    pass
            labels = effective_clusterer.cluster(subset_embeddings)
            n_clusters = effective_clusterer.num_clusters

            # All points noise or single cluster ⇒ cannot split further
            if n_clusters <= 1:
                break

            # Build cluster -> member indices mapping
            clusters: List[List[int]] = [[] for _ in range(n_clusters)]
            for idx, label in zip(working_indices, labels):
                clusters[label].append(idx)

            # Compute cluster masses
            masses = [float(cand_probs[cluster].sum().item()) for cluster in clusters]

            # Choose winning cluster deterministically
            winner_label = self._choose_cluster(clusters, masses, cand_probs)
            new_working_indices = clusters[winner_label]

            # Check if nothing changed to avoid infinite loop
            if last_working_indices == new_working_indices:
                break

            last_working_indices = working_indices
            working_indices = new_working_indices

            # Tighten parameters for next round (only meaningful for DBSCAN)
            eps *= self.eps_decay
            min_samples += self.min_samples_step

            if self._iter_logs is not None:
                self._iter_logs.append(
                    {
                        "iter": iteration,
                        "eps": eps,
                        "min_samples": min_samples,
                        "n_clusters": n_clusters,
                        "winner_size": len(working_indices),
                        "winner_mass": masses[winner_label],
                    }
                )

        # Final selection: argmax(prob) within working_indices
        final_probs = cand_probs[working_indices]
        rel_idx = torch.argmax(final_probs).item()
        chosen_idx = working_indices[rel_idx]
        return candidate_ids[chosen_idx]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _choose_cluster(clusters: List[List[int]], masses: List[float], probs: torch.Tensor) -> int:
        """Deterministic tie-break: mass → max single-token prob → min token id."""
        # 1. Highest mass
        max_mass = max(masses)
        candidate_labels = [i for i, m in enumerate(masses) if abs(m - max_mass) < PROB_EPSILON]
        if len(candidate_labels) == 1:
            return candidate_labels[0]

        # 2. Highest single-token probability inside cluster
        best_label = candidate_labels[0]
        best_prob = 0.0
        for lbl in candidate_labels:
            cluster_probs = probs[clusters[lbl]]
            max_prob = float(cluster_probs.max().item())
            if max_prob > best_prob + PROB_EPSILON:
                best_label = lbl
                best_prob = max_prob
            elif abs(max_prob - best_prob) < PROB_EPSILON:
                # 3. Lowest token id tie-break
                min_token_lbl = min(clusters[lbl])
                min_token_best = min(clusters[best_label])
                if min_token_lbl < min_token_best:
                    best_label = lbl
        return best_label

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------
    def get_iter_logs(self) -> Optional[List[dict]]:
        """Return per-iteration logs if debug_mode was enabled."""
        return self._iter_logs

    # For factory & metrics
    def get_params(self) -> dict:  # noqa: D401 – simple params dump
        return {
            "top_n": self.top_n,
            "initial_eps": self.initial_eps,
            "eps_decay": self.eps_decay,
            "initial_min_samples": self.initial_min_samples,
            "min_samples_step": self.min_samples_step,
            "max_iters": self.max_iters,
            "clusterer": type(self.clusterer).__name__ if self.clusterer is not None else "DBSCANClusterer",
        } 