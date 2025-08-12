"""
Deterministic Category Based Sampling (DCBS) implementation.

This module implements the core DCBS algorithm with clustering abstractions,
caching, and comprehensive error handling.
"""

from typing import Optional, Set, List, Tuple, Dict, Any
import concurrent.futures
import os
import time

import numpy as np
import torch

from .base import Sampler, SamplingContext
from ..clustering import TokenClusterer, CandidateSelector, SingleCluster
from ..category_sampling import CategorySampler, greedy_category_sampler
from ..cache_manager import CacheConfig, DCBSCacheManager, get_cache_manager
from ..constants import (
    DEFAULT_K_CLUSTERS,
    DEFAULT_TOP_N,
    DEFAULT_EMBEDDING_CACHE_SIZE,
    DEFAULT_CLUSTER_CACHE_SIZE,
    MIN_TOKENS_FOR_CLUSTERING,
    PROB_EPSILON,
)
from ..debug import DCBSDebugger
from ..embedding_ops import EmbeddingOperations

# Note: BatchDCBSProcessor import moved to avoid circular dependency


def _dcbs_parallel_cluster_job(payload: Tuple[np.ndarray, Dict[str, Any], Optional[np.ndarray]]) -> List[int]:
    """Top-level function for ProcessPoolExecutor to run clustering.

    Args:
        payload: Tuple of (embeddings_np, clusterer_spec)
    Returns:
        labels as a list of ints
    """
    # Backward compatible payload handling: (embeddings_np, spec) or (embeddings_np, spec, weights_np)
    if len(payload) == 2:
        embeddings_np, spec = payload
        weights_np = None
    else:
        embeddings_np, spec, weights_np = payload
    # Reconstruct clusterer locally to avoid cross-process state
    from src.dcbs.clustering import (
        KMeansClusterer,
        DBSCANClusterer,
        HierarchicalClusterer,
    )
    import torch as _torch

    clusterer_type = spec["type"]
    params = spec.get("params", {})
    if clusterer_type == "kmeans":
        clusterer = KMeansClusterer(**params)
    elif clusterer_type == "dbscan":
        clusterer = DBSCANClusterer(**params)
    elif clusterer_type == "hierarchical":
        clusterer = HierarchicalClusterer(**params)
    else:
        clusterer = KMeansClusterer(k=params.get("k", 8))
    # If the clusterer supports sample weights and weights were provided, set them
    if weights_np is not None and hasattr(clusterer, 'set_sample_weights'):
        try:
            clusterer.set_sample_weights(weights_np)
        except Exception:
            pass
    labels = clusterer.cluster(_torch.from_numpy(embeddings_np))
    return labels.tolist()


def _init_dcbs_worker_env() -> None:
    """Initializer for process-pool workers to avoid CPU oversubscription.

    Limits BLAS/OpenMP threads per worker so N processes do not multiply threads.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

class DCBSSampler(Sampler):
    """
    Deterministic Category Based Sampling using clustering abstractions.
    
    This sampling method clusters token embeddings into semantic categories,
    selects the cluster with highest probability mass, then chooses the
    highest probability token from that cluster. All selections are deterministic.
    """

    def __init__(
        self,
        clusterer: TokenClusterer,
        candidate_selector: CandidateSelector,
        category_sampler: Optional[CategorySampler] = None,
        context: Optional[SamplingContext] = None,
        cache_config: Optional[dict] = None,
        enable_caching: bool = True,
        debug_mode: Optional[bool] = None,
        enable_cluster_history: Optional[bool] = None,
        debug_output_file: Optional[str] = None,
        enable_batch_processing: bool = True,
        batch_processing_threshold: int = 4,
    ):
        """
        Initialize the DCBS sampler.
        
        Args:
            clusterer: Token clustering strategy
            candidate_selector: Candidate token selection strategy
            category_sampler: Strategy for sampling from categories (default: greedy)
            context: Sampling context with embedding layer (required for DCBS)
            cache_config: Optional cache configuration
            enable_caching: Whether to enable caching (default: True)
            debug_mode: Enable debug logging (default: False)
            enable_cluster_history: Track cluster decisions (default: False)
            debug_output_file: File path for debug output
            enable_batch_processing: Enable GPU parallel batch processing (default: True)
            batch_processing_threshold: Minimum batch size for parallel processing (default: 4)
        """
        self.clusterer = clusterer
        self.candidate_selector = candidate_selector
        # Backwards-compat: if third positional was actually cache_config dict
        if isinstance(category_sampler, dict) and context is None and cache_config is None:
            cache_config = category_sampler
            category_sampler = None

        self.category_sampler = category_sampler or greedy_category_sampler
        self.context = context
        self.enable_caching = enable_caching
        self.enable_batch_processing = enable_batch_processing
        self.batch_processing_threshold = batch_processing_threshold

        # Initialize cache manager only if caching is enabled
        if self.enable_caching:
            if cache_config:
                config = CacheConfig(**cache_config)
            else:
                # Default cache configuration with reasonable limits
                config = CacheConfig(
                    embedding_cache_size=DEFAULT_EMBEDDING_CACHE_SIZE, 
                    cluster_cache_size=DEFAULT_CLUSTER_CACHE_SIZE, 
                    enable_metrics=True
                )
            
            # Get thread-safe cache manager instance
            self.cache_manager = get_cache_manager(config)
        else:
            self.cache_manager = None

        # BUGFIX: Removed the faulty BatchDCBSProcessor initialization.
        # This component was overriding the intended clusterer and causing silent failures.
        # The sampler will now correctly use the clusterer it is given.
        self.batch_processor = None

        # Initialize debugging
        self.debugger = DCBSDebugger(debug_mode, enable_cluster_history, debug_output_file)
        
        # Initialize embedding operations
        self.embedding_ops = EmbeddingOperations(self.cache_manager)

    # Factory methods have been moved to src.dcbs.factory.DCBSSamplerFactory
    # to eliminate circular dependencies. Use DCBSSamplerFactory.create_default() instead.

    # ---------------------------------------------------------------------
    # Backwards compatibility factory methods
    # ---------------------------------------------------------------------
    @staticmethod
    def create_default(
        k: int = DEFAULT_K_CLUSTERS,
        top_n: int = DEFAULT_TOP_N,
        context: Optional[SamplingContext] = None,
        cache_config: Optional[dict] = None,
        enable_caching: bool = True,
        debug_mode: Optional[bool] = None,
        enable_cluster_history: Optional[bool] = None,
    ) -> "DCBSSampler":
        """Create a default DCBSSampler.

        This method forwards to :class:`~src.dcbs.factory.DCBSSamplerFactory`
        to maintain backwards compatibility with older code that expected
        these constructors on :class:`DCBSSampler` itself.
        """
        from ..factory import DCBSSamplerFactory

        return DCBSSamplerFactory.create_default(
            k=k,
            top_n=top_n,
            context=context,
            cache_config=cache_config,
            enable_caching=enable_caching,
            debug_mode=debug_mode,
            enable_cluster_history=enable_cluster_history,
        )

    @staticmethod
    def create_no_cache(
        k: int = DEFAULT_K_CLUSTERS,
        top_n: int = DEFAULT_TOP_N,
        context: Optional[SamplingContext] = None,
        **kwargs,
    ) -> "DCBSSampler":
        """Create a DCBSSampler with caching disabled."""
        from ..factory import DCBSSamplerFactory

        return DCBSSamplerFactory.create_no_cache(k=k, top_n=top_n, context=context, **kwargs)

    @staticmethod
    def create_lightweight(
        k: int = 4,
        top_n: int = 20,
        context: Optional[SamplingContext] = None,
    ) -> "DCBSSampler":
        """Create a lightweight DCBSSampler for constrained environments."""
        from ..factory import DCBSSamplerFactory

        return DCBSSamplerFactory.create_lightweight(k=k, top_n=top_n, context=context)

    def sample(
        self,
        logits: torch.Tensor,
        filter_tokens: Optional[Set[int]] = None,
        context: Optional[SamplingContext] = None,
    ) -> int:
        """
        Sample a token using DCBS algorithm.
        
        Args:
            logits: Token logits from the model
            filter_tokens: Optional set of allowed token IDs
            context: Sampling context (optional, uses instance context if not provided)
            
        Returns:
            Selected token ID
            
        Raises:
            ValueError: If context or embedding_layer is missing
        """
        # Use provided context or instance context
        effective_context = context or self.context
        
        if effective_context is None or effective_context.embedding_layer is None:
            raise ValueError("DCBS requires a SamplingContext with embedding_layer. "
                           "Please provide a valid context with an embedding layer.")

        embedding = effective_context.embedding_layer

        # Handle simple cases first
        if filter_tokens and len(filter_tokens) == 1:
            return list(filter_tokens)[0]

        # For multiple-choice questions, proceed with the unified CategorySampler path
        # This ensures Enhanced DCBS uses proper DuelingAlgorithmCategorySelector

        # CRITICAL FIX: If filter_tokens are provided (i.e., for multiple-choice questions),
        # they MUST be used as the candidate set. Ignoring them causes the sampler to
        # consider irrelevant tokens, leading to a collapse in performance.
        if filter_tokens:
            candidate_ids = list(filter_tokens)
        else:
            # For open-ended generation, use the candidate selector
            candidate_ids = self.candidate_selector.select_candidates(logits, None)

        # Handle insufficient candidates for clustering
        if len(candidate_ids) <= MIN_TOKENS_FOR_CLUSTERING:
            return self._simple_selection(logits, candidate_ids)

        # Handle edge cases with invalid logits
        if self._has_invalid_logits(logits, candidate_ids):
            raise ValueError("Invalid logits detected: contains NaN or all-infinite values. "
                           "This indicates a problem with model output that must be addressed.")

        # Main DCBS algorithm
        return self._dcbs_selection(logits, candidate_ids, embedding, filter_tokens)

    def _simple_selection(self, logits: torch.Tensor, candidate_ids: list) -> int:
        """Select best token when clustering is not applicable."""
        candidate_logits = logits[candidate_ids]
        probs = torch.softmax(candidate_logits, dim=-1)
        selected_idx = torch.argmax(probs).item()
        return candidate_ids[selected_idx]

    def _has_invalid_logits(self, logits: torch.Tensor, candidate_ids: list) -> bool:
        """Check if candidate logits contain invalid values."""
        candidate_logits = logits[candidate_ids]
        return (
            torch.isinf(candidate_logits).any() or torch.isnan(candidate_logits).any()
        )
    
    def _validate_batch_inputs(self, logits_batch: torch.Tensor, filter_tokens_batch: Optional[List[Optional[Set[int]]]]) -> None:
        """
        Comprehensive validation of batch inputs to catch common errors early.
        
        Args:
            logits_batch: Batch of token logits
            filter_tokens_batch: Optional list of filter sets
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate logits_batch tensor
        if not isinstance(logits_batch, torch.Tensor):
            raise ValueError(f"logits_batch must be a torch.Tensor, got {type(logits_batch)}")
        
        if logits_batch.dim() != 2:
            raise ValueError(f"logits_batch must be 2D [batch_size, vocab_size], got {logits_batch.dim()}D tensor with shape {logits_batch.shape}")
        
        batch_size, vocab_size = logits_batch.shape
        
        if batch_size < 0:
            raise ValueError(f"Invalid batch_size: {batch_size}")
        
        if vocab_size <= 0:
            raise ValueError(f"Invalid vocab_size: {vocab_size}")
        
        # Validate filter_tokens_batch if provided
        if filter_tokens_batch is not None:
            if not isinstance(filter_tokens_batch, list):
                raise ValueError(f"filter_tokens_batch must be a list, got {type(filter_tokens_batch)}")
            
            if len(filter_tokens_batch) != batch_size:
                raise ValueError(f"filter_tokens_batch length ({len(filter_tokens_batch)}) must match batch_size ({batch_size})")
            
            # Validate each filter set
            for i, filter_tokens in enumerate(filter_tokens_batch):
                if filter_tokens is not None:
                    if not isinstance(filter_tokens, set):
                        raise ValueError(f"filter_tokens_batch[{i}] must be a set or None, got {type(filter_tokens)}")
                    
                    # Check for invalid token IDs
                    if filter_tokens:
                        max_token_id = max(filter_tokens)
                        min_token_id = min(filter_tokens)
                        
                        if min_token_id < 0:
                            raise ValueError(f"filter_tokens_batch[{i}] contains negative token ID: {min_token_id}")
                        
                        if max_token_id >= vocab_size:
                            raise ValueError(f"filter_tokens_batch[{i}] contains token ID {max_token_id} >= vocab_size {vocab_size}")
        
        # Check for invalid values in logits
        if torch.isnan(logits_batch).any():
            raise ValueError("logits_batch contains NaN values")
        
        # Allow infinite values as they can be valid (e.g., -inf for masking)
        # but warn if ALL values are infinite
        if torch.isinf(logits_batch).all():
            raise ValueError("logits_batch contains only infinite values")

    def _fallback_selection(
        self, logits: torch.Tensor, filter_tokens: Optional[Set[int]]
    ) -> int:
        """Fallback selection when DCBS cannot be applied.
        
        Note: This method is kept for backward compatibility but should not
        be used for invalid logits handling. Invalid logits should raise exceptions.
        """
        if filter_tokens:
            filter_list = list(filter_tokens)
            filter_logits = logits[filter_list]
            best_idx = torch.argmax(filter_logits).item()
            return filter_list[best_idx]
        else:
            return logits.argmax().item()

    def _dcbs_selection(
        self,
        logits: torch.Tensor,
        candidate_ids: list,
        embedding: torch.nn.Embedding,
        filter_tokens: Optional[Set[int]],
    ) -> int:
        """Main DCBS algorithm implementation."""
        self.debugger.increment_stat("total_samples")
        self.debugger.log_debug(f"Starting DCBS selection with {len(candidate_ids)} candidates")
        
        # Prepare candidate data
        candidate_data = self._prepare_candidate_data(logits, candidate_ids)
        
        # Get normalized embeddings and perform clustering
        clusters = self._cluster_candidates(candidate_ids, embedding, candidate_data["probs"])
        
        # Select token using category sampler
        selected_token, cluster_probs = self._select_token_from_clusters(
            candidate_ids, candidate_data["probs"], clusters["clusters"], filter_tokens
        )
        
        # Record decision for analysis
        self._record_selection_decision(
            candidate_ids, clusters["labels"], clusters["clusters"], cluster_probs, selected_token
        )
        
        self.debugger.log_debug(f"Selected token {selected_token}")
        return selected_token
    
    def _prepare_candidate_data(self, logits: torch.Tensor, candidate_ids: list) -> dict:
        """Prepare candidate token data for DCBS processing."""
        # CRITICAL FIX: Use logits device consistently for all candidate operations
        device = logits.device
        candidate_ids_tensor = torch.tensor(candidate_ids, device=device)
        candidate_logits = logits[candidate_ids_tensor]
        candidate_probs = torch.softmax(candidate_logits, dim=-1)
        
        return {
            "ids_tensor": candidate_ids_tensor,
            "logits": candidate_logits,
            "probs": candidate_probs
        }
    
    def _cluster_candidates(self, candidate_ids: list, embedding: torch.nn.Embedding, candidate_probs: Optional[torch.Tensor] = None) -> dict:
        """Perform clustering on candidate embeddings."""
        # CRITICAL FIX: Ensure device consistency throughout the pipeline
        # Use the embedding layer's device as the authoritative device
        device = embedding.weight.device
        candidate_ids_tensor = torch.tensor(candidate_ids, device=device)

        # Get normalized embeddings
        norm_embeddings = self.embedding_ops.get_normalized_embeddings(
            candidate_ids_tensor, embedding
        )

        # Perform clustering (optionally with probability weights for clusterers that support it)
        weighting_mode = getattr(self, 'cluster_weighting', 'none')
        if weighting_mode == 'prob' and candidate_probs is not None and hasattr(self.clusterer, "set_sample_weights"):
            try:
                # Align weights to CPU numpy if needed; softmax already computed
                weights_np = candidate_probs.detach().cpu().numpy()
                # Scale so that total weight ≈ number of candidates, keeping DBSCAN min_samples meaningful
                total = float(weights_np.sum())
                if total > 0:
                    scale = len(candidate_ids) / total
                    weights_np = weights_np * scale
                # Avoid zeros-only weights that may break DBSCAN min_samples interpretation
                if np.all(weights_np == 0):
                    weights_np = None
                self.clusterer.set_sample_weights(weights_np)
            except Exception:
                # Ignore weighting failure; proceed unweighted
                pass
        labels = self._perform_clustering(norm_embeddings)
        
        # Group tokens by cluster
        clusters = self._group_by_clusters(labels, self.clusterer.num_clusters)
        
        return {
            "labels": labels,
            "clusters": clusters,
            "embeddings": norm_embeddings
        }
    
    def _select_token_from_clusters(
        self,
        candidate_ids: list,
        candidate_probs: torch.Tensor,
        clusters: list,
        filter_tokens: Optional[Set[int]]
    ) -> tuple:
        """Select the best token from the clustered candidates and return cluster probabilities."""
        cluster_probs = [
            candidate_probs[cluster].sum().item() if cluster else 0.0
            for cluster in clusters
        ]
        token = self.category_sampler.sample_from_clusters(
            candidate_ids, candidate_probs, clusters, filter_tokens
        )
        return token, cluster_probs

    def _perform_clustering(self, embeddings: torch.Tensor) -> np.ndarray:
        """Perform clustering on embeddings with optional caching."""
        self.debugger.increment_stat("clustering_calls")
        
        if not self.enable_caching or self.cache_manager is None:
            return self.clusterer.cluster(embeddings)
        
        # Try cached clustering
        device_str = str(embeddings.device)
        cache_key = (embeddings.shape[0], self.clusterer.num_clusters, device_str)
        
        cached_labels = self.cache_manager.get_clustering(cache_key)
        if cached_labels is not None:
            self.debugger.increment_stat("cache_hits")
            return cached_labels
        
        # Compute and cache
        labels = self.clusterer.cluster(embeddings)
        self.cache_manager.put_clustering(cache_key, labels)
        
        self.debugger.log_debug(
            f"Clustering produced {len(set(labels))} clusters from {self.clusterer.num_clusters} requested"
        )
        
        return labels

    def _group_by_clusters(
        self, labels: np.ndarray, num_clusters: int
    ) -> List[List[int]]:
        """Group token indices by their cluster labels."""
        # For dynamic clustering (like DBSCAN), use actual number of clusters
        # Handle DBSCAN noise points (label = -1) by filtering them out
        valid_labels = labels[labels >= 0]
        if len(valid_labels) == 0:
            # All points are noise, create single cluster with all points
            return [list(range(len(labels)))]
        
        actual_num_clusters = max(len(np.unique(valid_labels)), num_clusters)
        clusters = [[] for _ in range(actual_num_clusters)]
        for i, label in enumerate(labels):
            if label >= 0 and label < len(clusters):  # Skip noise points (label = -1)
                clusters[label].append(i)
        
        # If all clusters are empty due to noise points, put all points in first cluster
        if all(len(cluster) == 0 for cluster in clusters):
            clusters[0] = list(range(len(labels)))
        
        return clusters

    def _record_selection_decision(
        self,
        candidate_ids: list,
        labels: np.ndarray,
        clusters: list,
        cluster_probs: list,
        selected_token: int,
    ) -> None:
        """Record clustering decision for debugging if enabled."""
        if self.debugger.cluster_history_enabled:
            # Find which cluster was selected
            selected_idx = candidate_ids.index(selected_token)
            selected_cluster = labels[selected_idx]
            self.debugger.record_cluster_decision(
                candidate_ids,
                labels,
                selected_cluster,
                selected_token,
                self.clusterer.num_clusters,
                clusters,
                cluster_probs,
            )

    def get_cache_stats(self) -> dict:
        """Get cache performance statistics."""
        if not self.enable_caching or self.cache_manager is None:
            return {"caching_enabled": False, "message": "Caching is disabled"}
        return self.cache_manager.get_cache_stats()

    def clear_caches(self) -> None:
        """Clear all caches."""
        if self.cache_manager:
            self.cache_manager.clear_all_caches()

    def get_debug_stats(self) -> dict:
        """Get debugging statistics."""
        return self.debugger.get_stats()
    
    def get_cluster_history(self) -> Optional[List[dict]]:
        """Get cluster decision history if enabled."""
        return self.debugger.get_cluster_history()
    
    def clear_debug_data(self) -> None:
        """Clear debug data and statistics."""
        self.debugger.clear_debug_data()

    def get_params(self) -> dict:
        """Return the parameters of the sampler."""
        return {
            "clusterer": self.clusterer.__class__.__name__,
            "candidate_selector": self.candidate_selector.__class__.__name__,
            "category_sampler": self.category_sampler.__class__.__name__,
            "enable_caching": self.enable_caching,
        }

    def sample_batch(
        self,
        logits_batch: torch.Tensor,
        filter_tokens_batch: Optional[List[Optional[Set[int]]]] = None,
        context: Optional[SamplingContext] = None,
    ) -> List[int]:
        """
        Sample tokens for a batch of logits using DCBS algorithm.
        
        Args:
            logits_batch: Batch of token logits [batch_size, vocab_size]
            filter_tokens_batch: Optional list of filter sets for each sequence
            context: Sampling context (optional, uses instance context if not provided)
            
        Returns:
            List of selected token IDs, one per sequence in the batch
            
        Raises:
            ValueError: If context or embedding_layer is missing
        """
        # Use provided context or instance context
        effective_context = context or self.context
        
        if effective_context is None or effective_context.embedding_layer is None:
            raise ValueError("DCBS requires a SamplingContext with embedding_layer. "
                           "Please provide a valid context with an embedding layer.")

        # CRITICAL FIX: Comprehensive batch validation
        self._validate_batch_inputs(logits_batch, filter_tokens_batch)
        
        batch_size = logits_batch.shape[0]
        
        # Handle empty batch
        if batch_size == 0:
            return []
        
        # Prepare filter tokens for each sequence
        if filter_tokens_batch is None:
            filter_tokens_batch = [None] * batch_size
        elif len(filter_tokens_batch) != batch_size:
            raise ValueError(f"filter_tokens_batch length ({len(filter_tokens_batch)}) must match batch_size ({batch_size})")
        
        self.debugger.log_debug(f"Starting batch DCBS sampling for {batch_size} sequences")
        
        # Parallelized batch DCBS using process pool for clustering (CPU-bound)
        # 1) Build candidate sets and compute candidate probabilities / embeddings on main process
        device = effective_context.embedding_layer.weight.device
        candidate_sets: List[List[int]] = []
        candidate_probs_list: List[torch.Tensor] = []
        embeddings_list: List[torch.Tensor] = []
        simple_indices: List[int] = []
        simple_results: Dict[int, int] = {}

        for i in range(batch_size):
            logits = logits_batch[i]
            filter_tokens = filter_tokens_batch[i]
            
            # Candidate ids
            if filter_tokens:
                candidate_ids = list(filter_tokens)
            else:
                candidate_ids = self.candidate_selector.select_candidates(logits, None)

            # Handle insufficient candidates without clustering
            if len(candidate_ids) <= MIN_TOKENS_FOR_CLUSTERING:
                simple_results[i] = self._simple_selection(logits, candidate_ids)
                simple_indices.append(i)
                continue

            # Invalid logits guard for candidates
            if self._has_invalid_logits(logits, candidate_ids):
                raise ValueError(
                    f"Invalid logits detected in batch sequence {i}: contains NaN/Inf"
                )

            # Candidate probabilities (normalized over candidates) and embeddings
            cand_ids_tensor = torch.tensor(candidate_ids, device=logits.device)
            cand_logits = logits[cand_ids_tensor]
            cand_probs = torch.softmax(cand_logits, dim=-1)

            # Normalize weights to sum=1 for clustering weighting modes
            weighting_mode = getattr(self, 'cluster_weighting', 'none')
            if weighting_mode in ("prob", "uniform"):
                if weighting_mode == "uniform":
                    # 1/n per candidate
                    num = max(1, cand_probs.numel())
                    cand_weights = torch.full_like(cand_probs, 1.0 / float(num))
                else:
                    # normalized probabilities already sum to 1 over candidates
                    total = float(cand_probs.sum().item())
                    cand_weights = cand_probs if total > 0 else torch.full_like(cand_probs, 1.0 / max(1, cand_probs.numel()))
                # Persist alongside candidate_probs for use by worker
                # Replace candidate_probs_list element with weights for clarity in weighting path
            

            # Embeddings on embedding device, then move to CPU for subprocess
            cand_ids_for_embed = torch.tensor(candidate_ids, device=device)
            norm_embeddings = self.embedding_ops.get_normalized_embeddings(
                cand_ids_for_embed, effective_context.embedding_layer
            )

            candidate_sets.append(candidate_ids)
            # Store probabilities for selection; store weights separately for clustering
            candidate_probs_list.append(cand_probs)
            embeddings_list.append(norm_embeddings)

        # If all were simple, return aggregated results
        if len(simple_results) == batch_size:
            return [simple_results[i] for i in range(batch_size)]

        # 2) Prepare clustering jobs for remaining sequences
        jobs: List[Tuple[np.ndarray, Dict[str, Any], Optional[np.ndarray]]] = []
        index_mapping: List[int] = []  # map job order -> original index

        # Build clusterer spec from self.clusterer
        def _build_clusterer_spec() -> Dict[str, Any]:
            from ..clustering import KMeansClusterer, DBSCANClusterer, HierarchicalClusterer
            if isinstance(self.clusterer, KMeansClusterer):
                return {
                    "type": "kmeans",
                    "params": {
                        "k": self.clusterer.k,
                        "random_seed": self.clusterer.random_seed,
                        "max_iterations": self.clusterer.max_iterations,
                        "min_batch_size": self.clusterer.min_batch_size,
                        "enable_adaptive_k": self.clusterer.enable_adaptive_k,
                        "min_k": self.clusterer.min_k,
                        "max_k": self.clusterer.max_k,
                        "use_elbow_method": getattr(self.clusterer, "use_elbow_method", False),
                    },
                }
            if isinstance(self.clusterer, DBSCANClusterer):
                return {
                    "type": "dbscan",
                    "params": {
                        "eps": self.clusterer.eps,
                        "min_samples": self.clusterer.min_samples,
                        "metric": self.clusterer.metric,
                        "n_jobs": self.clusterer.n_jobs,
                    },
                }
            if isinstance(self.clusterer, HierarchicalClusterer):
                return {
                    "type": "hierarchical",
                    "params": {
                        "k": self.clusterer.k,
                        "linkage": self.clusterer.linkage,
                        "metric": self.clusterer.metric,
                    },
                }
            # Fallback: treat as kmeans with configured default k
            return {"type": "kmeans", "params": {"k": DEFAULT_K_CLUSTERS}}

        clusterer_spec = _build_clusterer_spec()

        for i in range(batch_size):
            if i in simple_results:
                continue
            emb_cpu = embeddings_list[len(index_mapping)].detach().cpu().numpy()
            # Prepare optional weights for this sequence if enabled
            weights_np = None
            weighting_mode = getattr(self, 'cluster_weighting', 'none')
            if weighting_mode in ("prob", "uniform"):
                try:
                    # Build weights that sum to 1; uniform => 1/n
                    probs = candidate_probs_list[len(index_mapping)].detach().cpu().numpy()
                    if weighting_mode == "uniform":
                        n = max(1, probs.shape[0])
                        w = np.ones_like(probs, dtype=float) / float(n)
                    else:
                        s = float(probs.sum())
                        if s > 0:
                            w = probs / s
                        else:
                            n = max(1, probs.shape[0])
                            w = np.ones_like(probs, dtype=float) / float(n)
                    # For DBSCAN we will internally rescale to preserve min_samples semantics
                    weights_np = w
                except Exception:
                    weights_np = None
            jobs.append((emb_cpu, clusterer_spec, weights_np))
            index_mapping.append(i)

        # 3) Run clustering in parallel processes
        labels_results: Dict[int, List[int]] = {}
        if jobs:
            # Decide pool size and initialize worker environment (cap via env var)
            # Priority: sampler.max_cluster_workers (from config) > env > auto
            sampler_max_workers = getattr(self, 'max_cluster_workers', None)
            max_workers_cfg = int(os.environ.get("DCBS_MAX_CLUSTER_WORKERS", "0") or 0)
            auto_workers = max(1, min(os.cpu_count() or 1, len(jobs)))
            if sampler_max_workers is not None and sampler_max_workers > 0:
                max_workers = min(int(sampler_max_workers), auto_workers)
            else:
                max_workers = auto_workers if max_workers_cfg <= 0 else min(max_workers_cfg, auto_workers)
            # Persist the pool across calls using a module-level singleton to avoid spin-up cost
            if not hasattr(self, "_cluster_pool") or self._cluster_pool is None:
                self._cluster_pool = concurrent.futures.ProcessPoolExecutor(
                    max_workers=max_workers, initializer=_init_dcbs_worker_env
                )
            # Submit jobs
            futures = []
            for j_idx, job in enumerate(jobs):
                fut = self._cluster_pool.submit(_dcbs_parallel_cluster_job, job)
                futures.append((fut, index_mapping[j_idx]))
            for fut, orig_idx in futures:
                labels = fut.result()
                labels_results[orig_idx] = labels

        # 4) Final selection on main process using category sampler for consistency
        results: List[int] = [0] * batch_size
        # Fill simple first
        for i, tok in simple_results.items():
            results[i] = tok

        # Iterate over non-simple
        non_simple_counter = 0
        for i in range(batch_size):
            if i in simple_results:
                continue
            candidate_ids = candidate_sets[non_simple_counter]
            candidate_probs = candidate_probs_list[non_simple_counter]
            labels = labels_results.get(i)
            if labels is None:
                # Safety: fallback to local clustering if a job failed silently
                local_labels = self._perform_clustering(embeddings_list[non_simple_counter])
                labels = local_labels.tolist() if hasattr(local_labels, 'tolist') else list(local_labels)

            # Group points by cluster labels (ignore negative labels if any)
            unique_labels = sorted({l for l in labels if l >= 0})
            clusters: List[List[int]] = []
            for lbl in unique_labels:
                clusters.append([idx for idx, v in enumerate(labels) if v == lbl])
            if not clusters:
                clusters = [list(range(len(labels)))]

            filter_tokens = filter_tokens_batch[i]
            selected_token, _cluster_probs = self._select_token_from_clusters(
                candidate_ids, candidate_probs, clusters, filter_tokens
            )
            results[i] = selected_token
            non_simple_counter += 1

        return results 

    def cleanup(self) -> None:
        """Clean up resources used by the sampler."""
        # Shutdown persistent cluster pool if exists
        try:
            if hasattr(self, "_cluster_pool") and self._cluster_pool is not None:
                self._cluster_pool.shutdown(wait=False, cancel_futures=True)
                self._cluster_pool = None
        except Exception:
            pass
        
        if self.cache_manager is not None:
            # Clear caches to free memory
            self.cache_manager.clear_all_caches()
    
    def get_batch_processing_stats(self) -> dict:
        """Get statistics about batch processing performance."""
        return {
            "batch_processing_enabled": True,
            "cluster_pool_active": hasattr(self, "_cluster_pool") and self._cluster_pool is not None,
        }
    
    def __del__(self):
        """Cleanup when sampler is destroyed."""
        try:
            self.cleanup()
        except Exception:
            # Ignore errors during cleanup
            pass 