"""
Disjunctive Category Beam Search (DCBS) implementation.

Reference: Smith, J. et al. (2023). "DCBS: Semantically Diverse Sampling for LMs."
"""

import torch
import numpy as np
from typing import Optional, Set, Dict, Tuple, List, Any, TypedDict, NamedTuple
from sklearn.cluster import MiniBatchKMeans
import os
import csv
import time
from collections import OrderedDict
import random

# ----- Type Definitions -----
class ClusterResult(NamedTuple):
    """Results of token clustering."""
    labels: np.ndarray
    cluster_probs: List[float]
    
class TokenDistribution(TypedDict):
    """Token probability distribution and metadata."""
    cluster_dists: List[float]
    token_dists: List[float]
    token_ids: List[int]
    selected_cluster: int
    cluster_assignments: List[int]

class DCBSConfig(TypedDict):
    """DCBS algorithm configuration."""
    temperature: float
    min_tokens_for_clustering: int
    clustering_random_seed: int
    min_kmeans_iterations: int
    min_batch_size: int

class CacheConfig(TypedDict):
    """Cache configuration."""
    embedding_cache_size: int
    cluster_cache_size: int

# ----- Default Configuration -----
# These defaults are used if no configuration is provided
DEFAULT_DCBS_CONFIG: DCBSConfig = {
    "temperature": 1.0,
    "min_tokens_for_clustering": 3,
    "clustering_random_seed": 42,
    "min_kmeans_iterations": 5,
    "min_batch_size": 3584,
}

DEFAULT_CACHE_CONFIG: CacheConfig = {
    "embedding_cache_size": 1000,
    "cluster_cache_size": 200,
}

# Small value to prevent division by zero
PROB_EPSILON = 1e-6

# ----- Cache Structures -----
# LRU caches to avoid redundant computations
_token_embedding_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
_embedding_cluster_cache: OrderedDict[Tuple[int, int, str], np.ndarray] = OrderedDict()


def _top_n_with_force(
    logits: torch.Tensor, top_n: int, force_include_ids=None
) -> List[int]:
    """Get top-n token IDs with forced inclusions."""
    sorted_indices = torch.argsort(logits, descending=True)
    top_indices = set(sorted_indices[:top_n].cpu().tolist())

    if force_include_ids:
        for idx in force_include_ids:
            if idx not in top_indices:
                top_indices.add(idx)

    return list(top_indices)


def get_cached_embeddings(
    token_ids: torch.Tensor, 
    embedding: torch.nn.Embedding, 
    cache_config: Optional[CacheConfig] = None
) -> torch.Tensor:
    """Gets embeddings using cache when possible.
    
    Args:
        token_ids: Token IDs
        embedding: Model's embedding layer
        cache_config: Cache config
        
    Returns:
        Token embeddings
    """
    if cache_config is None:
        cache_config = DEFAULT_CACHE_CONFIG
        
    embedding_cache_size = cache_config["embedding_cache_size"]
    
    device = token_ids.device
    result = torch.zeros((len(token_ids), embedding.embedding_dim), device=device)

    uncached_indices = []
    uncached_ids = []

    for i, token_id in enumerate(token_ids.cpu().tolist()):
        if token_id in _token_embedding_cache:
            cached_embed = _token_embedding_cache[token_id]
            if cached_embed.device != device:
                cached_embed = cached_embed.to(device)
            result[i] = cached_embed

            if token_id in _token_embedding_cache:
                _token_embedding_cache.move_to_end(token_id)
        else:
            uncached_indices.append(i)
            uncached_ids.append(token_id)

    if uncached_indices:
        with torch.no_grad():
            if embedding.weight.shape[0] > max(uncached_ids):
                uncached_embeds = embedding(torch.tensor(uncached_ids, device=device))

                for i, idx in enumerate(uncached_indices):
                    token_id = uncached_ids[i]
                    _token_embedding_cache[token_id] = uncached_embeds[i].detach()
                    result[idx] = uncached_embeds[i]

                    if len(_token_embedding_cache) > embedding_cache_size:
                        _token_embedding_cache.popitem(last=False)

    return result


def get_cached_clustering(
    embeddings: torch.Tensor, 
    k: int, 
    dcbs_config: Optional[DCBSConfig] = None,
    cache_config: Optional[CacheConfig] = None
) -> np.ndarray:
    """Gets clustering results using cache.
    
    Args:
        embeddings: Token embeddings
        k: Number of clusters
        dcbs_config: DCBS config
        cache_config: Cache config
        
    Returns:
        Cluster labels
    """
    if dcbs_config is None:
        dcbs_config = DEFAULT_DCBS_CONFIG
        
    if cache_config is None:
        cache_config = DEFAULT_CACHE_CONFIG
    
    clustering_random_seed = dcbs_config["clustering_random_seed"]
    min_kmeans_iterations = dcbs_config["min_kmeans_iterations"]
    min_batch_size = dcbs_config["min_batch_size"]
    cluster_cache_size = cache_config["cluster_cache_size"]
    
    device_str = str(embeddings.device)
    cache_key = (embeddings.shape[0], k, device_str)

    if cache_key in _embedding_cluster_cache:
        _embedding_cluster_cache.move_to_end(cache_key)
        return _embedding_cluster_cache[cache_key]

    embeddings_np = embeddings.detach().cpu().numpy()
    batch_size = max(min_batch_size, len(embeddings_np))

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        max_iter=min_kmeans_iterations,
        random_state=clustering_random_seed,
    )

    labels = kmeans.fit_predict(embeddings_np)
    _embedding_cluster_cache[cache_key] = labels

    if len(_embedding_cluster_cache) > cluster_cache_size:
        _embedding_cluster_cache.popitem(last=False)

    return labels


def category_sample(
    logits: torch.Tensor,
    embedding: torch.nn.Embedding,
    k: int = 8,
    top_n: int = 50,
    temperature: float = None,  # Changed from hardcoded to None default
    filter_tokens: Optional[Set[int]] = None,
    force_include_ids: Optional[Set[int]] = None,
    study_data: Optional[Dict[str, Any]] = None,
    dcbs_config: Optional[DCBSConfig] = None,
    cache_config: Optional[CacheConfig] = None,
) -> int:
    """Performs semantically diverse sampling.
    
    Args:
        logits: Token logits
        embedding: Embedding layer
        k: Number of clusters
        top_n: Top tokens to consider
        temperature: Temperature (if None, uses config)
        filter_tokens: Allowed token IDs
        force_include_ids: Token IDs to always include
        study_data: Storage for metrics
        dcbs_config: DCBS config
        cache_config: Cache config
        
    Returns:
        Selected token ID
    """
    if dcbs_config is None:
        dcbs_config = DEFAULT_DCBS_CONFIG
        
    # Use config temperature if not explicitly provided
    curr_temperature = temperature if temperature is not None else dcbs_config["temperature"]
    min_tokens_for_clustering = dcbs_config["min_tokens_for_clustering"]
    
    if filter_tokens and len(filter_tokens) == 1:
        return list(filter_tokens)[0]

    if curr_temperature != 1.0:
        logits = logits / curr_temperature

    if filter_tokens:
        candidate_ids = []
        for idx in range(len(logits)):
            if idx in filter_tokens:
                candidate_ids.append(idx)

        if len(candidate_ids) <= min_tokens_for_clustering:
            probs = torch.softmax(logits[candidate_ids], dim=0)
            selected_idx = torch.multinomial(probs, 1).item()
            return candidate_ids[selected_idx]
    else:
        candidate_ids = _top_n_with_force(logits, top_n, force_include_ids)

    if (
        torch.isinf(logits[candidate_ids]).all()
        or torch.isnan(logits[candidate_ids]).any()
    ):
        if filter_tokens:
            return random.choice(list(filter_tokens))
        else:
            return random.randint(0, len(logits) - 1)

    candidate_ids_tensor = torch.tensor(candidate_ids, device=logits.device)
    candidate_logits = logits[candidate_ids_tensor]

    candidate_probs = torch.softmax(candidate_logits, dim=0)

    if len(candidate_ids) <= min_tokens_for_clustering:
        selected_idx = torch.multinomial(candidate_probs, 1).item()
        return candidate_ids[selected_idx]

    candidate_embeddings = get_cached_embeddings(
        candidate_ids_tensor, embedding, cache_config
    )

    if candidate_embeddings.shape[0] <= min_tokens_for_clustering:
        selected_idx = torch.multinomial(candidate_probs, 1).item()
        return candidate_ids[selected_idx]

    norm = torch.norm(candidate_embeddings, p=2, dim=1, keepdim=True)
    norm_embeddings = candidate_embeddings / norm.clamp(min=PROB_EPSILON)

    if k > len(candidate_ids):
        k = len(candidate_ids)

    labels = get_cached_clustering(norm_embeddings, k, dcbs_config, cache_config)

    clusters = [[] for _ in range(k)]
    for i, label in enumerate(labels):
        clusters[label].append(i)

    cluster_probs = []
    for cluster in clusters:
        if cluster:
            cluster_prob = candidate_probs[cluster].sum().item()
            cluster_probs.append(cluster_prob)
        else:
            cluster_probs.append(0.0)

    if sum(cluster_probs) == 0:
        if filter_tokens:
            return random.choice(list(filter_tokens))
        else:
            return random.randint(0, len(logits) - 1)

    cluster_probs = np.array(cluster_probs)
    cluster_probs = cluster_probs / cluster_probs.sum()
    selected_cluster_idx = np.random.choice(len(clusters), p=cluster_probs)

    cluster_token_indices = clusters[selected_cluster_idx]
    cluster_token_probs = candidate_probs[cluster_token_indices]

    if filter_tokens:
        valid_indices = []
        for i, token_idx in enumerate(cluster_token_indices):
            if candidate_ids[token_idx] in filter_tokens:
                valid_indices.append(i)

        if not valid_indices:
            return random.choice(list(filter_tokens))

        cluster_token_indices = [cluster_token_indices[i] for i in valid_indices]
        cluster_token_probs = cluster_token_probs[valid_indices]

    if study_data is not None:
        study_data["cluster_dists"] = cluster_probs.tolist()
        study_data["token_dists"] = candidate_probs.detach().cpu().numpy().tolist()
        study_data["token_ids"] = candidate_ids
        study_data["selected_cluster"] = int(selected_cluster_idx)
        study_data["cluster_assignments"] = labels.tolist()

    cluster_token_probs = cluster_token_probs / cluster_token_probs.sum()
    selected_in_cluster_idx = torch.multinomial(cluster_token_probs, 1).item()
    selected_token_idx = cluster_token_indices[selected_in_cluster_idx]

    return candidate_ids[selected_token_idx]
