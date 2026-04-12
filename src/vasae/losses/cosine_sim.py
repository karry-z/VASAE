"""Chunked cosine similarity utilities.

Computes pairwise cosine similarity between two sets of vectors in chunks
to avoid materialising the full (n × m) similarity matrix in memory.
"""

from typing import Callable

import torch
import torch.nn.functional as F


def chunked_cosine_sim(
    features: torch.Tensor,
    references: torch.Tensor,
    reduce_fn: Callable[[torch.Tensor], torch.Tensor],
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Compute cosine similarity between *features* and *references*, reduced per row.

    For each chunk of rows in *features*, the full (chunk × n_refs) similarity
    matrix is computed, passed through *reduce_fn*, and then discarded so only
    the reduced result is kept in memory.

    Args:
        features: (n_features, dim) — vectors to query.
        references: (n_refs, dim) — reference set (e.g. token embeddings).
        reduce_fn: Called on each (chunk_size, n_refs) similarity sub-matrix.
            Must return a tensor whose first dimension equals the chunk size.
        chunk_size: Number of feature rows per chunk.

    Returns:
        ``torch.cat`` of all per-chunk *reduce_fn* outputs.
    """
    f_norm = F.normalize(features, dim=1)
    r_norm = F.normalize(references.to(f_norm.dtype), dim=1)

    parts = []
    for i in range(0, f_norm.size(0), chunk_size):
        sim = f_norm[i : i + chunk_size] @ r_norm.T
        parts.append(reduce_fn(sim))

    return torch.cat(parts)
