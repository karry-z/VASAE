"""Geometric alignment and logit attribution utilities."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vasae.losses.cosine_sim import chunked_cosine_sim


@dataclass
class GeometricAlignmentResult:
    """Result of batched cosine similarity between features and reference vectors."""

    max_sims: torch.Tensor  # (n_features,)
    topk_sims: torch.Tensor  # (n_features, top_k)
    topk_indices: torch.Tensor  # (n_features, top_k) long


@torch.no_grad()
def compute_geometric_alignment(
    features: torch.Tensor,
    references: torch.Tensor,
    top_k: int = 1,
    chunk_size: int = 512,
    device: torch.device | str = "cpu",
) -> GeometricAlignmentResult:
    """Compute batched cosine similarity between feature vectors and reference vectors.

    Args:
        features: (n_features, dim) -- e.g. ``decoder.weight.data.T``.
        references: (n_refs, dim) -- e.g. embedding weight ``W_E``.
        top_k: number of top matches to return per feature.
        chunk_size: chunk size for the batched matmul.
        device: computation device.

    Returns:
        GeometricAlignmentResult with max_sims, topk_sims, topk_indices
        (all on CPU).
    """
    device = torch.device(device)
    features = features.to(device).float()
    references = references.to(device).float()

    def _reduce(sim: torch.Tensor) -> torch.Tensor:
        # Pack max_sim, topk_sims, topk_indices into one tensor per chunk.
        # Indices are cast to float for packing (safe for vocab ≤ 2^23 ≈ 8M).
        ms = sim.max(dim=1)[0].unsqueeze(1)  # (chunk, 1)
        ts, ti = sim.topk(top_k, dim=1)  # (chunk, top_k) each
        return torch.cat([ms, ts, ti.float()], dim=1)

    packed = chunked_cosine_sim(features, references, _reduce, chunk_size=chunk_size)
    max_sims = packed[:, 0]
    topk_sims = packed[:, 1 : 1 + top_k]
    topk_indices = packed[:, 1 + top_k :].long()

    return GeometricAlignmentResult(
        max_sims=max_sims.cpu(),
        topk_sims=topk_sims.cpu(),
        topk_indices=topk_indices.cpu(),
    )
