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
    batch_size: int = 512,
    device: torch.device | str = "cpu",
) -> GeometricAlignmentResult:
    """Compute batched cosine similarity between feature vectors and reference vectors.

    Args:
        features: (n_features, dim) -- e.g. ``decoder.weight.data.T``.
        references: (n_refs, dim) -- e.g. embedding weight ``W_E``.
        top_k: number of top matches to return per feature.
        batch_size: chunk size for the batched matmul.
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
        ms = sim.max(dim=1)[0].unsqueeze(1)        # (chunk, 1)
        ts, ti = sim.topk(top_k, dim=1)            # (chunk, top_k) each
        return torch.cat([ms, ts, ti.float()], dim=1)

    packed = chunked_cosine_sim(features, references, _reduce, chunk_size=batch_size)
    max_sims = packed[:, 0]
    topk_sims = packed[:, 1 : 1 + top_k]
    topk_indices = packed[:, 1 + top_k :].long()

    return GeometricAlignmentResult(
        max_sims=max_sims.cpu(),
        topk_sims=topk_sims.cpu(),
        topk_indices=topk_indices.cpu(),
    )


@dataclass
class LogitAttributionResult:
    """Result of logit attribution analysis (la_i = W_U @ d_i)."""

    entropy: torch.Tensor  # (n_features,)
    max_mean_ratio: torch.Tensor  # (n_features,)
    top1_concentration: torch.Tensor  # (n_features,)
    top5_concentration: torch.Tensor  # (n_features,)
    max_logit: torch.Tensor  # (n_features,)
    max_token_id: torch.Tensor  # (n_features,) long
    topk_vals: torch.Tensor  # (n_features, top_k)
    topk_tokens: torch.Tensor  # (n_features, top_k) long


@torch.no_grad()
def compute_logit_attribution(
    features: torch.Tensor,
    W_U: torch.Tensor,
    top_k: int = 5,
    batch_size: int = 1024,
    device: torch.device | str = "cpu",
) -> LogitAttributionResult:
    """Compute logit attribution ``la_i = features @ W_U.T`` and derived metrics.

    Args:
        features: (n_features, dim_input) -- decoder features.
        W_U: (vocab_size, dim_input) -- unembedding weight matrix.
        top_k: number of top tokens to track per feature.
        batch_size: chunk size.
        device: computation device.

    Returns:
        LogitAttributionResult with per-feature entropy, concentration, etc.
        (all tensors on CPU).
    """
    device = torch.device(device)
    features = features.to(device)
    W_U = W_U.to(device)
    n_features = features.size(0)

    all_entropy = []
    all_max_mean_ratio = []
    all_top1_conc = []
    all_top5_conc = []
    all_max_logit = []
    all_max_token_id = []
    all_topk_vals = []
    all_topk_tokens = []

    for start in range(0, n_features, batch_size):
        end = min(start + batch_size, n_features)
        batch_d = features[start:end]

        la = batch_d @ W_U.T  # (batch, vocab_size)

        probs = F.softmax(la, dim=1)
        log_probs = F.log_softmax(la, dim=1)
        entropy = -(probs * log_probs).sum(dim=1)

        la_abs = la.abs()
        max_val, _ = la_abs.max(dim=1)
        mean_val = la_abs.mean(dim=1)
        max_mean_ratio = max_val / (mean_val + 1e-10)

        top5_vals, _ = probs.topk(5, dim=1)
        top1_conc = top5_vals[:, 0]
        top5_conc = top5_vals.sum(dim=1)

        real_max_val, real_max_idx = la.max(dim=1)

        topk_v, topk_i = la.topk(top_k, dim=1)

        all_entropy.append(entropy.cpu())
        all_max_mean_ratio.append(max_mean_ratio.cpu())
        all_top1_conc.append(top1_conc.cpu())
        all_top5_conc.append(top5_conc.cpu())
        all_max_logit.append(real_max_val.cpu())
        all_max_token_id.append(real_max_idx.cpu())
        all_topk_vals.append(topk_v.cpu())
        all_topk_tokens.append(topk_i.cpu())

    return LogitAttributionResult(
        entropy=torch.cat(all_entropy),
        max_mean_ratio=torch.cat(all_max_mean_ratio),
        top1_concentration=torch.cat(all_top1_conc),
        top5_concentration=torch.cat(all_top5_conc),
        max_logit=torch.cat(all_max_logit),
        max_token_id=torch.cat(all_max_token_id),
        topk_vals=torch.cat(all_topk_vals),
        topk_tokens=torch.cat(all_topk_tokens),
    )
