"""Alive-feature alignment helpers."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AliveAlignmentStats:
    n_features: int
    n_alive: int
    n_aligned: int
    n_alive_aligned: int
    alive_alignment_rate: float
    dead_rate: float


def compute_alive_alignment_stats(
    alive_mask: torch.Tensor,
    alignment_scores: torch.Tensor,
    threshold: float = 0.8,
) -> AliveAlignmentStats:
    """Compute alignment rate among features that fired at least once.

    A feature is aligned when its alignment score is greater than or equal to
    ``threshold``. Dead features are excluded from the rate denominator.
    """
    if alive_mask.ndim != 1 or alignment_scores.ndim != 1:
        raise ValueError("alive_mask and alignment_scores must be 1D tensors")
    if alive_mask.numel() != alignment_scores.numel():
        raise ValueError("alive_mask and alignment_scores must have the same length")

    alive = alive_mask.bool().cpu()
    aligned = (alignment_scores.cpu() >= threshold).bool()
    n_features = int(alive.numel())
    n_alive = int(alive.sum().item())
    n_aligned = int(aligned.sum().item())
    n_alive_aligned = int((alive & aligned).sum().item())

    alive_alignment_rate = n_alive_aligned / n_alive if n_alive > 0 else 0.0
    dead_rate = 1.0 - (n_alive / n_features) if n_features > 0 else 0.0

    return AliveAlignmentStats(
        n_features=n_features,
        n_alive=n_alive,
        n_aligned=n_aligned,
        n_alive_aligned=n_alive_aligned,
        alive_alignment_rate=alive_alignment_rate,
        dead_rate=dead_rate,
    )
