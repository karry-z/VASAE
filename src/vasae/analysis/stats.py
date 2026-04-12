"""Tensor summary statistics utilities."""

import torch


def summarize_tensor(t: torch.Tensor) -> dict[str, float]:
    """Compute summary statistics for a 1-D tensor.

    Returns a dict with keys: mean, std, median, min, max, p5, p25, p75, p95.
    """
    t = t.float()
    return {
        "mean": t.mean().item(),
        "std": t.std().item(),
        "median": t.median().item(),
        "min": t.min().item(),
        "max": t.max().item(),
        "p5": t.quantile(0.05).item(),
        "p25": t.quantile(0.25).item(),
        "p75": t.quantile(0.75).item(),
        "p95": t.quantile(0.95).item(),
    }
