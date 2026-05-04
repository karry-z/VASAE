from typing import Any, Dict

import torch

from vasae.metrics.base import IMetric


class ActivityStats(IMetric):
    """Dataset-level SAE feature activity statistics.

    Tracks feature activations across all batches in one metric pass. This is
    stateful because dead feature rate is a global statistic, not a per-batch
    average.

    Expects context keys:
    - "sparse_activations": sparse SAE activations  (N, D)
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.feature_counts: torch.Tensor | None = None
        self.l0_sum = 0.0
        self.n_samples = 0

    def compute(self, context: Dict[str, Any]) -> Dict[str, float]:
        z: torch.Tensor = context["sparse_activations"]
        z = z.reshape(-1, z.size(-1))

        if self.feature_counts is None:
            self.feature_counts = torch.zeros(z.size(-1), device=z.device)

        nonzero = z != 0
        self.feature_counts += nonzero.sum(dim=0)
        self.l0_sum += nonzero.sum(dim=1).sum().item()
        self.n_samples += z.size(0)
        return {}

    def finalize(self) -> Dict[str, float]:
        if self.feature_counts is None or self.n_samples == 0:
            return {
                "dead_rate": 0.0,
                "l0": 0.0,
                "n_samples": 0,
                "n_alive": 0,
                "alive_features": [],
            }

        dead_rate = (self.feature_counts == 0).float().mean().item()
        l0 = self.l0_sum / self.n_samples
        alive_features = (self.feature_counts > 0).nonzero(as_tuple=True)[0].tolist()
        return {
            "dead_rate": dead_rate,
            "l0": l0,
            "n_samples": self.n_samples,
            "n_alive": len(alive_features),
            "alive_features": alive_features,
        }
