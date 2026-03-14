from typing import Any, Dict

import torch

from vasae.metrics.base import IMetric


class VarianceExplained(IMetric):
    """Variance Explained = 1 - MSE(x, x_recon) / Var(x)

    Measures how much of the input variance is captured by the reconstruction.
    Value of 1.0 means perfect reconstruction; 0.0 means the reconstruction
    is no better than predicting the mean.

    Expects context keys:
    - "hidden_states": original activations  (*, D)
    - "hidden_states_recon": SAE reconstruction  (*, D)
    """

    def compute(self, context: Dict[str, Any]) -> Dict[str, float]:
        x = context["hidden_states"]
        x_recon = context["hidden_states_recon"]

        # Flatten to (N, D)
        x_flat = x.reshape(-1, x.size(-1))
        xr_flat = x_recon.reshape(-1, x_recon.size(-1))

        mse = (x_flat - xr_flat).pow(2).sum()
        var = (x_flat - x_flat.mean(dim=0, keepdim=True)).pow(2).sum()

        ve = 1.0 - (mse / var.clamp(min=1e-8))
        return {"variance_explained": ve.item()}
