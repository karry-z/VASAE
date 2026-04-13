from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from vasae.metrics.base import IMetric


class LogitLens:
    def __init__(self, unembed_layer: nn.Linear, ln=None):
        self.unembed_layer = unembed_layer
        self.ln = ln

    def unembed(self, activation: torch.Tensor) -> torch.Tensor:
        activation = activation.to(
            device=self.unembed_layer.weight.device,
            dtype=self.unembed_layer.weight.dtype,
        )
        if self.ln is not None:
            activation = self.ln(activation)
        with torch.no_grad():
            logits = self.unembed_layer(activation)
        return logits

    def top1(self, activation: torch.Tensor) -> Dict:
        logits = self.unembed(activation)
        probs = logits.softmax(dim=-1)
        token_probs, token_ids = probs.max(dim=-1)
        return {
            "token_ids": token_ids,
            "token_probs": token_probs,
            "probs": probs,
        }


def compute_token_prediction_acc(reconstruct_tokens, tokens):
    reconstruct_tokens = np.array(reconstruct_tokens)
    tokens = np.array(tokens)
    correct = reconstruct_tokens == tokens
    return np.mean(correct).item()


class LogitLensAccMetric(IMetric):
    """Logit lens accuracy metric.

    Expects context keys:
    - "hidden_states": original activations
    - "hidden_states_recon": SAE reconstruction
    """

    def __init__(self, logitlens: LogitLens):
        self.logitlens = logitlens

    def compute(self, context: Dict[str, Any]) -> Dict[str, float]:
        data = context["hidden_states"]
        decoded = context["hidden_states_recon"]

        data_ids = self.logitlens.top1(data)["token_ids"].cpu()
        recons_ids = self.logitlens.top1(decoded)["token_ids"].cpu()

        acc = compute_token_prediction_acc(
            data_ids.flatten().tolist(),
            recons_ids.flatten().tolist(),
        )

        return {"logitlens_acc": acc}
