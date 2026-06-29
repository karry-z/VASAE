from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnsight import NNsight

from vasae.engine import IMetric, patch_and_forward


class VarianceExplained(IMetric):
    """Variance Explained = 1 - MSE(x, x_recon) / Var(x)."""

    def compute(self, context: Dict[str, Any]) -> Dict[str, float]:
        x: torch.Tensor = context["hidden_states"]
        x_recon: torch.Tensor = context["hidden_states_recon"]

        x_flat = x.reshape(-1, x.size(-1))
        xr_flat = x_recon.reshape(-1, x_recon.size(-1))

        mse = (x_flat - xr_flat).pow(2).sum()
        var = (x_flat - x_flat.mean(dim=0, keepdim=True)).pow(2).sum()

        ve = 1.0 - (mse / var.clamp(min=1e-8))
        return {"variance_explained": ve.item()}


def cross_entropy(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> float:
    """Compute mean cross-entropy loss over valid next-token positions."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    )
    loss = loss.view(shift_labels.shape)
    loss = (loss * shift_mask).sum() / shift_mask.sum().clamp(min=1)
    return loss.item()


class CELossRecovered(IMetric):
    """Loss Recovered = 1 - (CE_sae - CE_id) / (CE_zero - CE_id)."""

    def __init__(self, model: NNsight, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx

    def compute(self, context: Dict[str, Any]) -> Dict[str, float]:
        input_ids = context["input_ids"]
        attention_mask = context["attention_mask"]
        sae_model = context["sae_model"]

        logits_id = patch_and_forward(
            self.model, input_ids, attention_mask,
            self.layer_idx, lambda h: h,
        )
        ce_id = cross_entropy(logits_id, input_ids, attention_mask)

        logits_sae = patch_and_forward(
            self.model, input_ids, attention_mask,
            self.layer_idx, lambda h: sae_model(h.float()).hidden_states_recon.to(h.dtype),
        )
        ce_sae = cross_entropy(logits_sae, input_ids, attention_mask)

        logits_zero = patch_and_forward(
            self.model, input_ids, attention_mask,
            self.layer_idx, lambda h: torch.zeros_like(h),
        )
        ce_zero = cross_entropy(logits_zero, input_ids, attention_mask)

        loss_recovered = 1.0 - (ce_sae - ce_id) / (ce_zero - ce_id + 1e-8)

        return {
            "ce_id": ce_id,
            "ce_sae": ce_sae,
            "ce_zero": ce_zero,
            "loss_recovered": loss_recovered,
        }


class LogitLens:
    def __init__(self, unembed_layer: nn.Linear, ln=None):
        self.unembed_layer = unembed_layer
        self.ln = ln

    def unembed(self, activation: torch.Tensor) -> torch.Tensor:
        activation = activation.to(device=self.unembed_layer.weight.device, dtype=self.unembed_layer.weight.dtype)
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


class LogitLensAccuracy:
    def compute(self, reconstruct_tokens, tokens):
        reconstruct_tokens = np.array(reconstruct_tokens)
        tokens = np.array(tokens)
        correct = reconstruct_tokens == tokens
        return np.mean(correct).item()


class LogitLensMetric(IMetric):
    """Logit lens accuracy metric."""

    def __init__(self, logitlens: LogitLens, logitlens_acc: LogitLensAccuracy = None):
        self.logitlens = logitlens
        self.logitlens_acc = logitlens_acc or LogitLensAccuracy()

    def compute(self, context: Dict[str, Any]) -> Dict[str, float]:
        data = context["hidden_states"]
        decoded = context["hidden_states_recon"]

        data_ids = self.logitlens.top1(data)["token_ids"].cpu()
        recons_ids = self.logitlens.top1(decoded)["token_ids"].cpu()

        acc = self.logitlens_acc.compute(
            data_ids.flatten().tolist(),
            recons_ids.flatten().tolist(),
        )

        return {"logitlens_acc": acc}
