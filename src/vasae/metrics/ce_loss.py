from typing import Any, Dict

import torch
import torch.nn.functional as F
from nnsight import NNsight

from vasae.engine.intervention import patch_and_forward
from vasae.metrics.base import IMetric


def cross_entropy(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> float:
    """Compute mean cross-entropy loss over valid (non-padded) next-token positions."""
    # Shift: predict next token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    )
    loss = loss.view(shift_labels.shape)
    # Mask out padding
    loss = (loss * shift_mask).sum() / shift_mask.sum().clamp(min=1)
    return loss.item()


class CELossRecovered(IMetric):
    """Loss Recovered = 1 - (CE_sae - CE_id) / (CE_zero - CE_id)

    Measures how much of the model's performance the SAE reconstruction preserves.

    Expects context keys:
    - "input_ids": token IDs
    - "attention_mask": attention mask
    - "sae_model": SAE model instance
    """

    def __init__(self, model: NNsight, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx

    def compute(self, context: Dict[str, Any]) -> Dict[str, float]:
        input_ids = context["input_ids"]
        attention_mask = context["attention_mask"]
        sae_model = context["sae_model"]

        # CE(id): no intervention
        logits_id = patch_and_forward(
            self.model, input_ids, attention_mask,
            self.layer_idx, lambda h: h,
        )
        ce_id = cross_entropy(logits_id, input_ids, attention_mask)

        # CE(sae): SAE reconstruction
        logits_sae = patch_and_forward(
            self.model, input_ids, attention_mask,
            self.layer_idx, lambda h: sae_model(h).hidden_states_recon,
        )
        ce_sae = cross_entropy(logits_sae, input_ids, attention_mask)

        # CE(zero): zero ablation
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
