"""PyTorch forward-hook utilities for activation patching / ablation.

These use raw ``register_forward_hook`` (no nnsight).  For nnsight-based
intervention see ``vasae.engine.intervention``.
"""

from typing import Callable

import torch
import torch.nn as nn


def make_intervention_hook(
    intervention_fn: Callable[[torch.Tensor], torch.Tensor],
) -> Callable:
    """Create a forward hook that applies *intervention_fn* to the hidden state.

    Handles both tuple outputs (e.g. GPT-2: ``(hidden_states, presents, ...)``)
    and plain tensor outputs transparently.

    Args:
        intervention_fn: ``(hidden_states) -> modified_hidden_states``.

    Returns:
        A hook function suitable for ``module.register_forward_hook()``.
    """

    def hook(module, input, output):  # noqa: A002
        if isinstance(output, tuple):
            h = output[0]
            return (intervention_fn(h),) + output[1:]
        return intervention_fn(output)

    return hook


@torch.no_grad()
def run_with_hook(
    model: nn.Module,
    target_layer: nn.Module,
    intervention_fn: Callable[[torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run a forward pass with *intervention_fn* applied at *target_layer*.

    Args:
        model: the full language model.
        target_layer: the specific transformer layer module to hook.
        intervention_fn: ``(hidden_states) -> modified_hidden_states``.
        input_ids: ``(B, S)`` token ids.
        attention_mask: ``(B, S)`` attention mask.

    Returns:
        logits: ``(B, S, vocab_size)`` detached.
    """
    handle = target_layer.register_forward_hook(make_intervention_hook(intervention_fn))
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits.detach()
    finally:
        handle.remove()
