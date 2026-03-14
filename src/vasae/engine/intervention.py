from typing import Callable

import torch
from nnsight import NNsight



def _get_layer_proxy(model: NNsight, layer_idx: int):
    """Resolve the layer proxy inside an nnsight trace context.

    Must access through the nnsight model (not model._model) to get
    proxy objects that support .output inside trace().
    """
    m = model._model
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        return model.transformer.h[layer_idx]
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(m, "model") and hasattr(m.model, "decoder") and hasattr(m.model.decoder, "layers"):
        return model.model.decoder.layers[layer_idx]
    if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
        return model.gpt_neox.layers[layer_idx]
    raise ValueError(
        f"Cannot find transformer layers for {type(m).__name__}. "
        "Add support in _get_layer_proxy()."
    )


def extract_activations(
    model: NNsight, input_ids: torch.Tensor, layer_idx: int
) -> torch.Tensor:
    """Extract activations from a specific transformer layer (model-agnostic)."""
    with model.trace(input_ids):
        layer = _get_layer_proxy(model, layer_idx)
        h = layer.output[0].save()
    return h


def patch_and_forward(
    model: NNsight,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int,
    intervention_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Patch activations at a specific layer and return final logits (model-agnostic)."""
    with model.trace(input_ids, attention_mask=attention_mask):
        layer = _get_layer_proxy(model, layer_idx)
        h = layer.output[0]
        layer.output[0] = intervention_fn(h)
        logits = model.output.logits.save()
    return logits
