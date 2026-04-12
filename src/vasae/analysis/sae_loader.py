"""Convenience loaders for SAE models in analysis scripts."""

from pathlib import Path

import torch

from vasae.models.sae import SAEModel


def load_sae_for_analysis(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> SAEModel:
    """Load an SAE from a pretrained directory, ready for analysis.

    Loads via ``from_pretrained``, moves to *device*, and sets eval mode.

    Args:
        path: path to HF-format SAE directory.
        device: target device.

    Returns:
        ``SAEModel`` in eval mode on the specified device.
    """
    sae = SAEModel.from_pretrained(str(path)).eval()
    return sae.to(device)


def get_decoder_features(sae: SAEModel) -> torch.Tensor:
    """Return decoder features as ``(n_features, dim_input)`` tensor.

    Equivalent to ``sae.decoder.weight.data.T``.
    """
    return sae.decoder.weight.data.T
