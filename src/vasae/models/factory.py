from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(
    model_name: str, device: str = "cuda", dtype=None
) -> Tuple[nn.Module, AutoTokenizer]:
    """Load any HuggingFace causal LM and its tokenizer.

    Returns:
        (model, tokenizer)
    """
    kwargs = {}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model: nn.Module = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tokenizer


def get_layers(model: nn.Module) -> nn.ModuleList:
    """Return the transformer layer list for any supported HF causal LM.

    Supports:
    - GPT-2, GPT-Neo, GPT-J: model.transformer.h
    - LLaMA, Mistral, Qwen, Phi, Gemma: model.model.layers
    - OPT: model.model.decoder.layers
    """
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if (
        hasattr(model, "model")
        and hasattr(model.model, "decoder")
        and hasattr(model.model.decoder, "layers")
    ):
        return model.model.decoder.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError(
        f"Cannot find transformer layers for {type(model).__name__}. "
        "Add support in vasae.models.factory.get_layers()."
    )


def get_embedding(model: nn.Module) -> nn.Embedding:
    """Return the token embedding layer (W_E) for any supported HF causal LM."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    if (
        hasattr(model, "model")
        and hasattr(model.model, "decoder")
        and hasattr(model.model.decoder, "embed_tokens")
    ):
        return model.model.decoder.embed_tokens
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "embed_in"):
        return model.gpt_neox.embed_in
    raise ValueError(
        f"Cannot find embedding layer for {type(model).__name__}. "
        "Add support in vasae.models.factory.get_embedding()."
    )


def get_lm_head(model: nn.Module) -> nn.Linear:
    """Return the language model head (unembedding) for any supported HF causal LM."""
    if hasattr(model, "lm_head"):
        return model.lm_head
    if hasattr(model, "embed_out"):
        return model.embed_out
    raise ValueError(
        f"Cannot find lm_head for {type(model).__name__}. "
        "Add support in vasae.models.factory.get_lm_head()."
    )


# ---------------------------------------------------------------------------
# Legacy offline helpers (used by offline training scripts)
# ---------------------------------------------------------------------------
@dataclass
class BlackBoxModelConfig:
    name: str = "gpt2"
    dir: Path | None = None


def load_unembedding_layer(cfg: BlackBoxModelConfig) -> nn.Linear:
    return torch.load(cfg.dir / "unemb.pth", weights_only=False)


def load_embedding_layer(cfg: BlackBoxModelConfig) -> nn.Embedding:
    return torch.load(cfg.dir / "emb.pth", weights_only=False)


def get_blackbox_model(model_name, device):
    """Legacy helper — prefer load_model() for new code."""
    return load_model(model_name, device=device)
