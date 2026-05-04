import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nnsight import NNsight


@dataclass
class OnlineLLMContext:
    llm: Any
    tokenizer: Any
    nn_model: "NNsight"
    embedding: Any
    lm_head: Any
    n_layers: int
    dim_model: int
    vocab_size: int


def dtype_from_name(dtype_name: str | None):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name is None:
        return None
    return dtype_map[dtype_name]


def load_online_llm(
    model_name: str,
    device: str,
    dtype_name: str | None,
    layer_idx: int,
) -> OnlineLLMContext:
    from nnsight import NNsight  # nnsight ~10s

    from vasae.models.factory import get_embedding, get_layers, get_lm_head, load_model

    logger.info(f"Loading {model_name}...")

    llm, tokenizer = load_model(
        model_name,
        device=device,
        dtype=dtype_from_name(dtype_name),
    )
    nn_model = NNsight(llm)
    layers = get_layers(llm)
    embedding = get_embedding(llm)
    lm_head = get_lm_head(llm)
    n_layers = len(layers)

    dim_model = embedding.weight.size(1)
    vocab_size = embedding.weight.size(0)
    logger.info(
        f"Model: {type(llm).__name__}, dim={dim_model}, vocab={vocab_size}, "
        f"layers={n_layers}, using layer {layer_idx}"
    )

    return OnlineLLMContext(
        llm=llm,
        tokenizer=tokenizer,
        nn_model=nn_model,
        embedding=embedding,
        lm_head=lm_head,
        n_layers=n_layers,
        dim_model=dim_model,
        vocab_size=vocab_size,
    )


def attach_sae_embeddings(
    sae_model,
    embedding,
    freeze_decoder: bool,
):
    if sae_model.config.tied_decoder:
        sae_model.attach_embedding(embedding, freeze=freeze_decoder)
    if sae_model.config.anchor_coeff > 0:
        sae_model.attach_anchor_embedding(embedding)
    return sae_model
