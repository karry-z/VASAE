import pytest
import torch
from torch import nn

from vasae.models.online import attach_sae_embeddings, dtype_from_name
from vasae.models.sae import SAEConfig, SAEModel


@pytest.mark.parametrize(
    "name,dtype",
    [
        (None, None),
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
        ("float32", torch.float32),
    ],
)
def test_dtype_from_name(name, dtype):
    assert dtype_from_name(name) is dtype


def test_attach_sae_embeddings_ties_decoder():
    cfg = SAEConfig(dim_model=4, dim_sparse=10, tied_decoder=True)
    model = SAEModel(cfg)
    emb = nn.Embedding(10, 4)

    attach_sae_embeddings(model, emb, freeze_decoder=True)

    assert model.decoder.weight.requires_grad is False
    assert torch.allclose(model.decoder.weight, emb.weight.T)


def test_attach_sae_embeddings_attaches_anchor_embedding():
    cfg = SAEConfig(dim_model=4, dim_sparse=10, anchor_coeff=0.1)
    model = SAEModel(cfg)
    emb = nn.Embedding(10, 4)

    attach_sae_embeddings(model, emb, freeze_decoder=True)

    assert model._anchor_embedding is emb
