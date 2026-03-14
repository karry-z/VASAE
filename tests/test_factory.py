"""Unit tests for model-agnostic factory helpers."""

import pytest
import torch
import torch.nn as nn

from vasae.models.factory import get_layers, get_embedding, get_lm_head


class FakeGPT2:
    """Mimics GPT2LMHeadModel structure."""
    def __init__(self):
        self.transformer = type("T", (), {
            "h": nn.ModuleList([nn.Linear(4, 4) for _ in range(3)]),
            "wte": nn.Embedding(10, 4),
        })()
        self.lm_head = nn.Linear(4, 10, bias=False)


class FakeLlama:
    """Mimics LlamaForCausalLM structure."""
    def __init__(self):
        self.model = type("M", (), {
            "layers": nn.ModuleList([nn.Linear(4, 4) for _ in range(5)]),
            "embed_tokens": nn.Embedding(20, 4),
        })()
        self.lm_head = nn.Linear(4, 20, bias=False)


class FakeOPT:
    """Mimics OPTForCausalLM structure."""
    def __init__(self):
        self.model = type("M", (), {
            "decoder": type("D", (), {
                "layers": nn.ModuleList([nn.Linear(4, 4) for _ in range(2)]),
                "embed_tokens": nn.Embedding(15, 4),
            })(),
        })()
        self.lm_head = nn.Linear(4, 15, bias=False)


class FakeNeoX:
    """Mimics GPTNeoXForCausalLM structure."""
    def __init__(self):
        self.gpt_neox = type("N", (), {
            "layers": nn.ModuleList([nn.Linear(4, 4) for _ in range(4)]),
            "embed_in": nn.Embedding(12, 4),
        })()
        self.embed_out = nn.Linear(4, 12, bias=False)


class TestGetLayers:
    def test_gpt2(self):
        m = FakeGPT2()
        layers = get_layers(m)
        assert len(layers) == 3

    def test_llama(self):
        m = FakeLlama()
        layers = get_layers(m)
        assert len(layers) == 5

    def test_opt(self):
        m = FakeOPT()
        layers = get_layers(m)
        assert len(layers) == 2

    def test_neox(self):
        m = FakeNeoX()
        layers = get_layers(m)
        assert len(layers) == 4

    def test_unsupported(self):
        with pytest.raises(ValueError, match="Cannot find transformer layers"):
            get_layers(object())


class TestGetEmbedding:
    def test_gpt2(self):
        m = FakeGPT2()
        emb = get_embedding(m)
        assert isinstance(emb, nn.Embedding)
        assert emb.num_embeddings == 10

    def test_llama(self):
        m = FakeLlama()
        emb = get_embedding(m)
        assert emb.num_embeddings == 20

    def test_opt(self):
        m = FakeOPT()
        emb = get_embedding(m)
        assert emb.num_embeddings == 15

    def test_neox(self):
        m = FakeNeoX()
        emb = get_embedding(m)
        assert emb.num_embeddings == 12

    def test_unsupported(self):
        with pytest.raises(ValueError, match="Cannot find embedding"):
            get_embedding(object())


class TestGetLmHead:
    def test_lm_head(self):
        m = FakeGPT2()
        head = get_lm_head(m)
        assert isinstance(head, nn.Linear)

    def test_embed_out(self):
        m = FakeNeoX()
        head = get_lm_head(m)
        assert isinstance(head, nn.Linear)
        assert head.out_features == 12

    def test_unsupported(self):
        with pytest.raises(ValueError, match="Cannot find lm_head"):
            get_lm_head(object())
