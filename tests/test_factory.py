"""Unit tests for model-agnostic factory helpers."""

import pytest
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from vasae.models.factory import get_embedding, get_layers, get_lm_head


@pytest.fixture(scope="session")
def gpt2():
    cfg = AutoConfig.from_pretrained("gpt2")
    cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.vocab_size = 2, 16, 2, 32
    return AutoModelForCausalLM.from_config(cfg)


@pytest.fixture(scope="session")
def llama():
    cfg = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B")
    cfg.num_hidden_layers = 2
    cfg.hidden_size = 16
    cfg.num_attention_heads = 2
    cfg.num_key_value_heads = 2
    cfg.intermediate_size = 32
    cfg.vocab_size = 32
    return AutoModelForCausalLM.from_config(cfg)


class TestGetLayers:
    def test_gpt2(self, gpt2):
        assert len(get_layers(gpt2)) == 2

    def test_llama(self, llama):
        assert len(get_layers(llama)) == 2


class TestGetEmbedding:
    def test_gpt2(self, gpt2):
        emb = get_embedding(gpt2)
        assert isinstance(emb, nn.Embedding)
        assert emb.num_embeddings == 32

    def test_llama(self, llama):
        emb = get_embedding(llama)
        assert isinstance(emb, nn.Embedding)
        assert emb.num_embeddings == 32


class TestGetLmHead:
    def test_gpt2(self, gpt2):
        head = get_lm_head(gpt2)
        assert isinstance(head, nn.Linear)
        assert head.out_features == 32

    def test_llama(self, llama):
        head = get_lm_head(llama)
        assert isinstance(head, nn.Linear)
        assert head.out_features == 32


@pytest.mark.parametrize(
    "fn, msg",
    [
        (get_layers, "Cannot find transformer layers"),
        (get_embedding, "Cannot find embedding"),
        (get_lm_head, "Cannot find lm_head"),
    ],
)
def test_unsupported_raises(fn, msg):
    with pytest.raises(ValueError, match=msg):
        fn(object())
