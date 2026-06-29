import pytest
import torch
import torch.nn as nn

from vasae.models.sae import SAEConfig, SAEModel, SAEOutput


DIM_INPUT = 16
DIM_SPARSE = 64
BATCH = 8


class TestSAEConfig:
    def test_defaults_are_learnable_topk(self):
        cfg = SAEConfig()
        assert cfg.dim_input == 768
        assert cfg.dim_sparse == 8192
        assert cfg.k == 64
        assert cfg.decoder_mode == "learnable"
        assert cfg.anchor_coeff == 0.0

    def test_topk_requires_positive_k(self):
        with pytest.raises(ValueError, match="k must be > 0"):
            SAEConfig(k=0)

    def test_invalid_decoder_mode(self):
        with pytest.raises(ValueError, match="decoder_mode"):
            SAEConfig(decoder_mode="unknown")

    def test_invalid_anchor_mode(self):
        with pytest.raises(ValueError, match="anchor_mode"):
            SAEConfig(anchor_mode="unknown")


def _make_model(**kwargs) -> SAEModel:
    cfg = SAEConfig(
        dim_input=DIM_INPUT,
        dim_sparse=DIM_SPARSE,
        k=4,
        **kwargs,
    )
    return SAEModel(cfg)


class TestSAEForward:
    def test_output_type_and_shapes(self):
        model = _make_model()
        x = torch.randn(BATCH, DIM_INPUT)
        out = model(x)
        assert isinstance(out, SAEOutput)
        assert out.hidden_states_recon.shape == (BATCH, DIM_INPUT)
        assert out.sparse_activations.shape == (BATCH, DIM_SPARSE)
        assert out.loss.ndim == 0

    def test_topk_sparse_activations(self):
        model = _make_model()
        x = torch.randn(BATCH, DIM_INPUT)
        out = model(x)
        assert ((out.sparse_activations != 0).sum(dim=-1) <= 4).all()

    def test_backward(self):
        model = _make_model()
        x = torch.randn(BATCH, DIM_INPUT)
        out = model(x)
        out.loss.backward()
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)

    def test_pre_activations_optional(self):
        model = _make_model()
        x = torch.randn(BATCH, DIM_INPUT)
        out_no = model(x, output_pre_activations=False)
        out_yes = model(x, output_pre_activations=True)
        assert out_no.pre_activations is None
        assert out_yes.pre_activations.shape == (BATCH, DIM_SPARSE)

    def test_loss_per_sample(self):
        model = _make_model()
        x = torch.randn(BATCH, DIM_INPUT)
        out = model(x, output_loss_per_sample=True)
        assert out.loss_per_sample.shape == (BATCH,)

    def test_3d_input(self):
        model = _make_model()
        x = torch.randn(2, 10, DIM_INPUT)
        out = model(x)
        assert out.hidden_states_recon.shape == (2, 10, DIM_INPUT)
        assert out.sparse_activations.shape == (2, 10, DIM_SPARSE)

    def test_tuple_output(self):
        model = _make_model()
        x = torch.randn(BATCH, DIM_INPUT)
        out = model(x, return_dict=False)
        assert isinstance(out, tuple)
        assert out[0].ndim == 0


class TestVocabAnchor:
    def test_plain_baseline_anchor_coeff_zero(self):
        model = _make_model(anchor_coeff=0.0)
        model.attach_vocab_anchor(nn.Embedding(100, DIM_INPUT))
        x = torch.randn(BATCH, DIM_INPUT)
        out = model(x)
        assert out.loss_anchor is None

    @pytest.mark.parametrize("anchor_mode", ["nearest", "logsumexp", "softmax"])
    def test_anchor_loss_present_when_enabled(self, anchor_mode):
        model = _make_model(
            anchor_coeff=0.1,
            anchor_mode=anchor_mode,
            anchor_topk=5,
        )
        model.attach_vocab_anchor(nn.Embedding(100, DIM_INPUT))
        x = torch.randn(BATCH, DIM_INPUT)
        out = model(x)
        assert out.loss_anchor is not None
        assert out.loss.requires_grad

    def test_anchor_requires_matching_embedding_dim(self):
        model = _make_model(anchor_coeff=0.1)
        with pytest.raises(ValueError, match="dim_input"):
            model.attach_vocab_anchor(nn.Embedding(100, DIM_INPUT + 1))


class TestHardTiedBaseline:
    def test_attach_tied_decoder_embedding(self):
        vocab_size = DIM_SPARSE
        cfg = SAEConfig(
            dim_input=DIM_INPUT,
            dim_sparse=vocab_size,
            k=4,
            decoder_mode="hard_tied_baseline",
        )
        model = SAEModel(cfg)
        emb = nn.Embedding(vocab_size, DIM_INPUT)
        model.attach_tied_decoder_embedding(emb, freeze=True)

        assert model.decoder.weight.requires_grad is False
        assert torch.allclose(model.decoder.weight, emb.weight.T)

    def test_tied_decoder_requires_baseline_mode(self):
        model = _make_model()
        with pytest.raises(ValueError, match="hard_tied_baseline"):
            model.attach_tied_decoder_embedding(nn.Embedding(DIM_SPARSE, DIM_INPUT))

    def test_forward_with_hard_tied_baseline(self):
        vocab_size = DIM_SPARSE
        cfg = SAEConfig(
            dim_input=DIM_INPUT,
            dim_sparse=vocab_size,
            k=4,
            decoder_mode="hard_tied_baseline",
        )
        model = SAEModel(cfg)
        model.attach_tied_decoder_embedding(nn.Embedding(vocab_size, DIM_INPUT))

        x = torch.randn(BATCH, DIM_INPUT)
        out = model(x)
        assert out.hidden_states_recon.shape == (BATCH, DIM_INPUT)
