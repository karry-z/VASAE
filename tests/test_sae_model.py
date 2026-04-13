"""Unit tests for SAEConfig + SAEModel."""

import pytest
import torch
import torch.nn as nn

from vasae.models.sae import SAEConfig, SAEModel, SAEOutput


# ---------------------------------------------------------------------------
# SAEConfig validation
# ---------------------------------------------------------------------------
class TestSAEConfig:
    def test_invalid_encoder_type(self):
        with pytest.raises(ValueError, match="encoder_type"):
            SAEConfig(encoder_type="transformer")

    def test_invalid_sparsity_type(self):
        with pytest.raises(ValueError, match="sparsity_type"):
            SAEConfig(sparsity_type="random")

    def test_topk_requires_positive_k(self):
        with pytest.raises(ValueError, match="k must be > 0"):
            SAEConfig(sparsity_type="topk", k=0)

    def test_topk_rejects_l1(self):
        with pytest.raises(ValueError, match="Do not use L1"):
            SAEConfig(sparsity_type="topk", k=4, l1_coeff=0.1)

    def test_invalid_anchor_mode(self):
        with pytest.raises(ValueError, match="anchor_mode"):
            SAEConfig(anchor_mode="unknown")

    def test_valid_topk_config(self):
        cfg = SAEConfig(sparsity_type="topk", k=8, l1_coeff=0.0)
        assert cfg.k == 8

    def test_valid_batch_topk_config(self):
        cfg = SAEConfig(sparsity_type="batch_topk", k=4, per_item_in_eval=True)
        assert cfg.per_item_in_eval is True


# ---------------------------------------------------------------------------
# SAEModel forward pass
# ---------------------------------------------------------------------------
DIM_MODEL = 16
DIM_SPARSE = 64
BATCH = 8


@pytest.fixture(params=["linear", "mlp"])
def encoder_type(request):
    return request.param


@pytest.fixture(
    params=[
        ("none", 0),
        ("topk", 4),
        ("batch_topk", 4),
    ]
)
def sparsity_cfg(request):
    return request.param


class TestSAEModelForward:
    def _make_model(self, encoder_type, sparsity_type, k, **kwargs):
        cfg = SAEConfig(
            dim_model=DIM_MODEL,
            dim_sparse=DIM_SPARSE,
            encoder_type=encoder_type,
            sparsity_type=sparsity_type,
            k=k,
            l1_coeff=1e-3 if sparsity_type == "none" else 0.0,
            **kwargs,
        )
        return SAEModel(cfg)

    def test_output_type(self, encoder_type, sparsity_cfg):
        sp_type, k = sparsity_cfg
        model = self._make_model(encoder_type, sp_type, k)
        x = torch.randn(BATCH, DIM_MODEL)
        out = model(x)
        assert isinstance(out, SAEOutput)

    def test_output_shapes(self, encoder_type, sparsity_cfg):
        sp_type, k = sparsity_cfg
        model = self._make_model(encoder_type, sp_type, k)
        x = torch.randn(BATCH, DIM_MODEL)
        out = model(x)
        assert out.hidden_states_recon.shape == (BATCH, DIM_MODEL)
        assert out.sparse_activations.shape == (BATCH, DIM_SPARSE)

    def test_loss_is_scalar(self, encoder_type, sparsity_cfg):
        sp_type, k = sparsity_cfg
        model = self._make_model(encoder_type, sp_type, k)
        x = torch.randn(BATCH, DIM_MODEL)
        out = model(x)
        assert out.loss.ndim == 0

    def test_loss_backward(self, encoder_type, sparsity_cfg):
        sp_type, k = sparsity_cfg
        model = self._make_model(encoder_type, sp_type, k)
        x = torch.randn(BATCH, DIM_MODEL)
        out = model(x)
        out.loss.backward()
        # At least one parameter should have gradient
        has_grad = any(
            p.grad is not None for p in model.parameters() if p.requires_grad
        )
        assert has_grad

    def test_l1_loss_reported_when_active(self):
        model = self._make_model("linear", "none", 0)
        x = torch.randn(BATCH, DIM_MODEL)
        out = model(x)
        assert out.l1_loss is not None
        assert out.l1_loss > 0

    def test_l1_loss_none_with_topk(self):
        model = self._make_model("linear", "topk", 4)
        x = torch.randn(BATCH, DIM_MODEL)
        out = model(x)
        assert out.l1_loss is None

    def test_pre_activations_optional(self):
        model = self._make_model("linear", "none", 0)
        x = torch.randn(BATCH, DIM_MODEL)
        out_no = model(x, output_pre_activations=False)
        out_yes = model(x, output_pre_activations=True)
        assert out_no.pre_activations is None
        assert out_yes.pre_activations is not None
        assert out_yes.pre_activations.shape == (BATCH, DIM_SPARSE)

    def test_loss_per_sample(self):
        model = self._make_model("linear", "none", 0)
        x = torch.randn(BATCH, DIM_MODEL)
        out = model(x, output_loss_per_sample=True)
        assert out.loss_per_sample is not None
        assert out.loss_per_sample.shape == (BATCH,)

    def test_3d_input(self):
        model = self._make_model("linear", "topk", 4)
        x = torch.randn(2, 10, DIM_MODEL)
        out = model(x)
        assert out.hidden_states_recon.shape == (2, 10, DIM_MODEL)
        assert out.sparse_activations.shape == (2, 10, DIM_SPARSE)


# ---------------------------------------------------------------------------
# Tied decoder / attach_embedding
# ---------------------------------------------------------------------------
class TestTiedDecoder:
    def test_attach_embedding(self):
        vocab_size = 64
        cfg = SAEConfig(
            dim_model=DIM_MODEL,
            dim_sparse=vocab_size,
            tied_decoder=True,
        )
        model = SAEModel(cfg)
        emb = nn.Embedding(vocab_size, DIM_MODEL)
        model.attach_embedding(emb, freeze=True)

        assert model.decoder.weight.requires_grad is False
        # Decoder weight should be embedding^T
        assert torch.allclose(model.decoder.weight, emb.weight.T)

    def test_attach_embedding_wrong_dim(self):
        cfg = SAEConfig(dim_model=DIM_MODEL, dim_sparse=DIM_SPARSE, tied_decoder=True)
        model = SAEModel(cfg)
        emb = nn.Embedding(100, DIM_MODEL)  # vocab != dim_sparse
        with pytest.raises(ValueError, match="dim_sparse"):
            model.attach_embedding(emb)

    def test_forward_with_tied_decoder(self):
        vocab_size = 64
        cfg = SAEConfig(
            dim_model=DIM_MODEL,
            dim_sparse=vocab_size,
            tied_decoder=True,
        )
        model = SAEModel(cfg)
        emb = nn.Embedding(vocab_size, DIM_MODEL)
        model.attach_embedding(emb, freeze=True)

        x = torch.randn(BATCH, DIM_MODEL)
        out = model(x)
        assert out.hidden_states_recon.shape == (BATCH, DIM_MODEL)


# ---------------------------------------------------------------------------
# Anchor loss
# ---------------------------------------------------------------------------
class TestAnchorLoss:
    @pytest.mark.parametrize("anchor_mode", ["hard", "logsumexp", "softmax"])
    def test_anchor_loss_modes(self, anchor_mode):
        vocab_size = 64
        cfg = SAEConfig(
            dim_model=DIM_MODEL,
            dim_sparse=vocab_size,
            anchor_coeff=0.1,
            anchor_mode=anchor_mode,
            anchor_topk=5,
        )
        model = SAEModel(cfg)
        emb = nn.Embedding(vocab_size, DIM_MODEL)
        model.attach_anchor_embedding(emb)

        x = torch.randn(BATCH, DIM_MODEL)
        out = model(x)
        assert out.loss_anchor is not None

    def test_no_anchor_without_embedding(self):
        cfg = SAEConfig(
            dim_model=DIM_MODEL,
            dim_sparse=DIM_SPARSE,
            anchor_coeff=0.1,
        )
        model = SAEModel(cfg)
        x = torch.randn(BATCH, DIM_MODEL)
        out = model(x)
        assert out.loss_anchor is None


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------
class TestEncodeDecode:
    def test_encode_returns_pre_and_z(self):
        cfg = SAEConfig(
            dim_model=DIM_MODEL, dim_sparse=DIM_SPARSE, sparsity_type="topk", k=4
        )
        model = SAEModel(cfg)
        x = torch.randn(BATCH, DIM_MODEL)
        pre, z = model.encode(x)
        assert pre.shape == (BATCH, DIM_SPARSE)
        assert z.shape == (BATCH, DIM_SPARSE)
        # z should be sparse
        assert (z == 0).sum() > 0

    def test_decode_shape(self):
        cfg = SAEConfig(dim_model=DIM_MODEL, dim_sparse=DIM_SPARSE)
        model = SAEModel(cfg)
        z = torch.randn(BATCH, DIM_SPARSE)
        out = model.decode(z)
        assert out.shape == (BATCH, DIM_MODEL)

    def test_nonneg_latents(self):
        cfg = SAEConfig(
            dim_model=DIM_MODEL,
            dim_sparse=DIM_SPARSE,
            nonneg_latents=True,
            sparsity_type="none",
        )
        model = SAEModel(cfg)
        x = torch.randn(BATCH, DIM_MODEL)
        pre, z = model.encode(x)
        # z should have no negative values (ReLU applied)
        assert (z >= 0).all()

    def test_no_nonneg_latents(self):
        cfg = SAEConfig(
            dim_model=DIM_MODEL,
            dim_sparse=DIM_SPARSE,
            nonneg_latents=False,
            sparsity_type="none",
        )
        model = SAEModel(cfg)
        torch.manual_seed(0)
        x = torch.randn(BATCH, DIM_MODEL)
        pre, z = model.encode(x)
        # Without ReLU, z can be negative
        assert (z < 0).any()
