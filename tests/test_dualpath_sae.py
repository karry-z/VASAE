"""Unit tests for DualPathSAE."""

import pytest
import torch
import torch.nn as nn

from vasae.models.dualpath_sae import DualPathSAE, DualPathSAEOutput


DIM_INPUT = 16
VOCAB_SIZE = 64
D_PCA = 8
BATCH = 8


class TestDualPathSAE:
    @pytest.fixture
    def model(self):
        m = DualPathSAE(DIM_INPUT, VOCAB_SIZE, D_PCA)
        emb = nn.Embedding(VOCAB_SIZE, DIM_INPUT)
        m.attach_embedding(emb)
        P_k = torch.randn(DIM_INPUT, D_PCA)
        mean_r = torch.randn(DIM_INPUT)
        m.attach_pca(P_k, mean_r)
        return m

    def test_output_type(self, model):
        x = torch.randn(BATCH, DIM_INPUT)
        out = model(x)
        assert isinstance(out, DualPathSAEOutput)

    def test_output_shapes(self, model):
        x = torch.randn(BATCH, DIM_INPUT)
        out = model(x)
        assert out.h_recon.shape == (BATCH, DIM_INPUT)
        assert out.z.shape == (BATCH, VOCAB_SIZE)
        assert out.y.shape == (BATCH, D_PCA)
        assert out.h_sparse.shape == (BATCH, DIM_INPUT)

    def test_loss_components(self, model):
        x = torch.randn(BATCH, DIM_INPUT)
        out = model(x)
        assert out.loss.ndim == 0
        assert out.recon_loss.ndim == 0
        assert out.l1_z.ndim == 0
        assert out.l1_y.ndim == 0

    def test_loss_backward(self, model):
        x = torch.randn(BATCH, DIM_INPUT)
        out = model(x)
        out.loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad
