"""Unit tests for DecomposeSAEModel."""

import pytest
import torch
import torch.nn as nn

from vasae.models.decompose_sae import DecomposeSAEModel, DecomposeSAEOutput


DIM_INPUT = 16
DIM_SPARSE = 64
D_PCA = 8
K = 4
BATCH = 8


class TestDecomposeSAEModel:
    @pytest.fixture
    def model(self):
        m = DecomposeSAEModel(DIM_INPUT, DIM_SPARSE, D_PCA, K)
        W = torch.randn(DIM_INPUT, D_PCA)
        m.attach_pca(W)
        return m

    def test_output_type(self, model):
        x = torch.randn(BATCH, DIM_INPUT)
        out = model(x)
        assert isinstance(out, DecomposeSAEOutput)

    def test_output_shapes(self, model):
        x = torch.randn(BATCH, DIM_INPUT)
        out = model(x)
        assert out.h_recon.shape == (BATCH, DIM_INPUT)
        assert out.z_s.shape == (BATCH, DIM_SPARSE)
        assert out.z_d.shape == (BATCH, D_PCA)
        assert out.h_sparse.shape == (BATCH, DIM_INPUT)

    def test_loss_scalar(self, model):
        x = torch.randn(BATCH, DIM_INPUT)
        out = model(x)
        assert out.loss.ndim == 0

    def test_loss_backward(self, model):
        x = torch.randn(BATCH, DIM_INPUT)
        out = model(x)
        out.loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad

    def test_attach_embedding(self):
        model = DecomposeSAEModel(DIM_INPUT, DIM_SPARSE, D_PCA, K)
        W = torch.randn(DIM_INPUT, D_PCA)
        model.attach_pca(W)
        emb = nn.Embedding(DIM_SPARSE, DIM_INPUT)
        model.attach_embedding(emb, freeze=True)
        assert model.decoder_sparse.weight.requires_grad is False
