"""Unit tests for encoder modules."""

import torch

from vasae.models.encoders import LinearEncoder, MLPEncoder


class TestLinearEncoder:
    def test_output_shape(self):
        enc = LinearEncoder(dim_input=16, dim_sparse=64)
        x = torch.randn(8, 16)
        out = enc(x)
        assert out.shape == (8, 64)

    def test_3d_input(self):
        enc = LinearEncoder(dim_input=16, dim_sparse=64)
        x = torch.randn(2, 10, 16)
        out = enc(x)
        assert out.shape == (2, 10, 64)


class TestMLPEncoder:
    def test_output_shape(self):
        enc = MLPEncoder(dim_input=16, dim_sparse=64)
        x = torch.randn(8, 16)
        out = enc(x)
        assert out.shape == (8, 64)

    def test_3d_input(self):
        enc = MLPEncoder(dim_input=16, dim_sparse=64)
        x = torch.randn(2, 10, 16)
        out = enc(x)
        assert out.shape == (2, 10, 64)

    def test_custom_hidden_mult(self):
        enc = MLPEncoder(dim_input=16, dim_sparse=64, hidden_mult=2)
        x = torch.randn(8, 16)
        out = enc(x)
        assert out.shape == (8, 64)
