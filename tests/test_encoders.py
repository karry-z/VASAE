import torch

from vasae.models.encoders import LinearEncoder


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
