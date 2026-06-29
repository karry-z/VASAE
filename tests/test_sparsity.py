import torch

from vasae.models.sparsity import TopKSparse


class TestTopKSparse:
    def test_output_shape(self):
        sp = TopKSparse(k=4)
        x = torch.randn(8, 32)
        out = sp(x)
        assert out.shape == x.shape

    def test_sparsity_level(self):
        sp = TopKSparse(k=4)
        x = torch.randn(8, 32)
        out = sp(x)
        assert ((out != 0).sum(dim=-1) <= 4).all()

    def test_k_larger_than_dim(self):
        sp = TopKSparse(k=100)
        x = torch.randn(8, 32)
        out = sp(x)
        assert torch.allclose(out, x)

    def test_values_preserved(self):
        sp = TopKSparse(k=4)
        x = torch.randn(8, 32)
        out = sp(x)
        mask = out != 0
        assert torch.allclose(out[mask], x[mask])

    def test_use_abs(self):
        sp = TopKSparse(k=2, use_abs=True)
        x = torch.tensor([[0.1, -5.0, 0.2, -4.0]])
        out = sp(x)
        assert out[0, 1] == -5.0
        assert out[0, 3] == -4.0
        assert out[0, 0] == 0.0
        assert out[0, 2] == 0.0
