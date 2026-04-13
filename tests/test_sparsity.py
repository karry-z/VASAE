"""Unit tests for sparsity modules."""

import torch

from vasae.models.sparsity import BatchTopKSparse, TopKSparse


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
        # Each row should have at most k non-zero entries
        for i in range(8):
            assert (out[i] != 0).sum() <= 4

    def test_k_larger_than_dim(self):
        sp = TopKSparse(k=100)
        x = torch.randn(8, 32)
        out = sp(x)
        # k clamped to dim, so all values kept
        assert torch.allclose(out, x)

    def test_values_preserved(self):
        """Non-zero entries should have the same values as the input."""
        sp = TopKSparse(k=4)
        x = torch.randn(8, 32)
        out = sp(x)
        mask = out != 0
        assert torch.allclose(out[mask], x[mask])

    def test_use_abs(self):
        sp = TopKSparse(k=2, use_abs=True)
        # Create input where largest abs values are negative
        x = torch.tensor([[0.1, -5.0, 0.2, -4.0]])
        out = sp(x)
        assert out[0, 1] == -5.0
        assert out[0, 3] == -4.0
        assert out[0, 0] == 0.0
        assert out[0, 2] == 0.0


class TestBatchTopKSparse:
    def test_output_shape(self):
        sp = BatchTopKSparse(k=4)
        x = torch.randn(8, 32)
        sp.train()
        out = sp(x)
        assert out.shape == x.shape

    def test_global_sparsity_train(self):
        sp = BatchTopKSparse(k=2)
        sp.train()
        x = torch.randn(4, 16)
        out = sp(x)
        # Total non-zeros should be k * n_items = 2 * 4 = 8
        assert (out != 0).sum() == 8

    def test_per_item_eval(self):
        sp = BatchTopKSparse(k=2, per_item_in_eval=True)
        sp.eval()
        x = torch.randn(4, 16)
        out = sp(x)
        # Each row should have at most k non-zero entries
        for i in range(4):
            assert (out[i] != 0).sum() <= 2

    def test_batch_mode_in_eval_without_per_item(self):
        sp = BatchTopKSparse(k=2, per_item_in_eval=False)
        sp.eval()
        x = torch.randn(4, 16)
        out = sp(x)
        # Still uses global topk
        assert (out != 0).sum() == 8

    def test_values_preserved(self):
        sp = BatchTopKSparse(k=4)
        sp.train()
        x = torch.randn(8, 32)
        out = sp(x)
        mask = out != 0
        assert torch.allclose(out[mask], x[mask])
