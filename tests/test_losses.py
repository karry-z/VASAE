"""Tests for vasae.losses."""

import pytest
import torch
import torch.nn.functional as F

from vasae.losses.anchor import AnchorLoss
from vasae.losses.cosine_sim import chunked_cosine_sim


# ---------------------------------------------------------------------------
# chunked_cosine_sim
# ---------------------------------------------------------------------------
class TestChunkedCosineSim:
    def test_max_reduce_matches_full(self):
        """Chunked max-reduce should match a single full matmul."""
        features = torch.randn(20, 8)
        references = torch.randn(30, 8)

        # full computation
        f_norm = F.normalize(features, dim=1)
        r_norm = F.normalize(references, dim=1)
        expected = (f_norm @ r_norm.T).max(dim=1)[0]

        result = chunked_cosine_sim(
            features,
            references,
            reduce_fn=lambda sim: sim.max(dim=1)[0],
            chunk_size=7,
        )
        assert torch.allclose(result, expected, atol=1e-5)

    def test_chunk_size_invariance(self):
        """Different chunk sizes should produce the same result."""
        features = torch.randn(15, 8)
        references = torch.randn(25, 8)
        reduce_fn = lambda sim: sim.max(dim=1)[0]

        r1 = chunked_cosine_sim(features, references, reduce_fn, chunk_size=3)
        r2 = chunked_cosine_sim(features, references, reduce_fn, chunk_size=100)
        assert torch.allclose(r1, r2, atol=1e-6)

    def test_identity_max_sim(self):
        """Features identical to references should have max sim ~1."""
        vecs = torch.randn(10, 8)
        result = chunked_cosine_sim(
            vecs,
            vecs,
            reduce_fn=lambda sim: sim.max(dim=1)[0],
            chunk_size=4,
        )
        assert torch.allclose(result, torch.ones(10), atol=1e-5)

    def test_2d_reduce(self):
        """reduce_fn can return (chunk, k) — result should be (n, k)."""
        features = torch.randn(12, 8)
        references = torch.randn(20, 8)

        result = chunked_cosine_sim(
            features,
            references,
            reduce_fn=lambda sim: sim.topk(3, dim=1)[0],
            chunk_size=5,
        )
        assert result.shape == (12, 3)


# ---------------------------------------------------------------------------
# AnchorLoss
# ---------------------------------------------------------------------------
class TestAnchorLoss:
    def test_perfect_alignment(self):
        """If features == references, loss should be ~ -1."""
        loss_fn = AnchorLoss(mode="hard")
        vecs = torch.randn(10, 8)
        loss = loss_fn(vecs, vecs)
        assert loss.item() == pytest.approx(-1.0, abs=1e-5)

    @pytest.mark.parametrize("mode", ["hard", "logsumexp", "softmax"])
    def test_all_modes_run(self, mode):
        loss_fn = AnchorLoss(mode=mode, topk=5)
        features = torch.randn(16, 8)
        references = torch.randn(32, 8)
        loss = loss_fn(features, references)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_gradient_flows(self):
        loss_fn = AnchorLoss(mode="hard")
        features = torch.randn(10, 8, requires_grad=True)
        references = torch.randn(20, 8)
        loss = loss_fn(features, references)
        loss.backward()
        assert features.grad is not None
        assert features.grad.shape == (10, 8)

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            AnchorLoss(mode="unknown")
