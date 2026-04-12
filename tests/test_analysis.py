"""Tests for vasae.analysis modules."""

import json
import re
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from vasae.analysis.alignment import (
    compute_geometric_alignment,
    compute_logit_attribution,
)
from vasae.analysis.hooks import make_intervention_hook, run_with_hook
from vasae.analysis.io import (
    discover_checkpoints,
    load_layer_results,
    save_figure,
    save_results,
)
from vasae.analysis.stats import summarize_tensor


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------


class TestGeometricAlignment:
    def test_identity(self):
        """Features identical to references should have max_sim ~1."""
        features = torch.randn(4, 8)
        result = compute_geometric_alignment(features, features, top_k=2)
        assert result.max_sims.shape == (4,)
        assert torch.allclose(result.max_sims, torch.ones(4), atol=1e-5)

    def test_orthogonal(self):
        """Orthogonal features should have sim ~0."""
        features = torch.eye(4)
        references = torch.zeros(1, 4)
        references[0, 0] = 1.0  # only aligned with feature 0
        result = compute_geometric_alignment(features, references, top_k=1)
        assert result.max_sims[0].item() > 0.99
        for i in range(1, 4):
            assert abs(result.max_sims[i].item()) < 1e-5

    def test_topk_shape(self):
        features = torch.randn(10, 8)
        references = torch.randn(20, 8)
        result = compute_geometric_alignment(features, references, top_k=3)
        assert result.topk_sims.shape == (10, 3)
        assert result.topk_indices.shape == (10, 3)

    def test_batch_size(self):
        """Different batch sizes should give the same result."""
        features = torch.randn(10, 8)
        references = torch.randn(20, 8)
        r1 = compute_geometric_alignment(features, references, top_k=2, batch_size=3)
        r2 = compute_geometric_alignment(features, references, top_k=2, batch_size=100)
        assert torch.allclose(r1.max_sims, r2.max_sims, atol=1e-5)
        assert torch.equal(r1.topk_indices, r2.topk_indices)


class TestLogitAttribution:
    def test_shapes(self):
        features = torch.randn(5, 8)
        W_U = torch.randn(10, 8)  # vocab_size=10
        result = compute_logit_attribution(features, W_U, top_k=3)
        assert result.entropy.shape == (5,)
        assert result.max_mean_ratio.shape == (5,)
        assert result.top1_concentration.shape == (5,)
        assert result.top5_concentration.shape == (5,)
        assert result.max_logit.shape == (5,)
        assert result.max_token_id.shape == (5,)
        assert result.topk_vals.shape == (5, 3)
        assert result.topk_tokens.shape == (5, 3)

    def test_entropy_nonnegative(self):
        features = torch.randn(4, 8)
        W_U = torch.randn(20, 8)
        result = compute_logit_attribution(features, W_U)
        assert (result.entropy >= 0).all()

    def test_concentration_range(self):
        features = torch.randn(4, 8)
        W_U = torch.randn(20, 8)
        result = compute_logit_attribution(features, W_U)
        assert (result.top1_concentration >= 0).all()
        assert (result.top1_concentration <= 1).all()
        assert (result.top5_concentration >= result.top1_concentration).all()


# ---------------------------------------------------------------------------
# hooks
# ---------------------------------------------------------------------------


class _DummyOutput:
    def __init__(self, logits):
        self.logits = logits


class _DummyModel(nn.Module):
    """Minimal model: a single Linear layer followed by a 'head'."""

    def __init__(self, dim):
        super().__init__()
        self.layer = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.layer.weight)
        self.head = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.head.weight)

    def forward(self, input_ids=None, attention_mask=None):
        x = input_ids.float()
        x = self.layer(x)
        x = self.head(x)
        return _DummyOutput(logits=x)


class TestHooks:
    def test_make_intervention_hook_tuple(self):
        """Hook should modify first element of tuple output."""
        hook = make_intervention_hook(lambda h: h * 2)
        result = hook(None, None, (torch.ones(2, 3), "extra"))
        assert isinstance(result, tuple)
        assert torch.equal(result[0], torch.ones(2, 3) * 2)
        assert result[1] == "extra"

    def test_make_intervention_hook_tensor(self):
        """Hook should modify plain tensor output."""
        hook = make_intervention_hook(lambda h: h + 1)
        result = hook(None, None, torch.zeros(2, 3))
        assert torch.equal(result, torch.ones(2, 3))

    def test_run_with_hook(self):
        model = _DummyModel(4)
        x = torch.ones(1, 4)
        mask = torch.ones(1, 4)

        # Without hook: identity -> identity = x
        clean = model(input_ids=x).logits
        assert torch.allclose(clean, x)

        # With hook: zero the layer output
        logits = run_with_hook(model, model.layer, lambda h: h * 0, x, mask)
        assert torch.allclose(logits, torch.zeros(1, 4))


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


class TestSummarize:
    def test_keys(self):
        t = torch.randn(100)
        s = summarize_tensor(t)
        expected_keys = {"mean", "std", "median", "min", "max", "p5", "p25", "p75", "p95"}
        assert set(s.keys()) == expected_keys

    def test_constant(self):
        t = torch.full((50,), 3.0)
        s = summarize_tensor(t)
        assert abs(s["mean"] - 3.0) < 1e-5
        assert abs(s["median"] - 3.0) < 1e-5
        assert abs(s["min"] - 3.0) < 1e-5
        assert abs(s["max"] - 3.0) < 1e-5


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------


class TestIO:
    def test_discover_checkpoints(self, tmp_path):
        (tmp_path / "run_L0_soft").mkdir()
        (tmp_path / "run_L3_soft").mkdir()
        (tmp_path / "run_L5_plain").mkdir()
        (tmp_path / "ignore.txt").touch()

        pattern = re.compile(r"run_L(\d+)_soft$")
        result = discover_checkpoints(tmp_path, pattern)
        assert set(result.keys()) == {0, 3}
        assert result[0] == tmp_path / "run_L0_soft"

    def test_save_and_load_results(self, tmp_path):
        data = {"layer_idx": 5, "score": 0.99}
        tensors = {"t": torch.tensor([1.0, 2.0])}
        save_results(tmp_path / "L5", data, tensors=tensors)

        with open(tmp_path / "L5" / "results.json") as f:
            loaded = json.load(f)
        assert loaded["layer_idx"] == 5

        loaded_t = torch.load(tmp_path / "L5" / "tensors.pt", weights_only=True)
        assert torch.equal(loaded_t["t"], tensors["t"])

    def test_load_layer_results(self, tmp_path):
        for li in [0, 3]:
            d = tmp_path / f"L{li}"
            d.mkdir()
            with open(d / "results.json", "w") as f:
                json.dump({"layer_idx": li, "val": li * 10}, f)

        results = load_layer_results(tmp_path)
        assert set(results.keys()) == {0, 3}
        assert results[3]["val"] == 30

    def test_save_figure(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        save_figure(fig, tmp_path, "test_fig")
        plt.close(fig)

        assert (tmp_path / "test_fig.png").exists()
        assert (tmp_path / "test_fig.pdf").exists()
