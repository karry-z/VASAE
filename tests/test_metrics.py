"""Unit tests for metric interface, LogitLens, and Aggregator."""

import pytest
import torch
import torch.nn as nn

from vasae.metrics.base import Aggregator, IMetric, MetricComposer
from vasae.metrics.logitlens import (
    LogitLens,
    LogitLensAccMetric,
    compute_token_prediction_acc,
)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
class TestAggregator:
    def test_single_batch(self):
        agg = Aggregator()
        agg.add({"loss": 2.0, "acc": 0.8}, batch_size=10)
        result = agg.compute()
        assert result["loss"] == pytest.approx(2.0)
        assert result["acc"] == pytest.approx(0.8)

    def test_weighted_average(self):
        agg = Aggregator()
        agg.add({"loss": 1.0}, batch_size=10)
        agg.add({"loss": 3.0}, batch_size=10)
        result = agg.compute()
        assert result["loss"] == pytest.approx(2.0)

    def test_unequal_weights(self):
        agg = Aggregator()
        agg.add({"loss": 1.0}, batch_size=1)
        agg.add({"loss": 3.0}, batch_size=3)
        result = agg.compute()
        # (1*1 + 3*3) / (1+3) = 10/4 = 2.5
        assert result["loss"] == pytest.approx(2.5)

    def test_skips_none(self):
        agg = Aggregator()
        agg.add({"loss": 1.0, "optional": None}, batch_size=10)
        result = agg.compute()
        assert "loss" in result
        assert "optional" not in result

    def test_sparse_keys(self):
        """Different batches can report different keys."""
        agg = Aggregator()
        agg.add({"loss": 1.0, "acc": 0.5}, batch_size=10)
        agg.add({"loss": 2.0}, batch_size=10)
        result = agg.compute()
        assert result["loss"] == pytest.approx(1.5)
        assert result["acc"] == pytest.approx(0.5)  # only 1 batch contributed


# ---------------------------------------------------------------------------
# MetricComposer
# ---------------------------------------------------------------------------
class DummyMetric(IMetric):
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def compute(self, context):
        return {self.key: self.value}


class TestMetricComposer:
    def test_compose_two_metrics(self):
        mc = MetricComposer(
            [
                DummyMetric("a", 1.0),
                DummyMetric("b", 2.0),
            ]
        )
        result = mc.compute({})
        assert result == {"a": 1.0, "b": 2.0}

    def test_empty_composer(self):
        mc = MetricComposer([])
        result = mc.compute({})
        assert result == {}


# ---------------------------------------------------------------------------
# LogitLensAccuracy
# ---------------------------------------------------------------------------
class TestTokenPredictionAcc:
    def test_perfect(self):
        result = compute_token_prediction_acc([1, 2, 3], [1, 2, 3])
        assert result == pytest.approx(1.0)

    def test_none_match(self):
        result = compute_token_prediction_acc([1, 2, 3], [4, 5, 6])
        assert result == pytest.approx(0.0)

    def test_partial(self):
        result = compute_token_prediction_acc([1, 2, 3, 4], [1, 2, 5, 6])
        assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# LogitLens
# ---------------------------------------------------------------------------
class TestLogitLens:
    @pytest.fixture
    def unembed(self):
        # Simple 4-class unembedding layer
        layer = nn.Linear(8, 4, bias=False)
        # Make it deterministic
        with torch.no_grad():
            layer.weight.copy_(torch.eye(4, 8))
        return layer

    def test_unembed_shape(self, unembed):
        ll = LogitLens(unembed)
        x = torch.randn(2, 5, 8)
        logits = ll.unembed(x)
        assert logits.shape == (2, 5, 4)

    def test_top1_returns_dict(self, unembed):
        ll = LogitLens(unembed)
        x = torch.randn(2, 5, 8)
        result = ll.top1(x)
        assert "token_ids" in result
        assert "token_probs" in result
        assert "probs" in result

    def test_top1_shape(self, unembed):
        ll = LogitLens(unembed)
        x = torch.randn(2, 5, 8)
        result = ll.top1(x)
        assert result["token_ids"].shape == (2, 5)
        assert result["token_probs"].shape == (2, 5)


# ---------------------------------------------------------------------------
# LogitLensMetric
# ---------------------------------------------------------------------------
class TestLogitLensMetric:
    @pytest.fixture
    def metric(self):
        unembed = nn.Linear(8, 4, bias=False)
        ll = LogitLens(unembed)
        return LogitLensAccMetric(ll)

    def test_compute(self, metric):
        context = {
            "hidden_states": torch.randn(4, 8),
            "hidden_states_recon": torch.randn(4, 8),
        }
        result = metric.compute(context)
        assert "logitlens_acc" in result
        assert 0.0 <= result["logitlens_acc"] <= 1.0

    def test_identical_input_gives_perfect_accuracy(self, metric):
        x = torch.randn(4, 8)
        context = {"hidden_states": x, "hidden_states_recon": x}
        result = metric.compute(context)
        assert result["logitlens_acc"] == pytest.approx(1.0)
