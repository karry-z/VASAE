"""Unit tests for the unified Trainer class."""

import pytest
import torch

from vasae.metrics.base import IMetric, MetricComposer, Aggregator
from vasae.models.sae import SAEConfig, SAEModel
from vasae.engine.trainer import Trainer


DIM_INPUT = 16
DIM_SPARSE = 64
BATCH = 4
SEQ_LEN = 10
N_BATCHES = 3


class DummyMetric(IMetric):
    def compute(self, context):
        return {"dummy_metric": 0.5}


def _make_data_source(n_batches=N_BATCHES):
    """Simulate an offline DataLoader yielding dicts."""
    for _ in range(n_batches):
        yield {"activations": torch.randn(BATCH, SEQ_LEN, DIM_INPUT)}


def _make_online_data_source(n_batches=N_BATCHES):
    """Simulate an OnlineActivationSource yielding dicts with input_ids."""
    for _ in range(n_batches):
        yield {
            "activations": torch.randn(BATCH, SEQ_LEN, DIM_INPUT),
            "input_ids": torch.randint(0, 100, (BATCH, SEQ_LEN)),
            "attention_mask": torch.ones(BATCH, SEQ_LEN, dtype=torch.long),
        }


class CountingDataSource:
    def __init__(self, n_batches):
        self.n_batches = n_batches
        self.yielded = 0

    def __iter__(self):
        for _ in range(self.n_batches):
            self.yielded += 1
            yield {"activations": torch.randn(BATCH, SEQ_LEN, DIM_INPUT)}


@pytest.fixture
def trainer():
    cfg = SAEConfig(
        dim_input=DIM_INPUT,
        dim_sparse=DIM_SPARSE,
        sparsity_type="topk",
        k=4,
    )
    model = SAEModel(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = MetricComposer([DummyMetric()])
    return Trainer(
        sae_model=model,
        optimizer=optimizer,
        metrics=metrics,
        device="cpu",
    )


class TestTrainer:
    def test_train_epoch_returns_dict(self, trainer):
        result = trainer.train_epoch(_make_data_source())
        assert isinstance(result, dict)
        assert "loss" in result
        assert "dummy_metric" in result

    def test_train_epoch_max_batches(self, trainer):
        source = CountingDataSource(10)
        result = trainer.train_epoch(source, max_batches=1)
        assert isinstance(result, dict)
        assert source.yielded == 1

    def test_evaluate_max_batches(self, trainer):
        source = CountingDataSource(10)
        result = trainer.evaluate(source, max_batches=1)
        assert isinstance(result, dict)
        assert source.yielded == 1

    def test_evaluate_returns_dict(self, trainer):
        result = trainer.evaluate(_make_data_source())
        assert isinstance(result, dict)
        assert "loss" in result

    def test_evaluate_no_grad(self, trainer):
        """Evaluate should not accumulate gradients."""
        trainer.evaluate(_make_data_source())
        for p in trainer.sae_model.parameters():
            assert p.grad is None

    def test_train_updates_weights(self, trainer):
        # Snapshot initial weights
        params_before = {n: p.clone() for n, p in trainer.sae_model.named_parameters() if p.requires_grad}
        trainer.train_epoch(_make_data_source())
        changed = False
        for n, p in trainer.sae_model.named_parameters():
            if p.requires_grad and not torch.allclose(params_before[n], p):
                changed = True
                break
        assert changed, "Training should update at least one parameter"

    def test_online_data_source(self, trainer):
        """Trainer should handle data with input_ids/attention_mask keys."""
        result = trainer.evaluate(_make_online_data_source())
        assert isinstance(result, dict)
