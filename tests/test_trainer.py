"""Unit tests for the unified Trainer class."""

import pytest
import torch

from vasae.engine.trainer import Trainer
from vasae.metrics.base import Aggregator, IMetric, MetricComposer
from vasae.models.sae import SAEConfig, SAEModel

DIM_MODEL = 16
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
        yield {"activations": torch.randn(BATCH, SEQ_LEN, DIM_MODEL)}


def _make_online_data_source(n_batches=N_BATCHES):
    """Simulate an OnlineActivationSource yielding dicts with input_ids."""
    for _ in range(n_batches):
        yield {
            "activations": torch.randn(BATCH, SEQ_LEN, DIM_MODEL),
            "input_ids": torch.randint(0, 100, (BATCH, SEQ_LEN)),
            "attention_mask": torch.ones(BATCH, SEQ_LEN, dtype=torch.long),
        }


@pytest.fixture
def trainer():
    cfg = SAEConfig(
        dim_model=DIM_MODEL,
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
        # With max_batches=1, should only process ~1 batch
        result = trainer.train_epoch(_make_data_source(10), max_batches=1)
        assert isinstance(result, dict)

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
        params_before = {
            n: p.clone()
            for n, p in trainer.sae_model.named_parameters()
            if p.requires_grad
        }
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
