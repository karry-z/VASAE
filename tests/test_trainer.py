"""Unit tests for the unified Trainer class."""

import pytest
import torch

from vasae.engine.trainer import Trainer, _source_progress_totals
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


class CountingMetric(IMetric):
    def __init__(self):
        self.calls = 0

    def reset(self):
        self.calls = 0

    def compute(self, context):
        self.calls += 1
        return {"counting_metric": 1.0}


class TokenBudgetSource:
    total_token_budget = 20_000_000
    batch_size = 32
    max_length = 128


class SmallTokenBudgetSource:
    total_token_budget = BATCH * SEQ_LEN
    batch_size = BATCH
    max_length = SEQ_LEN

    def __iter__(self):
        yield from _make_online_data_source(3)


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
def model():
    cfg = SAEConfig(
        dim_model=DIM_MODEL,
        dim_sparse=DIM_SPARSE,
        sparsity_type="topk",
        k=4,
    )
    return SAEModel(cfg)


@pytest.fixture
def optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)


@pytest.fixture
def trainer(model):
    metrics = MetricComposer([DummyMetric()])
    return Trainer(
        sae_model=model,
        metrics=metrics,
        device="cpu",
    )


class TestTrainer:
    def test_progress_totals_from_token_budget(self):
        total_batches, estimated, total_tokens = _source_progress_totals(
            TokenBudgetSource()
        )
        assert total_batches == 4_883
        assert estimated is True
        assert total_tokens == 20_000_000

    def test_progress_totals_respect_max_batches(self):
        total_batches, estimated, total_tokens = _source_progress_totals(
            TokenBudgetSource(),
            max_batches=200,
        )
        assert total_batches == 200
        assert estimated is True
        assert total_tokens == 819_200

    def test_train_epoch_returns_dict(self, trainer, optimizer):
        result = trainer.train_epoch(_make_data_source(), optimizer=optimizer)
        assert isinstance(result, dict)
        assert "loss" in result
        assert "dummy_metric" in result

    def test_train_epoch_max_batches_is_exact(self, model, optimizer):
        metric = CountingMetric()
        trainer = Trainer(
            sae_model=model,
            metrics=MetricComposer([metric]),
            device="cpu",
        )
        result = trainer.train_epoch(
            _make_data_source(10),
            optimizer=optimizer,
            max_batches=1,
        )
        assert isinstance(result, dict)
        assert metric.calls == 1

    def test_train_epoch_logs_token_budget_progress(self, trainer, optimizer, caplog):
        caplog.set_level("INFO", logger="vasae.engine.trainer")
        trainer.train_epoch(
            SmallTokenBudgetSource(),
            optimizer=optimizer,
            max_batches=1,
            log_every=1,
        )
        messages = "\n".join(record.getMessage() for record in caplog.records)
        assert "batch 1/~1" in messages
        assert "tokens=40/40 (100.00%)" in messages
        assert "batch 1/?" not in messages

    def test_evaluate_max_batches_is_exact(self, model):
        metric = CountingMetric()
        trainer = Trainer(
            sae_model=model,
            metrics=MetricComposer([DummyMetric()]),
            eval_metrics=MetricComposer([metric]),
            device="cpu",
        )
        result = trainer.evaluate(_make_data_source(10), max_batches=2)
        assert isinstance(result, dict)
        assert metric.calls == 2

    def test_evaluate_returns_dict(self, trainer):
        result = trainer.evaluate(_make_data_source())
        assert isinstance(result, dict)
        assert "loss" in result

    def test_evaluate_no_grad(self, trainer):
        """Evaluate should not accumulate gradients."""
        trainer.evaluate(_make_data_source())
        for p in trainer.sae_model.parameters():
            assert p.grad is None

    def test_train_updates_weights(self, trainer, optimizer):
        # Snapshot initial weights
        params_before = {
            n: p.clone()
            for n, p in trainer.sae_model.named_parameters()
            if p.requires_grad
        }
        trainer.train_epoch(_make_data_source(), optimizer=optimizer)
        changed = False
        for n, p in trainer.sae_model.named_parameters():
            if p.requires_grad and not torch.allclose(params_before[n], p):
                changed = True
                break
        assert changed, "Training should update at least one parameter"

    def test_fit_returns_training_summary(self, trainer, optimizer):
        result = trainer.fit(
            train_source=_make_data_source(),
            eval_source=_make_data_source(),
            optimizer=optimizer,
            num_epochs=1,
        )

        assert result["stopped_epoch"] == 1
        assert "loss" in result["train"]
        assert "loss" in result["eval"]

    def test_online_data_source(self, trainer):
        """Trainer should handle data with input_ids/attention_mask keys."""
        result = trainer.evaluate(_make_online_data_source())
        assert isinstance(result, dict)
