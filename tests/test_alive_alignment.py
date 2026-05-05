import pytest
import torch

from vasae.analysis.alive_alignment import compute_alive_alignment_stats


def test_alive_alignment_rate_excludes_dead_features():
    stats = compute_alive_alignment_stats(
        alive_mask=torch.tensor([True, False, True, False]),
        alignment_scores=torch.tensor([0.9, 0.95, 0.1, 0.99]),
        threshold=0.8,
    )

    assert stats.n_features == 4
    assert stats.n_alive == 2
    assert stats.n_aligned == 3
    assert stats.n_alive_aligned == 1
    assert stats.alive_alignment_rate == pytest.approx(0.5)
    assert stats.dead_rate == pytest.approx(0.5)


def test_alive_alignment_threshold_is_inclusive():
    stats = compute_alive_alignment_stats(
        alive_mask=torch.tensor([True, True]),
        alignment_scores=torch.tensor([0.8, 0.7999]),
        threshold=0.8,
    )

    assert stats.n_aligned == 1
    assert stats.n_alive_aligned == 1
    assert stats.alive_alignment_rate == pytest.approx(0.5)


def test_alive_alignment_handles_no_alive_features():
    stats = compute_alive_alignment_stats(
        alive_mask=torch.tensor([False, False]),
        alignment_scores=torch.tensor([0.9, 0.1]),
        threshold=0.8,
    )

    assert stats.n_alive == 0
    assert stats.n_alive_aligned == 0
    assert stats.alive_alignment_rate == 0.0
    assert stats.dead_rate == 1.0
