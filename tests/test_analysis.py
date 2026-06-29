import pytest
import torch

from vasae.analysis import nearest_token_alignment, nearest_token_names


def test_nearest_token_alignment_returns_ids_and_scores():
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    vocab = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

    token_ids, scores = nearest_token_alignment(features, vocab, top_k=1)

    assert token_ids.tolist() == [[1], [0]]
    assert torch.allclose(scores, torch.ones(2, 1))


def test_nearest_token_names_maps_ids_to_strings():
    features = torch.tensor([[1.0, 0.0]])
    vocab = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    names = nearest_token_names(features, vocab, ["up", "right"], top_k=1)

    assert names == [[("right", pytest.approx(1.0))]]


def test_nearest_token_names_accepts_callable_lookup():
    features = torch.tensor([[0.0, 1.0]])
    vocab = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    names = nearest_token_names(features, vocab, lambda token_id: f"tok_{token_id}", top_k=1)

    assert names == [[("tok_0", pytest.approx(1.0))]]
