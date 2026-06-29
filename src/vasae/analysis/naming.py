from collections.abc import Callable, Sequence

import torch

from .alignment import nearest_token_alignment


def nearest_token_names(
    feature_directions: torch.Tensor,
    vocab_embeddings: torch.Tensor,
    token_lookup: Sequence[str] | Callable[[int], str],
    top_k: int = 1,
    chunk_size: int = 2048,
) -> list[list[tuple[str, float]]]:
    """Map feature directions to nearest-token names and cosine scores."""
    token_ids, scores = nearest_token_alignment(
        feature_directions=feature_directions,
        vocab_embeddings=vocab_embeddings,
        top_k=top_k,
        chunk_size=chunk_size,
    )

    names: list[list[tuple[str, float]]] = []
    for feature_ids, feature_scores in zip(token_ids.tolist(), scores.tolist()):
        rows = []
        for token_id, score in zip(feature_ids, feature_scores):
            if callable(token_lookup):
                token_name = token_lookup(token_id)
            else:
                token_name = token_lookup[token_id]
            rows.append((token_name, float(score)))
        names.append(rows)
    return names
