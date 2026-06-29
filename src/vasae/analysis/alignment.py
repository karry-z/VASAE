from typing import Tuple

import torch
import torch.nn.functional as F


def nearest_token_alignment(
    feature_directions: torch.Tensor,
    vocab_embeddings: torch.Tensor,
    top_k: int = 1,
    chunk_size: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return nearest token ids and cosine similarities for each feature direction."""
    if feature_directions.ndim != 2:
        raise ValueError("feature_directions must have shape [n_features, dim].")
    if vocab_embeddings.ndim != 2:
        raise ValueError("vocab_embeddings must have shape [vocab_size, dim].")
    if feature_directions.size(1) != vocab_embeddings.size(1):
        raise ValueError(
            "feature_directions and vocab_embeddings must have the same embedding dim."
        )
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    k = min(top_k, vocab_embeddings.size(0))
    features = F.normalize(feature_directions, dim=1)
    vocab = F.normalize(vocab_embeddings.to(features.dtype), dim=1)

    all_scores = []
    all_ids = []
    for start in range(0, features.size(0), chunk_size):
        sims = features[start : start + chunk_size] @ vocab.T
        scores, token_ids = sims.topk(k, dim=1)
        all_scores.append(scores)
        all_ids.append(token_ids)

    return torch.cat(all_ids, dim=0), torch.cat(all_scores, dim=0)
