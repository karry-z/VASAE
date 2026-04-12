"""Anchor loss: align decoder features to token embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cosine_sim import chunked_cosine_sim


class AnchorLoss(nn.Module):
    """Maximise cosine similarity between feature vectors and a reference set.

    Three reduction modes over the similarity row:

    * **hard** — ``max_j cos(f_i, r_j)``
    * **logsumexp** — ``logsumexp(topk_j cos(f_i, r_j))``
    * **softmax** — ``sum_j w_j cos(f_i, r_j)``, ``w = softmax(topk cos)``

    The loss is the negated mean of the per-feature reduced similarities,
    so minimising it pushes each feature toward its nearest reference(s).

    Args:
        mode: ``"hard"`` | ``"logsumexp"`` | ``"softmax"``.
        topk: Top-k used by the soft modes.  Ignored when *mode* = ``"hard"``.
        chunk_size: Chunk size for :func:`chunked_cosine_sim`.
    """

    def __init__(
        self,
        mode: str = "hard",
        topk: int = 10,
        chunk_size: int = 2048,
    ) -> None:
        super().__init__()
        if mode not in {"hard", "logsumexp", "softmax"}:
            raise ValueError(f"mode must be 'hard'|'logsumexp'|'softmax', got {mode}")
        self.mode = mode
        self.topk = topk
        self.chunk_size = chunk_size

    def _reduce(self, sim: torch.Tensor) -> torch.Tensor:
        """Reduce a (chunk, n_refs) similarity matrix to (chunk,)."""
        if self.mode == "hard":
            return sim.max(dim=1)[0]
        topk_sim = sim.topk(self.topk, dim=1)[0]
        if self.mode == "logsumexp":
            return torch.logsumexp(topk_sim, dim=1)
        # softmax
        w = F.softmax(topk_sim, dim=1)
        return (w * topk_sim).sum(dim=1)

    def forward(
        self,
        features: torch.Tensor,
        references: torch.Tensor,
    ) -> torch.Tensor:
        """Compute anchor loss.

        Args:
            features: (n_features, dim) — e.g. ``decoder.weight.T``.
            references: (n_refs, dim) — e.g. ``embedding.weight``.

        Returns:
            Scalar loss (lower = better aligned).
        """
        sims = chunked_cosine_sim(
            features, references,
            reduce_fn=self._reduce,
            chunk_size=self.chunk_size,
        )
        return -sims.mean()
