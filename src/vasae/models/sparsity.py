import torch
import torch.nn as nn


class TopKSparse(nn.Module):
    def __init__(self, k: int, use_abs: bool = False):
        super().__init__()
        self.k = int(k)
        self.use_abs = bool(use_abs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = min(self.k, x.size(-1))
        if self.use_abs:
            _, idx = torch.topk(torch.abs(x), k, dim=-1)
        else:
            _, idx = torch.topk(x, k, dim=-1)
        mask = torch.zeros_like(x)
        mask.scatter_(-1, idx, 1.0)
        return x * mask


class BatchTopKSparse(nn.Module):
    """
    Keeps top (k * n_items) activations globally across all items in the batch (or batch*time),
    where item = each slice along last dim.

    If per_item_in_eval=True, uses per-item topk during eval to avoid cross-sample coupling.
    """

    def __init__(self, k: int, per_item_in_eval: bool = False, use_abs: bool = False):
        super().__init__()
        self.k = int(k)
        self.per_item_in_eval = bool(per_item_in_eval)
        self.use_abs = bool(use_abs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) and self.per_item_in_eval:
            k = min(self.k, x.size(-1))
            if self.use_abs:
                _, idx = torch.topk(torch.abs(x), k, dim=-1)
            else:
                _, idx = torch.topk(x, k, dim=-1)
            mask = torch.zeros_like(x)
            mask.scatter_(-1, idx, 1.0)
            return x * mask

        d = x.size(-1)
        n_items = x.numel() // d
        k_total = self.k * n_items
        k_total = min(k_total, x.numel())  # safety

        flat = x.reshape(-1)
        if self.use_abs:
            _, idx = torch.topk(torch.abs(flat), k_total, sorted=False)
        else:
            _, idx = torch.topk(flat, k_total, sorted=False)
        mask = torch.zeros_like(flat)
        mask[idx] = 1.0
        return (flat * mask).reshape_as(x)


class IdentitySparsity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
