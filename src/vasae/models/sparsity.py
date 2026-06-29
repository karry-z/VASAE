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
