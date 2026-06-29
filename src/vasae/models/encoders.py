import torch
import torch.nn as nn


class LinearEncoder(nn.Module):
    def __init__(self, dim_input: int, dim_sparse: int):
        super().__init__()
        self.fc = nn.Linear(dim_input, dim_sparse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
