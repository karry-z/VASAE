import torch
import torch.nn as nn


class LinearEncoder(nn.Module):
    def __init__(self, dim_model: int, dim_sparse: int):
        super().__init__()
        self.fc = nn.Linear(dim_model, dim_sparse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLPEncoder(nn.Module):
    def __init__(self, dim_model: int, dim_sparse: int, hidden_mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_model, dim_model * hidden_mult),
            nn.GELU(),
            nn.Linear(dim_model * hidden_mult, dim_sparse),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
