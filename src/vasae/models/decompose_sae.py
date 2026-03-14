from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import LinearEncoder
from .sparsity import TopKSparse


@dataclass
class DecomposeSAEOutput:
    loss: torch.Tensor
    h_recon: torch.Tensor
    z_s: torch.Tensor
    z_d: torch.Tensor
    h_sparse: torch.Tensor


class DecomposeSAEModel(nn.Module):
    def __init__(self, dim_input: int, dim_sparse: int, d_pca: int, k: int):
        super().__init__()
        self.dim_input = dim_input
        self.dim_sparse = dim_sparse
        self.d_pca = d_pca
        self.k = k

        self.sparse_encoder = LinearEncoder(dim_input, dim_sparse)
        self.sparsity = TopKSparse(k)
        self.dense_encoder = nn.Linear(dim_input, d_pca)
        self.decoder_sparse = nn.Linear(dim_sparse, dim_input, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim_input))

        # PCA directions, set via attach_pca()
        self.register_buffer("W", None)

    @torch.no_grad()
    def attach_embedding(self, emb: nn.Embedding, freeze: bool = True):
        self.decoder_sparse.weight = nn.Parameter(
            emb.weight.T, requires_grad=not freeze
        )

    def attach_pca(self, W: torch.Tensor):
        """W: [dim_input, d_pca] — frozen PCA directions."""
        self.W = W

    def forward(self, h: torch.Tensor) -> DecomposeSAEOutput:
        # sparse path
        z_s = self.sparsity(F.relu(self.sparse_encoder(h)))
        h_sparse = self.decoder_sparse(z_s)

        # dense path
        z_d = self.dense_encoder(h)
        h_dense = z_d @ self.W.T

        # combine
        h_recon = h_sparse + h_dense + self.bias
        loss = F.mse_loss(h_recon, h)

        return DecomposeSAEOutput(
            loss=loss, h_recon=h_recon, z_s=z_s, z_d=z_d, h_sparse=h_sparse
        )
