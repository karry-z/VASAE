"""
Dual-Path SAE (Section 3.4 of proposal).

Decomposes residual streams into:
  - Sparse token path: z @ W_E with L1 penalty, decoder frozen to embedding matrix
  - Dense path: y @ P_k^T with L1 penalty, decoder frozen to PCA basis of embedding-orthogonal residuals

Both decoders are frozen. Sparsity is enforced via L1 (no TopK).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DualPathSAEOutput:
    loss: torch.Tensor
    recon_loss: torch.Tensor
    l1_z: torch.Tensor
    l1_y: torch.Tensor
    h_recon: torch.Tensor
    z: torch.Tensor  # sparse token code [batch, vocab]
    y: torch.Tensor  # dense code [batch, d_pca]
    h_sparse: torch.Tensor  # z @ W_E


class DualPathSAE(nn.Module):
    """Dual-Path SAE faithful to proposal Section 3.4.

    Frozen buffers:
        W_E    [vocab, d]   — token embedding matrix
        P_k    [d, d_pca]   — PCA basis of embedding-orthogonal residuals
        mean_r [d]          — mean of embedding-orthogonal residuals

    Learnable:
        sparse_encoder: Linear(d → vocab)
        dense_encoder:  Linear(d → d_pca)
    """

    def __init__(self, dim_input: int, vocab_size: int, d_pca: int,
                 lambda_z: float = 1e-3, lambda_y: float = 1e-4):
        super().__init__()
        self.dim_input = dim_input
        self.vocab_size = vocab_size
        self.d_pca = d_pca
        self.lambda_z = lambda_z
        self.lambda_y = lambda_y

        self.sparse_encoder = nn.Linear(dim_input, vocab_size)
        self.dense_encoder = nn.Linear(dim_input, d_pca)

        # Frozen buffers — set via attach_*() methods
        self.register_buffer("W_E", torch.zeros(vocab_size, dim_input))
        self.register_buffer("P_k", torch.zeros(dim_input, d_pca))
        self.register_buffer("mean_r", torch.zeros(dim_input))

    @torch.no_grad()
    def attach_embedding(self, emb: nn.Embedding):
        """Attach frozen W_E from embedding layer."""
        self.W_E = emb.weight.detach().clone()  # [vocab, d]

    @torch.no_grad()
    def attach_pca(self, P_k: torch.Tensor, mean_r: torch.Tensor):
        """Attach frozen PCA basis and residual mean.

        P_k: [d, d_pca]
        mean_r: [d]
        """
        self.P_k = P_k.detach().clone()
        self.mean_r = mean_r.detach().clone()

    def forward(self, h: torch.Tensor) -> DualPathSAEOutput:
        # Sparse token path
        z = F.relu(self.sparse_encoder(h))  # [batch, vocab]
        h_sparse = z @ self.W_E  # [batch, d]

        # Dense path
        y = self.dense_encoder(h)  # [batch, d_pca]
        h_dense = y @ self.P_k.T  # [batch, d]

        # Reconstruction
        h_recon = h_sparse + h_dense + self.mean_r

        # Losses
        recon_loss = F.mse_loss(h_recon, h)
        l1_z = z.abs().mean()
        l1_y = y.abs().mean()
        loss = recon_loss + self.lambda_z * l1_z + self.lambda_y * l1_y

        return DualPathSAEOutput(
            loss=loss,
            recon_loss=recon_loss,
            l1_z=l1_z,
            l1_y=l1_y,
            h_recon=h_recon,
            z=z,
            y=y,
            h_sparse=h_sparse,
        )
