import torch
import torch.nn as nn


class VASAE(nn.Module):
    def __init__(self, topk=4, emb_weight=None):
        super().__init__()
        self.encoder = nn.Linear(emb_weight.size(1), emb_weight.size(0))
        self.decoder = nn.Linear(emb_weight.size(0), emb_weight.size(1))
        self.decoder.requires_grad_(False)
        self.decoder.weight = nn.Parameter(emb_weight.T)
        self.k = topk

    def k_sparse(self, x):
        # 实现k-sparse约束
        topk, indices = torch.topk(x, self.k)
        mask = torch.zeros_like(x).scatter_(-1, indices, 1)
        return x * mask

    def forward(self, x):
        x = self.encoder(x)
        z = self.k_sparse(x)
        x_rec = self.decoder(z)
        return x_rec, z
