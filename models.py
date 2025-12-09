import torch
import torch.nn as nn
import torch.nn.functional as F


class KSparse(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        _, idx = torch.topk(x, self.k, dim=-1)
        mask = torch.zeros_like(x)
        mask.scatter_(-1, idx, 1.0)
        return x * mask


class BatchKSparse(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        x_shape = x.shape
        x_flat = x.reshape(-1)
        _, idx = torch.topk(x_flat, self.k)
        mask = torch.zeros_like(x_flat)
        mask[idx] = 1.0
        return (x_flat * mask).reshape(x_shape)


class SAEEncoder(nn.Module):
    def __init__(self, dim_input, dim_sparse, act_fn):
        super().__init__()
        self.fc = nn.Linear(dim_input, dim_sparse)
        self.act_fn = act_fn

    def forward(self, x):
        pre_activation = self.fc(x)
        z = self.act_fn(pre_activation)
        return pre_activation, z


class VanillaSAE(nn.Module):
    def __init__(self, dim_input, dim_sparse, l1_coeff=1e-3):
        super().__init__()
        self.encoder = SAEEncoder(dim_input, dim_sparse, nn.ReLU())
        self.decoder = nn.Linear(dim_sparse, dim_input)
        self.l1_coeff = l1_coeff

    def forward(self, x):
        _, z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def compute_loss(self, x, x_recon, z):
        recon_loss = F.mse_loss(x_recon, x)
        l1_loss = z.abs().mean()
        loss = recon_loss + self.l1_coeff * l1_loss
        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "l1_loss": l1_loss,
        }


class TopKSAE(nn.Module):
    def __init__(self, dim_input, dim_sparse, k):
        super().__init__()
        self.encoder = SAEEncoder(dim_input, dim_sparse, act_fn=KSparse(k))
        self.decoder = nn.Linear(dim_sparse, dim_input)
        self.k = k

    def forward(self, x):
        _, z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def compute_loss(self, x, x_recon, z):
        recon_loss = F.mse_loss(x_recon, x)
        return {"loss": recon_loss}


class BatchTopKSAE(nn.Module):
    """
    https://github.com/bartbussmann/BatchTopK/blob/main/sae.py
    """

    def __init__(self, dim_input, dim_sparse, k):
        super().__init__()
        self.encoder = SAEEncoder(dim_input, dim_sparse, act_fn=BatchKSparse(k))
        self.decoder = nn.Linear(dim_sparse, dim_input)
        self.k = k

    def forward(self, x):
        _, z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def compute_loss(self, x, x_recon, z):
        recon_loss = F.mse_loss(x_recon, x)
        return {"loss": recon_loss}


class VASAE(nn.Module):
    def __init__(self, k=4, embedding_weight=None, act_fn=None):
        super().__init__()
        dim_input = embedding_weight.size(1)
        dim_sparse = embedding_weight.size(0)

        if act_fn is None:
            act_fn = BatchKSparse(k)
        self.encoder = SAEEncoder(dim_input, dim_sparse, act_fn=act_fn)

        self.decoder = nn.Linear(dim_sparse, dim_input, bias=False)
        # fixed decoder tied to embedding matrix
        self.decoder.weight = nn.Parameter(
            embedding_weight.T,
            requires_grad=False,
        )

    def forward(self, x):
        _, z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def compute_loss(self, x, x_recon, z):
        loss_per_sample = F.mse_loss(x_recon, x, reduction="none").mean(
            dim=list(range(1, x_recon.ndim))
        )  # mean over all dims except batch
        recon_loss = loss_per_sample.mean()
        return {"loss": recon_loss, "loss_per_sample": loss_per_sample}
