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
    """
    BatchTopK-style k-sparse activation.

    Interprets k as "average number of active latents per item" (item = each slice
    along the last dim). For input shape (..., d), let n_items = prod(shape[:-1]).
    Keeps top (k * n_items) activations globally across the flattened tensor,
    then reshapes back.

    Optional: in eval mode, you may want per-item TopK to avoid cross-sample coupling.
    """

    def __init__(self, k: int, per_item_in_eval: bool = False):
        super().__init__()
        self.k = int(k)
        self.per_item_in_eval = per_item_in_eval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard SAE practice: non-negative latents
        # x = F.relu(x)

        # Optional: make inference independent per item
        if (not self.training) and self.per_item_in_eval:
            # per-item topk over last dim
            _, idx = torch.topk(x, k=min(self.k, x.size(-1)), dim=-1)
            mask = torch.zeros_like(x)
            mask.scatter_(-1, idx, 1.0)
            return x * mask

        # compute k_total = B (or B*T when T dim exists) * topk
        d = x.size(-1)
        n_items = x.numel() // d  # e.g. B or B*T
        k_total = self.k * n_items
        flat = x.reshape(
            -1
        )  # topk works on one-dim, so flatten x to pick topk globally.

        _, idx = torch.topk(flat, k_total, sorted=False)
        mask = torch.zeros_like(flat)
        mask[idx] = 1.0
        return (flat * mask).reshape_as(x)


class SAEEncoder(nn.Module):
    def __init__(self, dim_input, dim_sparse, act_fn):
        super().__init__()
        self.fc = nn.Linear(dim_input, dim_sparse)
        self.act_fn = act_fn

    def forward(self, x):
        pre_activation = self.fc(x)
        z = self.act_fn(pre_activation)
        return pre_activation, z


class SAEEncoderMLP(nn.Module):
    def __init__(self, dim_input, dim_sparse, act_fn):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_input, dim_input * 4),
            nn.GELU(),
            nn.Linear(dim_input * 4, dim_sparse),
        )
        self.act_fn = act_fn

    def forward(self, x):
        pre_activation = self.mlp(x)
        z = self.act_fn(pre_activation)
        return pre_activation, z


class SAEBase(nn.Module):
    def __init__(self, encoder: SAEEncoder, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        _, z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def from_pretrained(self, path):
        self.load_state_dict(torch.load(path))
        return self


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
        loss_per_sample = F.mse_loss(x_recon, x, reduction="none").mean(
            dim=list(range(1, x_recon.ndim))
        )  # mean over all dims except batch
        recon_loss = loss_per_sample.mean()
        l1_loss = z.abs().mean()
        loss = recon_loss + self.l1_coeff * l1_loss
        return {
            "loss": loss,
            "loss_per_sample": loss_per_sample,
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
        loss_per_sample = F.mse_loss(x_recon, x, reduction="none").mean(
            dim=list(range(1, x_recon.ndim))
        )  # mean over all dims except batch
        recon_loss = loss_per_sample.mean()
        return {"loss": recon_loss, "loss_per_sample": loss_per_sample}


class BatchTopKSAE(nn.Module):
    """
    https://github.com/bartbussmann/BatchTopK/blob/main/sae.py
    """

    def __init__(self, dim_input, dim_sparse, k, per_item_in_eval: bool = False):
        super().__init__()
        self.encoder = SAEEncoder(
            dim_input,
            dim_sparse,
            act_fn=BatchKSparse(k, per_item_in_eval=per_item_in_eval),
        )
        self.decoder = nn.Linear(dim_sparse, dim_input)
        self.k = k

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


class VASAE(nn.Module):
    def __init__(self, k=4, embedding_weight=None, act_fn=None, per_item_in_eval=False):
        super().__init__()
        dim_input = embedding_weight.size(1)
        dim_sparse = embedding_weight.size(0)

        if act_fn is None:
            act_fn = BatchKSparse(k, per_item_in_eval)
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


class VASAEMLP(nn.Module):
    def __init__(self, k=4, embedding_weight=None, act_fn=None, per_item_in_eval=False):
        super().__init__()
        dim_input = embedding_weight.size(1)
        dim_sparse = embedding_weight.size(0)

        if act_fn is None:
            act_fn = BatchKSparse(k, per_item_in_eval)
        self.encoder = SAEEncoderMLP(dim_input, dim_sparse, act_fn=act_fn)

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


class VASAE_ReLU(SAEBase):
    def __init__(self, embedding_weight, l1_coeff=1e-3):
        dim_input = embedding_weight.size(1)
        dim_sparse = embedding_weight.size(0)

        encoder = SAEEncoder(dim_input, dim_sparse, nn.ReLU())
        decoder = nn.Linear(dim_sparse, dim_input)

        decoder.weight = nn.Parameter(
            embedding_weight.T,
            requires_grad=False,
        )

        super().__init__(encoder, decoder)
        self.l1_coeff = l1_coeff

    def compute_loss(self, x, x_recon, z):
        loss_per_sample = F.mse_loss(x_recon, x, reduction="none").mean(
            dim=list(range(1, x_recon.ndim))
        )  # mean over all dims except batch
        recon_loss = loss_per_sample.mean()
        l1_loss = z.abs().mean()
        loss = recon_loss + self.l1_coeff * l1_loss
        return {
            "loss": loss,
            "loss_per_sample": loss_per_sample,
            "recon_loss": recon_loss,
            "l1_loss": l1_loss,
        }


class VASAE_LearnedDecoder(nn.Module):
    def __init__(
        self,
        k=4,
        embedding_weight=None,
        act_fn=None,
        per_item_in_eval=False,
        lambda_cos=0.1,
    ):
        super().__init__()
        dim_input = embedding_weight.size(1)
        dim_sparse = embedding_weight.size(0)

        if act_fn is None:
            act_fn = BatchKSparse(k, per_item_in_eval)
        self.encoder = SAEEncoder(dim_input, dim_sparse, act_fn=act_fn)

        self.decoder = nn.Linear(dim_sparse, dim_input, bias=False)

        self.decoder.weight = nn.Parameter(
            embedding_weight.T,
        )

        self.emb_weight = embedding_weight.T
        self.lambda_cos = lambda_cos

    def forward(self, x):
        _, z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def compute_loss(self, x, x_recon, z):
        loss_per_sample = F.mse_loss(x_recon, x, reduction="none").mean(
            dim=list(range(1, x_recon.ndim))
        )  # mean over all dims except batch
        recon_loss = loss_per_sample.mean()

        cos_sim = F.cosine_similarity(self.emb_weight, self.decoder.weight, dim=0)
        cos_loss = 1 - cos_sim.mean()
        loss = recon_loss + self.lambda_cos * cos_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "loss_per_sample": loss_per_sample,
            "cos_loss": cos_loss,
        }
