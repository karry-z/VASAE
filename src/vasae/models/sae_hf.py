from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput


# -------------------------
# Sparsity modules
# -------------------------
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


# -------------------------
# Encoders
# -------------------------
class LinearEncoder(nn.Module):
    def __init__(self, dim_input: int, dim_sparse: int):
        super().__init__()
        self.fc = nn.Linear(dim_input, dim_sparse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLPEncoder(nn.Module):
    def __init__(self, dim_input: int, dim_sparse: int, hidden_mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_input, dim_input * hidden_mult),
            nn.GELU(),
            nn.Linear(dim_input * hidden_mult, dim_sparse),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------
# HF Output
# -------------------------
@dataclass
class SAEOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    recon_loss: Optional[torch.Tensor] = None
    l1_loss: Optional[torch.Tensor] = None
    hidden_states_recon: Optional[torch.Tensor] = None
    sparse_activations: Optional[torch.Tensor] = None
    pre_activations: Optional[torch.Tensor] = None
    loss_per_sample: Optional[torch.Tensor] = None
    loss_lowrank: Optional[float] = None
    loss_anchor: Optional[torch.Tensor] = None


# -------------------------
# Config
# -------------------------
class SAEConfig(PretrainedConfig):
    model_type = "sae"

    def __init__(
        self,
        dim_input: int = 768,
        dim_sparse: int = 8192,
        encoder_type: str = "linear",  # "linear" | "mlp"
        sparsity_type: str = "none",  # "none" | "topk" | "batch_topk"
        k: int = 0,  # used if sparsity_type != "none"
        per_item_in_eval: bool = False,  # only for batch_topk
        nonneg_latents: bool = True,  # apply ReLU on pre_activations before sparsity
        l1_coeff: float = 0.0,  # only meaningful if sparsity_type == "none"
        tied_decoder: bool = False,  # if True, use attach_embedding() to tie
        mse_reduction: str = "mean",  # "mean" or "none" (we still provide loss_per_sample)
        sae_save_path: Path | None = None,
        freeze_decoder: bool = True,
        use_lowrank: bool = True,
        lowrank_coeff: float = 0.1,
        use_abs_topk: bool = False,  # use absolute value for topk selection
        anchor_coeff: float = 0.0,  # weak token-anchoring regularizer coefficient
        anchor_mode: str = "hard",  # "hard" | "logsumexp" | "softmax"
        anchor_topk: int = 10,  # top-k for soft anchor modes
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim_input = int(dim_input)
        self.dim_sparse = int(dim_sparse)
        self.encoder_type = encoder_type
        self.sparsity_type = sparsity_type
        self.k = int(k)
        self.per_item_in_eval = bool(per_item_in_eval)
        self.nonneg_latents = bool(nonneg_latents)
        self.l1_coeff = float(l1_coeff)
        self.tied_decoder = bool(tied_decoder)
        self.mse_reduction = mse_reduction
        self.sae_save_path = sae_save_path
        self.freeze_decoder = freeze_decoder
        self.use_lowrank = use_lowrank
        self.lowrank_coeff = lowrank_coeff
        self.use_abs_topk = bool(use_abs_topk)
        self.anchor_coeff = float(anchor_coeff)
        self.anchor_mode = anchor_mode
        self.anchor_topk = int(anchor_topk)

        self._validate()

    def _validate(self):
        if self.encoder_type not in {"linear", "mlp"}:
            raise ValueError(
                f"encoder_type must be 'linear' or 'mlp', got {self.encoder_type}"
            )
        if self.sparsity_type not in {"none", "topk", "batch_topk"}:
            raise ValueError(
                f"sparsity_type must be 'none'|'topk'|'batch_topk', got {self.sparsity_type}"
            )
        if self.sparsity_type != "none" and self.k <= 0:
            raise ValueError("k must be > 0 when using topk/batch_topk")
        if self.sparsity_type in {"topk", "batch_topk"} and self.l1_coeff > 0:
            raise ValueError("Do not use L1 with topk/batch_topk. Set l1_coeff=0.")
        if self.dim_input <= 0 or self.dim_sparse <= 0:
            raise ValueError("dim_input and dim_sparse must be positive.")
        if self.anchor_mode not in {"hard", "logsumexp", "softmax"}:
            raise ValueError(
                f"anchor_mode must be 'hard'|'logsumexp'|'softmax', got {self.anchor_mode}"
            )


# -------------------------
# Model
# -------------------------
class SAEModel(PreTrainedModel):
    config_class = SAEConfig
    base_model_prefix = "sae"

    def __init__(self, config: SAEConfig):
        super().__init__(config)

        # encoder
        if config.encoder_type == "linear":
            self.encoder = LinearEncoder(config.dim_input, config.dim_sparse)
        else:
            self.encoder = MLPEncoder(config.dim_input, config.dim_sparse)

        # sparsity
        if config.sparsity_type == "none":
            self.sparsity = IdentitySparsity()
            # relu
        elif config.sparsity_type == "topk":
            self.sparsity = TopKSparse(config.k, use_abs=config.use_abs_topk)
        else:
            self.sparsity = BatchTopKSparse(
                config.k, per_item_in_eval=config.per_item_in_eval, use_abs=config.use_abs_topk
            )

        # decoder
        # Note: tied-decoder weight is attached later; we still create a module for shape & save/load.
        self.decoder = nn.Linear(
            config.dim_sparse, config.dim_input, bias=(not config.tied_decoder)
        )

        self.decoder_lowrank = nn.Sequential(
            nn.Linear(config.dim_sparse, config.dim_sparse // 2),
            nn.Linear(config.dim_sparse // 2, config.dim_input),
        )

        self._tied_embedding: Optional[nn.Embedding] = None
        self._anchor_embedding: Optional[nn.Embedding] = None
        if config.tied_decoder:
            # We'll freeze decoder weight by default; actual tying via attach_embedding() TODO
            self.decoder.weight.requires_grad_(False)
            if self.decoder.bias is not None:
                self.decoder.bias.requires_grad_(False)  # TODO：fix or not

        self.learnable_lowrank_coeff = nn.Parameter(torch.randn(config.dim_input))

        self.post_init()

    @torch.no_grad()
    def attach_embedding(self, embedding: nn.Embedding, freeze: bool = True):
        """
        Tie decoder weights to embedding.weight.T (VASAE-style).
        embedding.weight: (vocab, dim_input) => decoder.weight: (dim_input, dim_sparse) expects dim_sparse == vocab.
        """
        if self.config.dim_sparse != embedding.weight.size(0):
            raise ValueError(
                f"dim_sparse ({self.config.dim_sparse}) must equal embedding vocab size "
                f"({embedding.weight.size(0)}) to tie decoder."
            )
        if self.config.dim_input != embedding.weight.size(1):
            raise ValueError(
                f"dim_input ({self.config.dim_input}) must equal embedding dim "
                f"({embedding.weight.size(1)}) to tie decoder."
            )

        self._tied_embedding = embedding
        self.config.tied_decoder = True

        # Make decoder weight a view/copy of embedding^T via Parameter that points to same storage is tricky.
        # The simplest robust way: assign a new Parameter with copied data and keep it synced if needed.
        # For VASAE you usually freeze, so sync isn't needed.
        self.decoder.weight = nn.Parameter(
            embedding.weight.T, requires_grad=(not freeze)
        )

        # Bias typically off for tied decoder; keep consistent.
        if self.decoder.bias is not None:
            self.decoder.bias.requires_grad_(not freeze)

    def attach_anchor_embedding(self, embedding: nn.Embedding):
        """Store embedding reference for anchor loss (does not modify decoder weights)."""
        self._anchor_embedding = embedding

    def encode(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pre = self.encoder(hidden_states)
        if self.config.nonneg_latents:
            pre_nonneg = F.relu(pre)
        else:
            pre_nonneg = pre
        z = self.sparsity(pre_nonneg)
        return pre, z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.decoder(z)
        if self.config.use_lowrank:
            out += self.config.lowrank_coeff * self.decoder_lowrank(z)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_dict: bool = True,
        output_pre_activations: bool = False,
        output_loss_per_sample: bool = True,
    ) -> SAEOutput:
        """
        hidden_states: (..., dim_input)
        """
        pre, z = self.encode(hidden_states)
        recon = self.decode(z)

        # loss per sample (reduce over all dims except leading batch-like dims)
        # For general (..., dim), treat the first dim as batch and reduce the rest.
        # If you pass (B,T,D), this yields per-item per-batch? -> we produce loss_per_sample over the first dim only by default.
        # To keep it predictable: collapse everything except first dim.
        x = hidden_states
        xr = recon
        if x.ndim == 2:
            # (B,D)
            mse_per = F.mse_loss(xr, x, reduction="none").mean(dim=1)  # (B,)
        else:
            # (B, ... , D) -> mean over dims 1..end
            mse_per = F.mse_loss(xr, x, reduction="none").mean(dim=-1)

        recon_loss = mse_per.mean()

        l1_loss = None
        total_loss = recon_loss
        recon_loss = recon_loss.detach().cpu().item()
        if self.config.l1_coeff > 0:
            l1_loss = z.abs().mean()
            total_loss = total_loss + self.config.l1_coeff * l1_loss
            l1_loss = l1_loss.detach().cpu().item()

        # anchor loss
        loss_anchor = None
        if self.config.anchor_coeff > 0 and self._anchor_embedding is not None:
            d_norm = F.normalize(self.decoder.weight.T, dim=1)  # (dim_sparse, dim_input)
            e_norm = F.normalize(self._anchor_embedding.weight, dim=1)  # (vocab, dim_input)
            chunk_size = 2048
            max_sims = []
            for i in range(0, d_norm.size(0), chunk_size):
                chunk = d_norm[i:i + chunk_size]
                sim = chunk @ e_norm.T  # (chunk, vocab)
                if self.config.anchor_mode == "hard":
                    max_sims.append(sim.max(dim=1)[0])
                elif self.config.anchor_mode == "logsumexp":
                    topk_sim = sim.topk(self.config.anchor_topk, dim=1)[0]
                    max_sims.append(torch.logsumexp(topk_sim, dim=1))
                elif self.config.anchor_mode == "softmax":
                    topk_sim = sim.topk(self.config.anchor_topk, dim=1)[0]
                    w = F.softmax(topk_sim, dim=1)
                    max_sims.append((w * topk_sim).sum(dim=1))
            loss_anchor = -torch.cat(max_sims).mean()
            total_loss = total_loss + self.config.anchor_coeff * loss_anchor

        if not return_dict:
            outs = (recon, z)
            if output_pre_activations:
                outs = (pre,) + outs
            return (total_loss,) + outs

        return SAEOutput(
            loss=total_loss,
            recon_loss=recon_loss,
            l1_loss=l1_loss,
            hidden_states_recon=recon,
            sparse_activations=z,
            pre_activations=(pre if output_pre_activations else None),
            loss_per_sample=(mse_per if output_loss_per_sample else None),
            loss_lowrank=(None),
            loss_anchor=loss_anchor,
        )


# -------------------------
# Quick usage examples
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # 1) Vanilla SAE (ReLU + optional L1)
    cfg = SAEConfig(
        dim_input=16,
        dim_sparse=64,
        encoder_type="linear",
        sparsity_type="none",
        nonneg_latents=True,
        l1_coeff=1e-3,
        tied_decoder=False,
    )
    m = SAEModel(cfg)
    x = torch.randn(8, 16)
    out = m(x, output_pre_activations=True)
    print(
        "vanilla loss:",
        float(out.loss),
        "recon:",
        float(out.recon_loss),
        "l1:",
        float(out.l1_loss),
    )

    # 2) TopK SAE (no L1)
    cfg2 = SAEConfig(
        dim_input=16,
        dim_sparse=64,
        encoder_type="linear",
        sparsity_type="topk",
        k=4,
        nonneg_latents=True,
        l1_coeff=0.0,
    )
    m2 = SAEModel(cfg2)
    out2 = m2(x)
    print("topk loss:", float(out2.loss))

    # 3) BatchTopK SAE (train batch-coupled, eval per-item optional)
    cfg3 = SAEConfig(
        dim_input=16,
        dim_sparse=64,
        encoder_type="mlp",
        sparsity_type="batch_topk",
        k=4,
        per_item_in_eval=True,
        nonneg_latents=True,
        l1_coeff=0.0,
    )
    m3 = SAEModel(cfg3)
    m3.train()
    out3 = m3(x)
    m3.eval()
    out3_eval = m3(x)
    print(
        "batch_topk train loss:", float(out3.loss), "eval loss:", float(out3_eval.loss)
    )

    # 4) Save / load
    # m3.save_pretrained("./sae_ckpt")
    # m3_loaded = SAEModel.from_pretrained("./sae_ckpt")
