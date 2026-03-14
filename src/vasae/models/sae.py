from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput

from .encoders import LinearEncoder, MLPEncoder
from .sparsity import BatchTopKSparse, IdentitySparsity, TopKSparse


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
        elif config.sparsity_type == "topk":
            self.sparsity = TopKSparse(config.k, use_abs=config.use_abs_topk)
        else:
            self.sparsity = BatchTopKSparse(
                config.k, per_item_in_eval=config.per_item_in_eval, use_abs=config.use_abs_topk
            )

        # decoder
        self.decoder = nn.Linear(
            config.dim_sparse, config.dim_input, bias=(not config.tied_decoder)
        )

        self.decoder_lowrank = nn.Sequential(
            nn.Linear(config.dim_sparse, config.dim_sparse // 2),
            nn.Linear(config.dim_sparse // 2, config.dim_input),
        )

        # Store as plain attributes (not nn.Module submodules) to avoid
        # interfering with save_pretrained tied-weight detection.
        object.__setattr__(self, "_tied_embedding", None)
        object.__setattr__(self, "_anchor_embedding", None)
        if config.tied_decoder:
            self.decoder.weight.requires_grad_(False)
            if self.decoder.bias is not None:
                self.decoder.bias.requires_grad_(False)

        self.learnable_lowrank_coeff = nn.Parameter(torch.randn(config.dim_input))

        self.post_init()

    @torch.no_grad()
    def attach_embedding(self, embedding: nn.Embedding, freeze: bool = True):
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

        object.__setattr__(self, "_tied_embedding", embedding)
        self.config.tied_decoder = True

        self.decoder.weight = nn.Parameter(
            embedding.weight.T.contiguous(), requires_grad=(not freeze)
        )

        if self.decoder.bias is not None:
            self.decoder.bias.requires_grad_(not freeze)

    def attach_anchor_embedding(self, embedding: nn.Embedding):
        object.__setattr__(self, "_anchor_embedding", embedding)

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
        pre, z = self.encode(hidden_states)
        recon = self.decode(z)

        x = hidden_states
        xr = recon
        if x.ndim == 2:
            mse_per = F.mse_loss(xr, x, reduction="none").mean(dim=1)
        else:
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
            d_norm = F.normalize(self.decoder.weight.T, dim=1)
            e_norm = F.normalize(self._anchor_embedding.weight, dim=1)
            chunk_size = 2048
            max_sims = []
            for i in range(0, d_norm.size(0), chunk_size):
                chunk = d_norm[i:i + chunk_size]
                sim = chunk @ e_norm.T
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
