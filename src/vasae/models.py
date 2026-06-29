from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput


class LinearEncoder(nn.Module):
    def __init__(self, dim_input: int, dim_sparse: int):
        super().__init__()
        self.fc = nn.Linear(dim_input, dim_sparse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


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


@dataclass
class SAEOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    recon_loss: Optional[float] = None
    hidden_states_recon: Optional[torch.Tensor] = None
    sparse_activations: Optional[torch.Tensor] = None
    pre_activations: Optional[torch.Tensor] = None
    loss_per_sample: Optional[torch.Tensor] = None
    loss_anchor: Optional[torch.Tensor] = None


class SAEConfig(PretrainedConfig):
    model_type = "sae"

    def __init__(
        self,
        dim_input: int = 768,
        dim_sparse: int = 8192,
        k: int = 64,
        nonneg_latents: bool = True,
        mse_reduction: str = "mean",
        sae_save_path: Path | None = None,
        use_abs_topk: bool = False,
        decoder_mode: str = "learnable",
        anchor_coeff: float = 0.0,
        anchor_mode: str = "nearest",
        anchor_topk: int = 10,
        anchor_every: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim_input = int(dim_input)
        self.dim_sparse = int(dim_sparse)
        self.k = int(k)
        self.nonneg_latents = bool(nonneg_latents)
        self.mse_reduction = mse_reduction
        self.sae_save_path = sae_save_path
        self.use_abs_topk = bool(use_abs_topk)
        self.decoder_mode = decoder_mode
        self.anchor_coeff = float(anchor_coeff)
        self.anchor_mode = anchor_mode
        self.anchor_topk = int(anchor_topk)
        self.anchor_every = int(anchor_every)

        self._validate()

    def _validate(self):
        if self.dim_input <= 0 or self.dim_sparse <= 0:
            raise ValueError("dim_input and dim_sparse must be positive.")
        if self.k <= 0:
            raise ValueError("k must be > 0 for TopK sparsity.")
        if self.decoder_mode not in {"learnable", "hard_tied_baseline"}:
            raise ValueError(
                "decoder_mode must be 'learnable' or 'hard_tied_baseline', "
                f"got {self.decoder_mode}"
            )
        if self.anchor_mode not in {"nearest", "logsumexp", "softmax"}:
            raise ValueError(
                "anchor_mode must be 'nearest', 'logsumexp', or 'softmax', "
                f"got {self.anchor_mode}"
            )
        if self.anchor_topk <= 0:
            raise ValueError("anchor_topk must be positive.")
        if self.anchor_every <= 0:
            raise ValueError("anchor_every must be positive.")


class SAEModel(PreTrainedModel):
    config_class = SAEConfig
    base_model_prefix = "sae"

    def __init__(self, config: SAEConfig):
        super().__init__(config)

        self.encoder = LinearEncoder(config.dim_input, config.dim_sparse)
        self.sparsity = TopKSparse(config.k, use_abs=config.use_abs_topk)
        self.decoder = nn.Linear(
            config.dim_sparse,
            config.dim_input,
            bias=(config.decoder_mode == "learnable"),
        )

        object.__setattr__(self, "_tied_embedding", None)
        object.__setattr__(self, "_anchor_embedding", None)
        object.__setattr__(self, "_anchor_step_counter", 0)

        if config.decoder_mode == "hard_tied_baseline":
            self.decoder.weight.requires_grad_(False)
            if self.decoder.bias is not None:
                self.decoder.bias.requires_grad_(False)

        self.post_init()

    @torch.no_grad()
    def attach_tied_decoder_embedding(
        self, embedding: nn.Embedding, freeze: bool = True
    ) -> None:
        if self.config.decoder_mode != "hard_tied_baseline":
            raise ValueError(
                "attach_tied_decoder_embedding is only valid when "
                "decoder_mode='hard_tied_baseline'."
            )
        if self.config.dim_sparse != embedding.weight.size(0):
            raise ValueError(
                f"dim_sparse ({self.config.dim_sparse}) must equal embedding vocab size "
                f"({embedding.weight.size(0)}) for hard_tied_baseline."
            )
        if self.config.dim_input != embedding.weight.size(1):
            raise ValueError(
                f"dim_input ({self.config.dim_input}) must equal embedding dim "
                f"({embedding.weight.size(1)}) for hard_tied_baseline."
            )

        object.__setattr__(self, "_tied_embedding", embedding)
        self.decoder.weight = nn.Parameter(
            embedding.weight.T.contiguous(), requires_grad=(not freeze)
        )
        if self.decoder.bias is not None:
            self.decoder.bias.requires_grad_(not freeze)

    def attach_vocab_anchor(self, embedding: nn.Embedding) -> None:
        if self.config.dim_input != embedding.weight.size(1):
            raise ValueError(
                f"dim_input ({self.config.dim_input}) must equal embedding dim "
                f"({embedding.weight.size(1)}) for vocab anchoring."
            )
        object.__setattr__(self, "_anchor_embedding", embedding)

    def encode(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pre = self.encoder(hidden_states)
        pre_for_topk = F.relu(pre) if self.config.nonneg_latents else pre
        z = self.sparsity(pre_for_topk)
        return pre, z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def _anchor_loss(self) -> Optional[torch.Tensor]:
        if self.config.anchor_coeff <= 0 or self._anchor_embedding is None or not self.training:
            return None

        self._anchor_step_counter += 1
        if self._anchor_step_counter % self.config.anchor_every != 0:
            return None

        d_norm = F.normalize(self.decoder.weight.T, dim=1)
        e_norm = F.normalize(self._anchor_embedding.weight.to(d_norm.dtype), dim=1)
        topk = min(self.config.anchor_topk, e_norm.size(0))

        chunk_size = 2048
        sims = []
        for i in range(0, d_norm.size(0), chunk_size):
            chunk = d_norm[i : i + chunk_size]
            sim = chunk @ e_norm.T
            if self.config.anchor_mode == "nearest":
                sims.append(sim.max(dim=1)[0])
            elif self.config.anchor_mode == "logsumexp":
                sims.append(torch.logsumexp(sim.topk(topk, dim=1)[0], dim=1))
            else:
                topk_sim = sim.topk(topk, dim=1)[0]
                weights = F.softmax(topk_sim, dim=1)
                sims.append((weights * topk_sim).sum(dim=1))

        return -torch.cat(sims).mean()

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_dict: bool = True,
        output_pre_activations: bool = False,
        output_loss_per_sample: bool = True,
    ) -> SAEOutput:
        pre, z = self.encode(hidden_states)
        recon = self.decode(z)

        if hidden_states.ndim == 2:
            mse_per = F.mse_loss(recon, hidden_states, reduction="none").mean(dim=1)
        else:
            mse_per = F.mse_loss(recon, hidden_states, reduction="none").mean(dim=-1)

        recon_loss_tensor = mse_per.mean()
        total_loss = recon_loss_tensor

        loss_anchor = self._anchor_loss()
        if loss_anchor is not None:
            total_loss = total_loss + self.config.anchor_coeff * loss_anchor

        if not return_dict:
            outs = (recon, z)
            if output_pre_activations:
                outs = (pre,) + outs
            return (total_loss,) + outs

        return SAEOutput(
            loss=total_loss,
            recon_loss=recon_loss_tensor.detach().cpu().item(),
            hidden_states_recon=recon,
            sparse_activations=z,
            pre_activations=(pre if output_pre_activations else None),
            loss_per_sample=(mse_per if output_loss_per_sample else None),
            loss_anchor=loss_anchor,
        )
