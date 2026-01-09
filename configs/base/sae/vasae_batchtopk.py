from pathlib import Path

from vasae.models.sae_hf import SAEConfig

sae_cfg = SAEConfig(
    encoder_type="linear",
    sparsity_type="batch_topk",
    k=8,
    per_item_in_eval=False,
    nonneg_latents=True,
    l1_coeff=0,
    tied_decoder=True,
)
