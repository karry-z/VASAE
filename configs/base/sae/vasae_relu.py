from vasae.models.sae_hf import SAEConfig

sae_cfg = SAEConfig(
    encoder_type="linear",
    sparsity_type="none",
    k=0,
    per_item_in_eval=False,
    nonneg_latents=True,
    l1_coeff=1e-3,
    tied_decoder=True,
)
