from pathlib import Path

from vasae.models.factory import BlackBoxModelConfig

blackbox_model_cfg = BlackBoxModelConfig(
    name="gpt2",
    dir=Path(r"/scratch/b5bq/pu22650.b5bq/VASAE_out/BlackBoxModels/gpt2"),
)
