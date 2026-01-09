import runpy
from pathlib import Path

config_dir = Path(r"/home/b5bq/pu22650.b5bq/work/VASAE/configs")
exp_name = Path(__file__).stem

sae_cfg = runpy.run_path(str(config_dir / "base" / "sae" / "sae_relu.py"))["sae_cfg"]
sae_cfg.sae_save_path = (
    f"/home/b5bq/pu22650.b5bq/work/VASAE/configs/exp/compare/sae_relu_l0.pth"
)

data_cfg = runpy.run_path(str(config_dir / "base" / "data.py"))["data_cfg"]
data_cfg.data_dir = Path(
    r"/scratch/b5bq/pu22650.b5bq/activations_gpt2_Geralt-Targaryen_openwebtext2"
)
data_cfg.layer_name = "transformer.h.0"

train_cfg = runpy.run_path(str(config_dir / "base" / "train.py"))["train_cfg"]

system_cfg = runpy.run_path(str(config_dir / "base" / "system.py"))["system_cfg"]
system_cfg["wandb"] = False

blackbox_model_cfg = runpy.run_path(
    str(config_dir / "base" / "blackbox_model" / "gpt2.py")
)["blackbox_model_cfg"]
