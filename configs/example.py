import runpy
from pathlib import Path

config_dir = Path(__file__).parent


sae_cfg = runpy.run_path(str(config_dir / "base" / "sae" / "vasae.py"))["sae_cfg"]
sae_cfg.sae_save_path = "/home/b5bq/pu22650.b5bq/work/VASAE/out/sae.pth"

data_cfg = runpy.run_path(str(config_dir / "base" / "data.py"))["data_cfg"]
data_cfg.meta_path = r"/scratch/b5bq/pu22650.b5bq/activations_gpt2_Geralt-Targaryen_openwebtext2/meta.json"

train_cfg = runpy.run_path(str(config_dir / "base" / "train.py"))["train_cfg"]

system_cfg = runpy.run_path(str(config_dir / "base" / "system.py"))["system_cfg"]

blackbox_model_cfg = runpy.run_path(
    str(config_dir / "base" / "blackbox_model" / "gpt2.py")
)["blackbox_model_cfg"]

exp_name = "test"
