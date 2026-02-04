from huggingface_hub import snapshot_download

MODEL_ID = "openai/clip-vit-base-patch32"
CACHE_DIR = "/scratch/b5bq/pu22650.b5bq/hf_cache"  # Use pre-downloaded models
model_path = snapshot_download(repo_id=MODEL_ID, cache_dir=CACHE_DIR)
