#!/bin/bash

#SBATCH --job-name=002F_llama_high_lambda
#SBATCH --output=exp/F002_AlignmentAnalysis/logs/%x_%j.log
#SBATCH --gpus=1
#SBATCH --time=0:30:00

cd ~/work/VASAE
VENV_SITE="$(uv run python -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="${VENV_SITE}/nvidia/cusparselt/lib:${VENV_SITE}/nvidia/cusparse/lib:${LD_LIBRARY_PATH}"

echo "=== 002_F: Geometric alignment check for Llama λ=5e-3 (L0, L15, L31) ==="
echo "Started on $(date)"

uv run python -c "
import torch
from vasae.analysis.alignment import compute_geometric_alignment
from vasae.analysis.sae_loader import load_sae_for_analysis, get_decoder_features
from vasae.models.factory import load_model, get_embedding
from shared_utils.log import get_logger

log = get_logger('high_lambda_check')
device = torch.device('cuda')

log.info('Loading Llama-3.1-8B...')
lm_model, tokenizer = load_model('meta-llama/Llama-3.1-8B', device=device)
W_E = get_embedding(lm_model).weight.data
log.info('vocab_size=%d, embed_dim=%d', W_E.shape[0], W_E.shape[1])

base = '/scratch/b5bq/pu22650.b5bq/VASAE_out/001A_F_AblationSoft'
checkpoints = {
    0:  f'{base}/001AF_llama_lambda_L0_a5e-3',
    15: f'{base}/001AF_llama_lambda_L15_a5e-3',
    31: f'{base}/001AF_llama_lambda_L31_a5e-3',
}

for layer_idx, path in checkpoints.items():
    log.info('--- Layer %d (lambda=5e-3) ---', layer_idx)
    sae = load_sae_for_analysis(path, device=device)
    features = get_decoder_features(sae)
    log.info('  n_features=%d', features.shape[0])

    geo = compute_geometric_alignment(features, W_E, top_k=1, device=device)
    sims = geo.max_sims

    n_aligned = (sims >= 0.8).sum().item()
    n_total = sims.shape[0]
    log.info('  max_sim: mean=%.4f, median=%.4f, max=%.4f', sims.mean(), sims.median(), sims.max())
    log.info('  aligned (s>=0.8): %d/%d (%.2f%%)', n_aligned, n_total, n_aligned/n_total*100)

    del sae, features, geo
    torch.cuda.empty_cache()

log.info('Done.')
"

echo "Finished on $(date)"
