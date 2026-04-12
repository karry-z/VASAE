#!/bin/bash

#SBATCH --job-name=002F_llama_5e-3_save
#SBATCH --output=exp/F002_AlignmentAnalysis/logs/%x_%j.log
#SBATCH --gpus=1
#SBATCH --time=0:30:00

cd ~/work/VASAE
VENV_SITE="$(uv run python -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="${VENV_SITE}/nvidia/cusparselt/lib:${VENV_SITE}/nvidia/cusparse/lib:${LD_LIBRARY_PATH}"

echo "=== Save geometric alignment data for Llama λ=5e-3 ==="

uv run python scripts/analyze/alignment/compute_geometric_alignment.py \
    --model-name meta-llama/Llama-3.1-8B \
    --sae-paths \
        0:/scratch/b5bq/pu22650.b5bq/VASAE_out/001A_F_AblationSoft/001AF_llama_lambda_L0_a5e-3 \
        15:/scratch/b5bq/pu22650.b5bq/VASAE_out/001A_F_AblationSoft/001AF_llama_lambda_L15_a5e-3 \
        31:/scratch/b5bq/pu22650.b5bq/VASAE_out/001A_F_AblationSoft/001AF_llama_lambda_L31_a5e-3 \
    --output-dir exp/F002_AlignmentAnalysis/llama_5e-3 \
    --device cuda
