#!/bin/bash

#SBATCH --job-name=006_freq
#SBATCH --output=exp/006_p_TokenFreqAnalysis/logs/%x_%j_%a.log
#SBATCH --gpus=1
#SBATCH --time=01:00:00

nvidia-smi --list-gpus
cd ~/work/VASAE

echo "Running on host $(hostname)"
echo "Started on $(date)"
echo "Directory is $(pwd)"
echo "Slurm job ID is ${SLURM_JOBID}"
echo "This jobs runs on the following machines:"
echo "${SLURM_JOB_NODELIST}"
printf "\n\n"

SCRATCH="/scratch/b5bq/pu22650.b5bq/VASAE_out"
OUTPUT_DIR="${SCRATCH}/006_freq"
mkdir -p "$OUTPUT_DIR"

# Compare 001 baseline (layer 6 & 11) with 002 anchor configs
# Adjust paths based on actual 001/002 output locations

# Layer 6 comparison
echo "=== Layer 6 analysis ==="
uv run python scripts/analyze_token_frequency.py \
    --output-dir "${OUTPUT_DIR}/layer_6" \
    --alignment-dirs \
        "${SCRATCH}/002_anchor/layer_6_lambda_0/analysis:plain" \
        "${SCRATCH}/002_anchor/layer_6_lambda_1e-4/analysis:anchor_1e-4" \
        "${SCRATCH}/002_anchor/layer_6_lambda_1e-3/analysis:anchor_1e-3" \
        "${SCRATCH}/002_anchor/layer_6_lambda_1e-2/analysis:anchor_1e-2"

# Layer 11 comparison
echo "=== Layer 11 analysis ==="
uv run python scripts/analyze_token_frequency.py \
    --output-dir "${OUTPUT_DIR}/layer_11" \
    --alignment-dirs \
        "${SCRATCH}/002_anchor/layer_11_lambda_0/analysis:plain" \
        "${SCRATCH}/002_anchor/layer_11_lambda_1e-4/analysis:anchor_1e-4" \
        "${SCRATCH}/002_anchor/layer_11_lambda_1e-3/analysis:anchor_1e-3" \
        "${SCRATCH}/002_anchor/layer_11_lambda_1e-2/analysis:anchor_1e-2"

echo "done"
