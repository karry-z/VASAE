#!/bin/bash

#SBATCH --job-name=ioi_casestudy
#SBATCH --output=exp/IOI_casestudy/logs/%x_%j.log
#SBATCH --gpus=1
#SBATCH --time=01:00:00

nvidia-smi --list-gpus
cd ~/work/VASAE

echo "Running on host $(hostname)"
echo "Started on $(date)"
echo "Directory is $(pwd)"
printf "\n\n"

SAE_ROOT="/scratch/b5bq/pu22650.b5bq/VASAE_out/010_soft_align"
OUTPUT_DIR="exp/IOI_casestudy/results"

mkdir -p "$OUTPUT_DIR"

uv run python scripts/plot/casestudy_ioi_features.py \
    --features 7:3733 4:4649 0:5628 6:5628 5:13 3:278 8:3477 1:5386 6:3042 0:783 \
    --sae-root "$SAE_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --n-prompts 100 \
    --device cuda

echo "done"
