#!/bin/bash

#SBATCH --job-name=ioi_feat_sweep
#SBATCH --output=exp/IOI/logs/%x_%j_%a.log
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --array=0-11

nvidia-smi --list-gpus
cd ~/work/VASAE

echo "Running on host $(hostname)"
echo "Started on $(date)"
echo "Directory is $(pwd)"
echo "Slurm job ID is ${SLURM_JOBID}"
echo "Slurm array task ID is ${SLURM_ARRAY_TASK_ID}"
echo "This jobs runs on the following machines:"
echo "${SLURM_JOB_NODELIST}"
printf "\n\n"

LAYER="${SLURM_ARRAY_TASK_ID}"

SAE_ROOT="/scratch/b5bq/pu22650.b5bq/VASAE_out/010_soft_align"
OUTPUT_DIR="/scratch/b5bq/pu22650.b5bq/VASAE_out/ioi_feature_sweep"

mkdir -p "$OUTPUT_DIR"

echo "=== Running IOI feature sweep on layer ${LAYER} ==="

uv run python scripts/eval_ioi_feature_sweep.py \
    --layer-idx "$LAYER" \
    --model-name gpt2 \
    --sae-root "$SAE_ROOT" \
    --n-prompts 100 \
    --seed 42 \
    --device cuda \
    --output-dir "$OUTPUT_DIR"

echo "done"
