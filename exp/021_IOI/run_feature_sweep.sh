#!/bin/bash

#SBATCH --job-name=021_ioi_sweep
#SBATCH --output=exp/021_IOI/logs/%x_%j_%a.log
#SBATCH --gpus=1
#SBATCH --time=04:00:00
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

SAE_ROOT="/scratch/b5bq/pu22650.b5bq/VASAE_out/001_F_Benchmarking"
OUTPUT_DIR="/scratch/b5bq/pu22650.b5bq/VASAE_out/021_ioi_feature_sweep"

mkdir -p "$OUTPUT_DIR"

echo "=== 021_IOI feature sweep: layer ${LAYER} ==="

uv run python scripts/eval/eval_ioi_feature_sweep.py \
    --layer-idx "$LAYER" \
    --model-name gpt2 \
    --sae-root "$SAE_ROOT" \
    --sae-pattern '001F_gpt2_L{layer}_soft' \
    --n-prompts 100 \
    --seed 42 \
    --device cuda \
    --output-dir "$OUTPUT_DIR"

echo "done"
