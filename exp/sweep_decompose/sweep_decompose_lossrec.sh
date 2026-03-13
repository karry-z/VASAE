#!/bin/bash

#SBATCH --job-name=decompose_lossrec
#SBATCH --output=exp/sweep_decompose/logs/%x_%j_%a.log
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=0-71%12

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

d_pca_values=(16 32 64 128 256 512)
layers=(0 1 2 3 4 5 6 7 8 9 10 11)

num_dpca=${#d_pca_values[@]}
layer_idx=$((SLURM_ARRAY_TASK_ID / num_dpca))
dpca_idx=$((SLURM_ARRAY_TASK_ID % num_dpca))

layer="${layers[$layer_idx]}"
d_pca="${d_pca_values[$dpca_idx]}"

SCRATCH="/scratch/b5bq/pu22650.b5bq/VASAE_out/sweep_decompose"
output_dir="${SCRATCH}/layer_${layer}/dpca_${d_pca}"

echo "Running: layer=$layer, d_pca=$d_pca"

uv run python scripts/eval_loss_recovered.py \
    --layer-idx "$layer" \
    --model-path "${output_dir}/model.pth" \
    --pca-path "${SCRATCH}/layer_${layer}/pca_components.pt" \
    --d-pca "$d_pca" \
    --output-path "${output_dir}/loss_recovered.json"

echo "done"
