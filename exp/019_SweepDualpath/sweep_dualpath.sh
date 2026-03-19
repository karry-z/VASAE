#!/bin/bash

#SBATCH --job-name=sweep_dualpath
#SBATCH --output=exp/sweep_dualpath/logs/%x_%j_%a.log
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --array=0-35%12

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

d_pca_values=(2 4 8)
layers=(0 1 2 3 4 5 6 7 8 9 10 11)

num_dpca=${#d_pca_values[@]}
layer_idx=$((SLURM_ARRAY_TASK_ID / num_dpca))
dpca_idx=$((SLURM_ARRAY_TASK_ID % num_dpca))

layer="${layers[$layer_idx]}"
d_pca="${d_pca_values[$dpca_idx]}"

SCRATCH="/scratch/b5bq/pu22650.b5bq/VASAE_out/sweep_dualpath"

shared_layer_dir="${SCRATCH}/layer_${layer}"
output_dir="${SCRATCH}/layer_${layer}/dpca_${d_pca}"

mkdir -p "$output_dir"
mkdir -p "$shared_layer_dir"

echo "Running: layer=$layer, d_pca=$d_pca"
echo "Output: $output_dir"

uv run python scripts/train_dualpath_sae.py \
    --exp-name "dualpath_layer${layer}_dpca${d_pca}" \
    --layer-name "transformer.h.${layer}" \
    --d-pca "$d_pca" \
    --lambda-z 1e-3 \
    --lambda-y 1e-4 \
    --output-dir "$output_dir" \
    --shared-layer-dir "$shared_layer_dir" \
    --wandb-group "sweep_dualpath"

echo "done"
