#!/bin/bash

#SBATCH --job-name=002_anchor
#SBATCH --output=exp/002_p_WeakTokenAnchoringRegularizer/logs/%x_%j_%a.log
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --array=0-7%4

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

# 2 layers x 4 lambda values = 8 tasks
layers=(6 6 6 6 11 11 11 11)
lambdas=(0 1e-4 1e-3 1e-2)

layer="${layers[$SLURM_ARRAY_TASK_ID]}"
lambda_idx=$(( SLURM_ARRAY_TASK_ID % 4 ))
lambda="${lambdas[$lambda_idx]}"

SCRATCH="/scratch/b5bq/pu22650.b5bq/VASAE_out/002_anchor"
output_dir="${SCRATCH}/layer_${layer}_lambda_${lambda}"
model_path="${output_dir}/sae.pth"
analysis_dir="${output_dir}/analysis"

mkdir -p "$output_dir"
mkdir -p "$analysis_dir"

echo "=== Step 1: Train SAE on layer ${layer} with anchor_coeff=${lambda} ==="

uv run python scripts/train_sae_gpt2_hf.py \
    --no-tied-decoder \
    --sparsity-type topk --k 8 \
    --nonneg-latents \
    --anchor-coeff "$lambda" \
    --layer-name "transformer.h.${layer}" \
    --num-epochs 20 --lr 1e-3 \
    --exp-name "002_anchor_layer${layer}_lambda${lambda}" \
    --wandb-group "002_anchor" \
    --sae-save-path "$model_path"

echo "=== Step 2: Analyze feature-vocab alignment ==="

uv run python scripts/analyze_feature_vocab_alignment.py \
    --model-path "$model_path" \
    --output-dir "$analysis_dir" \
    --top-k 10

echo "done"
