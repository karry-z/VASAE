#!/bin/bash

#SBATCH --job-name=005_random
#SBATCH --output=exp/005_p_RandomDictControl/logs/%x_%j_%a.log
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

# 2 layers x 2 lambdas x 2 random types = 8 tasks
layers=(6 6 6 6 11 11 11 11)
lambdas=(1e-4 1e-4 1e-3 1e-3 1e-4 1e-4 1e-3 1e-3)
random_types=(shuffle gaussian shuffle gaussian shuffle gaussian shuffle gaussian)

layer="${layers[$SLURM_ARRAY_TASK_ID]}"
lambda="${lambdas[$SLURM_ARRAY_TASK_ID]}"
random_type="${random_types[$SLURM_ARRAY_TASK_ID]}"

SCRATCH="/scratch/b5bq/pu22650.b5bq/VASAE_out/005_random"
output_dir="${SCRATCH}/layer_${layer}_lambda_${lambda}_${random_type}"
model_path="${output_dir}/sae.pth"
analysis_rand_dir="${output_dir}/analysis_vs_random"
analysis_real_dir="${output_dir}/analysis_vs_real"

mkdir -p "$output_dir"
mkdir -p "$analysis_rand_dir"
mkdir -p "$analysis_real_dir"

echo "=== Step 1: Train SAE on layer ${layer} with anchor_coeff=${lambda}, random_anchor=${random_type} ==="

uv run python scripts/train_sae_gpt2_hf.py \
    --no-tied-decoder \
    --sparsity-type topk --k 8 \
    --nonneg-latents \
    --anchor-coeff "$lambda" \
    --random-anchor "$random_type" \
    --layer-name "transformer.h.${layer}" \
    --num-epochs 20 --lr 1e-3 \
    --exp-name "005_random_layer${layer}_lambda${lambda}_${random_type}" \
    --wandb-group "005_random" \
    --sae-save-path "$model_path"

echo "=== Step 2: Analyze alignment vs random dictionary ==="

uv run python scripts/analyze_feature_vocab_alignment.py \
    --model-path "$model_path" \
    --output-dir "$analysis_rand_dir" \
    --embedding-override "${output_dir}/random_emb.pt" \
    --top-k 10

echo "=== Step 3: Analyze alignment vs real W_E ==="

uv run python scripts/analyze_feature_vocab_alignment.py \
    --model-path "$model_path" \
    --output-dir "$analysis_real_dir" \
    --top-k 10

echo "done"
