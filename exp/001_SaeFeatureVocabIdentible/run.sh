#!/bin/bash

#SBATCH --job-name=001_vocab_align
#SBATCH --output=exp/001_p_SaeFeatureVocabIdentible/logs/%x_%j_%a.log
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --array=0-11%6

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

layers=(0 1 2 3 4 5 6 7 8 9 10 11)
layer="${layers[$SLURM_ARRAY_TASK_ID]}"

SCRATCH="/scratch/b5bq/pu22650.b5bq/VASAE_out/001_plain_sae"
output_dir="${SCRATCH}/layer_${layer}"
model_path="${output_dir}/sae.pth"
analysis_dir="${output_dir}/analysis"

mkdir -p "$output_dir"
mkdir -p "$analysis_dir"

echo "=== Step 1: Train plain SAE on layer ${layer} ==="

uv run python scripts/train_sae_gpt2_hf.py \
    --no-tied-decoder \
    --sparsity-type topk --k 8 \
    --nonneg-latents \
    --layer-name "transformer.h.${layer}" \
    --num-epochs 20 --lr 1e-3 \
    --exp-name "001_plain_sae_layer${layer}" \
    --wandb-group "001_vocab_align" \
    --sae-save-path "$model_path"

echo "=== Step 2: Analyze feature-vocab alignment ==="

uv run python scripts/analyze_feature_vocab_alignment.py \
    --model-path "$model_path" \
    --output-dir "$analysis_dir" \
    --top-k 10

echo "done"
