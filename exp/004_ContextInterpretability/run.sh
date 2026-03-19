#!/bin/bash

#SBATCH --job-name=004_context
#SBATCH --output=exp/004_p_ContextInterpretability/logs/%x_%j_%a.log
#SBATCH --gpus=1
#SBATCH --time=02:00:00
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

# 2 layers x 4 lambda values = 8 tasks (same as 002)
layers=(6 6 6 6 11 11 11 11)
lambdas=(0 1e-4 1e-3 1e-2)

layer="${layers[$SLURM_ARRAY_TASK_ID]}"
lambda_idx=$(( SLURM_ARRAY_TASK_ID % 4 ))
lambda="${lambdas[$lambda_idx]}"

SCRATCH="/scratch/b5bq/pu22650.b5bq/VASAE_out/002_anchor"
model_dir="${SCRATCH}/layer_${layer}_lambda_${lambda}"
model_path="${model_dir}/sae.pth"
analysis_dir="${model_dir}/analysis"
output_dir="${model_dir}/context_analysis"

mkdir -p "$output_dir"

echo "=== Analyzing context interpretability for layer ${layer}, lambda=${lambda} ==="

uv run python scripts/analyze_context_interpretability.py \
    --model-path "$model_path" \
    --alignment-path "${analysis_dir}/alignment_results.json" \
    --layer-name "transformer.h.${layer}" \
    --output-dir "$output_dir" \
    --num-top-features 200 \
    --top-contexts 20 \
    --context-window 10

echo "done"
