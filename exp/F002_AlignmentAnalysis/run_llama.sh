#!/bin/bash

#SBATCH --job-name=F002_align_llama
#SBATCH --output=exp/F002_AlignmentAnalysis/logs/%x_%j_%a.log
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --array=0-2

cd ~/work/VASAE
VENV_SITE="$(uv run python -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="${VENV_SITE}/nvidia/cusparselt/lib:${VENV_SITE}/nvidia/cusparse/lib:${LD_LIBRARY_PATH}"
nvidia-smi --list-gpus

echo "Running on host $(hostname)"
echo "Started on $(date)"
echo "Directory is $(pwd)"
echo "Slurm job ID is ${SLURM_JOBID}"
echo "Slurm array task ID is ${SLURM_ARRAY_TASK_ID}"
echo "This jobs runs on the following machines:"
echo "${SLURM_JOB_NODELIST}"
printf "\n\n"

# Map array task ID to actual layer indices
LAYERS=(0 15 31)
layer=${LAYERS[${SLURM_ARRAY_TASK_ID}]}

CKPT_BASE=/scratch/b5bq/pu22650.b5bq/VASAE_out/001A_F_AblationSoft

echo "=== F002 Alignment Quality: Llama-3.1-8B (lambda=5e-3) layer=${layer} ==="

mkdir -p exp/F002_AlignmentAnalysis/logs

uv run python scripts/analyze/alignment/analyze_alignment_quality.py \
    --model-name meta-llama/Llama-3.1-8B \
    --sae-paths "${layer}:${CKPT_BASE}/001AF_llama_lambda_L${layer}_a5e-3" \
    --layer-idx "$layer" \
    --n-samples 5000 \
    --max-length 256 \
    --batch-size 16 \
    --top-m 50 \
    --output-dir exp/F002_AlignmentAnalysis/llama_5e-3 \
    --device cuda

echo "done"
