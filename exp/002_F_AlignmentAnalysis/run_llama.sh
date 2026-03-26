#!/bin/bash

#SBATCH --job-name=002F_align_llama
#SBATCH --output=exp/002_F_AlignmentAnalysis/logs/%x_%j_%a.log
#SBATCH --gpus=1
#SBATCH --time=8:00:00
#SBATCH --array=0-31

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

layer=${SLURM_ARRAY_TASK_ID}

echo "=== 002_F Alignment Quality: Llama-3.1-8B layer=${layer} ==="

mkdir -p exp/002_F_AlignmentAnalysis/logs

uv run python scripts/analyze_alignment_quality.py \
    --results-dir /scratch/b5bq/pu22650.b5bq/VASAE_out/001_F_Benchmarking \
    --model-name meta-llama/Llama-3.1-8B \
    --variant soft \
    --baseline-variant plain \
    --layer-idx "$layer" \
    --n-samples 5000 \
    --n-causal-samples 500 \
    --n-causal-features 600 \
    --max-length 256 \
    --batch-size 16 \
    --top-k-positions 100 \
    --ctx-window 32 \
    --output-dir exp/002_F_AlignmentAnalysis/llama \
    --device cuda

echo "done"
