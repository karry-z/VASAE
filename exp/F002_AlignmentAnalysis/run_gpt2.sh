#!/bin/bash

#SBATCH --job-name=F002_align_gpt2
#SBATCH --output=exp/F002_AlignmentAnalysis/logs/%x_%j_%a.log
#SBATCH --gpus=1
#SBATCH --time=2:00:00
#SBATCH --array=0-11

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

echo "=== F002 Alignment Quality: GPT-2 layer=${layer} ==="

mkdir -p exp/F002_AlignmentAnalysis/logs

uv run python scripts/analyze/alignment/analyze_alignment_quality.py \
    --results-dir /scratch/b5bq/pu22650.b5bq/VASAE_out/001_F_Benchmarking \
    --model-name gpt2 \
    --variant soft \
    --baseline-variant plain \
    --layer-idx "$layer" \
    --n-samples 5000 \
    --max-length 256 \
    --batch-size 32 \
    --top-m 50 \
    --output-dir exp/F002_AlignmentAnalysis/gpt2 \
    --device cuda

echo "done"
