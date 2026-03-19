#!/bin/bash

#SBATCH --job-name=011_io_decomp
#SBATCH --output=exp/011_p_IODecomposition/logs/%x_%j_%a.log
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
printf "\n\n"

# Analyze 3 representative layers from the best 010 config (k=32, a=1e-4)
layers=(2 6 11)
layer="${layers[$SLURM_ARRAY_TASK_ID]}"

SCRATCH="/scratch/b5bq/pu22650.b5bq/VASAE_out/010_soft_align"
SAE_PATH="${SCRATCH}/010_soft_gpt2_L${layer}_k32_a1e-4"
OUTPUT_DIR="exp/011_p_IODecomposition/L${layer}_k32"

echo "=== I/O Decomposition: layer=${layer} ==="

uv run python scripts/analyze_feature_io_decomposition.py \
    --sae-path "$SAE_PATH" \
    --model-name gpt2 \
    --layer-idx "$layer" \
    --dataset wikitext \
    --dataset-config wikitext-103-raw-v1 \
    --n-samples 500 \
    --n-causal-samples 100 \
    --batch-size 16 \
    --top-k 5 \
    --output-dir "$OUTPUT_DIR"

echo "done"
