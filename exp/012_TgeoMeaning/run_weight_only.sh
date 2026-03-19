#!/bin/bash

#SBATCH --job-name=012a_tgeo_weight
#SBATCH --output=exp/012_p_TgeoMeaning/logs/%x_%j.log
#SBATCH --gpus=1
#SBATCH --time=1:00:00

cd ~/work/VASAE
VENV_SITE="$(uv run python -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="${VENV_SITE}/nvidia/cusparselt/lib:${VENV_SITE}/nvidia/cusparse/lib:${LD_LIBRARY_PATH}"
nvidia-smi --list-gpus

echo "Running on host $(hostname)"
echo "Started on $(date)"
echo "Directory is $(pwd)"
printf "\n\n"

SCRATCH="/scratch/b5bq/pu22650.b5bq/VASAE_out/010_soft_align"
OUTPUT_DIR="exp/012_p_TgeoMeaning/weight_only"

echo "=== 012a: Weight-only t_geo analysis (all 12 layers) ==="

uv run python scripts/analyze_tgeo_weight_only.py \
    --model-name gpt2 \
    --sae-dir "$SCRATCH" \
    --sae-pattern "010_soft_gpt2_L{layer}_k32_a1e-4" \
    --layers 0-11 \
    --knn-k 10 \
    --n-null 5 \
    --output-dir "$OUTPUT_DIR"

echo "done"
