#!/bin/bash

#SBATCH --job-name=001AF_freq_llama
#SBATCH --output=exp/001A_F_AblationSoft/logs/%x_%j_%a.log
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --array=0-14

cd ~/work/VASAE
export HF_HOME=/scratch/b5bq/pu22650.b5bq/hf_cache
VENV_SITE="$(uv run python -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="${VENV_SITE}/nvidia/cusparselt/lib:${VENV_SITE}/nvidia/cusparse/lib:${LD_LIBRARY_PATH}"

echo "Running on host $(hostname)"
echo "Started on $(date)"
echo "Slurm job ID is ${SLURM_JOBID}, array task ID is ${SLURM_ARRAY_TASK_ID}"
printf "\n"

# 3 layers x 5 anchor_every = 15 tasks
task_id=${SLURM_ARRAY_TASK_ID}
LAYERS=(0 15 31)
FREQS=(1 10 50 100 500)

layer_idx=$(( task_id / 5 ))
freq_idx=$(( task_id % 5 ))
layer=${LAYERS[$layer_idx]}
freq=${FREQS[$freq_idx]}

SCRATCH="/scratch/b5bq/pu22650.b5bq/VASAE_out/001A_F_AblationSoft"
mkdir -p "$SCRATCH"

exp_name="001AF_llama_freq_L${layer}_every${freq}"
echo "=== layer=${layer}, anchor_every=${freq} ==="

# Reuse 001_F results for anchor_every=50 (same config as 001_F soft)
BENCH="/scratch/b5bq/pu22650.b5bq/VASAE_out/001_F_Benchmarking"
if [ "$freq" -eq 50 ]; then
    src="${BENCH}/001F_llama_L${layer}_soft"
    if [ -f "${src}/results.json" ] && [ ! -e "${SCRATCH}/${exp_name}" ]; then
        ln -s "$src" "${SCRATCH}/${exp_name}"
        echo "Symlinked from 001_F: ${src} -> ${SCRATCH}/${exp_name}"
    fi
fi

# Skip if results already exist
if [ -f "${SCRATCH}/${exp_name}/results.json" ]; then
    echo "Results already exist for ${exp_name}, skipping."
    exit 0
fi

uv run python scripts/train_sae_online.py \
    --model-name meta-llama/Llama-3.1-8B \
    --dtype bfloat16 \
    --layer-idx "$layer" \
    --dataset wikitext \
    --dataset-config wikitext-103-raw-v1 \
    --max-length 128 \
    --train-batchsize 8 \
    --eval-batchsize 8 \
    --train-samples 20000 \
    --eval-samples 2000 \
    --test-samples 5000 \
    --dim-sparse 128256 \
    --sparsity-type topk \
    --k 32 \
    --nonneg-latents \
    --anchor-coeff 1e-4 \
    --anchor-mode hard \
    --anchor-every "$freq" \
    --num-epochs 20 \
    --patience 3 \
    --lr 1e-3 \
    --wandb-group "001AF_freq_llama" \
    --save-dir "$SCRATCH" \
    --exp-name "$exp_name"

echo "done"
