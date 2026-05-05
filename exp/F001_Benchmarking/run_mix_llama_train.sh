#!/bin/bash

#SBATCH --job-name=001Fmix_llama
#SBATCH --output=exp/F001_Benchmarking/logs/%x_%j_%a.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-9

set -euo pipefail

cd "${VASAE_HOME:-$HOME/work/VASAE}"

if [ -z "${VASAE_OUT:-}" ]; then
    echo "VASAE_OUT is required; set it to the project output directory."
    exit 1
fi

export HF_HOME="${HF_HOME:-${PROJECTDIR:-/projects/b5bq}/hf_cache}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TQDM_DISABLE=1

LAYERS=(0 8 16 24 31)
VARIANTS=(plain soft)

task_id=${SLURM_ARRAY_TASK_ID:-0}
num_tasks=$(( ${#LAYERS[@]} * ${#VARIANTS[@]} ))

if [ "$task_id" -lt 0 ] || [ "$task_id" -ge "$num_tasks" ]; then
    echo "Invalid task_id=${task_id}; expected 0-$(( num_tasks - 1 ))."
    exit 1
fi

layer_idx=$(( task_id / ${#VARIANTS[@]} ))
variant_idx=$(( task_id % ${#VARIANTS[@]} ))
layer=${LAYERS[$layer_idx]}
variant=${VARIANTS[$variant_idx]}

OUT_DIR="${OUT_DIR:-${VASAE_OUT}/F001_Benchmarking_mix}"
CORPUS_DIR="${CORPUS_DIR:-${VASAE_OUT}/Dataset/data}"
TRAIN_TOKENS="${TRAIN_TOKENS:-20000000}"
VALID_TOKENS="${VALID_TOKENS:-100000}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
PATIENCE="${PATIENCE:-3}"
MAX_LENGTH="${MAX_LENGTH:-128}"
BATCH_SIZE="${BATCH_SIZE:-8}"
K="${K:-32}"
LR="${LR:-1e-3}"
ANCHOR_COEFF="${ANCHOR_COEFF:-5e-3}"
ANCHOR_EVERY="${ANCHOR_EVERY:-50}"
DIM_SPARSE=128256

exp_name="001Fmix_llama_L${layer}_${variant}"
RUN_DIR="${OUT_DIR}/${exp_name}"

COMMON_ARGS=(
    --model-name meta-llama/Llama-3.1-8B
    --dtype bfloat16
    --layer-idx "$layer"
    --data-source jsonl
    --corpus-dir "$CORPUS_DIR"
    --corpora fineweb dclm pile
    --train-tokens "$TRAIN_TOKENS"
    --valid-tokens "$VALID_TOKENS"
    --max-length "$MAX_LENGTH"
    --train-batchsize "$BATCH_SIZE"
    --valid-batchsize "$BATCH_SIZE"
    --sparsity-type topk
    --k "$K"
    --nonneg-latents
    --num-epochs "$NUM_EPOCHS"
    --patience "$PATIENCE"
    --lr "$LR"
    --wandb-group "001Fmix_llama"
    --save-dir "$OUT_DIR"
)

if [ "$variant" = "plain" ]; then
    VARIANT_ARGS=(--dim-sparse "$DIM_SPARSE")
else
    VARIANT_ARGS=(
        --dim-sparse "$DIM_SPARSE"
        --anchor-coeff "$ANCHOR_COEFF"
        --anchor-mode hard
        --anchor-every "$ANCHOR_EVERY"
    )
fi

TRAIN_CMD=(
    uv run --no-sync python scripts/training/train_sae_online.py
    "${COMMON_ARGS[@]}"
    "${VARIANT_ARGS[@]}"
    --exp-name "$exp_name"
)

echo "Running on host $(hostname)"
echo "Started on $(date)"
echo "Directory is $(pwd)"
echo "SLURM_JOBID=${SLURM_JOBID:-local}"
echo "SLURM_ARRAY_TASK_ID=${task_id}"
echo "Layer=${layer}"
echo "Variant=${variant}"
echo "OUT_DIR=${OUT_DIR}"
echo "RUN_DIR=${RUN_DIR}"
echo "CORPUS_DIR=${CORPUS_DIR}"
echo "TRAIN_TOKENS=${TRAIN_TOKENS}"
echo "VALID_TOKENS=${VALID_TOKENS}"
echo "NUM_EPOCHS=${NUM_EPOCHS}"
echo "PATIENCE=${PATIENCE}"
printf "\n"

if [ "${DRY_RUN:-0}" = "1" ]; then
    printf "DRY_RUN command:"
    printf " %q" "${TRAIN_CMD[@]}"
    printf "\n"
    exit 0
fi

mkdir -p exp/F001_Benchmarking/logs "$OUT_DIR"

if [ -f "${RUN_DIR}/results.json" ]; then
    echo "Training results already exist for ${exp_name}; skipping."
    exit 0
fi

if [ -e "${RUN_DIR}/model.safetensors" ] || [ -e "${RUN_DIR}/config.json" ]; then
    if [ "${FORCE_RETRAIN:-0}" != "1" ]; then
        echo "Found an incomplete/intermediate run at ${RUN_DIR}."
        echo "It has checkpoint files but no results.json; set FORCE_RETRAIN=1 to overwrite."
        exit 2
    fi
    echo "FORCE_RETRAIN=1 set; training will overwrite checkpoint files in ${RUN_DIR}."
fi

if [ "${RUN_UV_SYNC:-0}" = "1" ]; then
    uv sync --frozen
fi

VENV_SITE="$(uv run --no-sync python -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="${VENV_SITE}/nvidia/cusparselt/lib:${VENV_SITE}/nvidia/cusparse/lib:${LD_LIBRARY_PATH:-}"
nvidia-smi --list-gpus

"${TRAIN_CMD[@]}"

echo "done"
