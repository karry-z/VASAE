#!/bin/bash

#SBATCH --job-name=023_train_eval
#SBATCH --output=exp/023_Dataset/logs/%x_%j.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

cd ~/work/VASAE

VENV_SITE="$(uv run python -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="${VENV_SITE}/nvidia/cusparselt/lib:${VENV_SITE}/nvidia/cusparse/lib:${LD_LIBRARY_PATH}"

mkdir -p exp/023_Dataset/logs
mkdir -p "$VASAE_OUT"

CORPUS_DIR="${CORPUS_DIR:-${VASAE_OUT}/Dataset/data}"
MODEL_NAME="${MODEL_NAME:-gpt2}"
LAYER_IDX="${LAYER_IDX:-5}"
DTYPE="${DTYPE:-}"
DIM_SPARSE="${DIM_SPARSE:-}"
RUN_NAME="${RUN_NAME:-gpt2_L${LAYER_IDX}_mixture_soft}"
RUN_DIR="${RUN_DIR:-${VASAE_OUT}/Dataset/runs/${RUN_NAME}}"
TRAIN_TOKENS="${TRAIN_TOKENS:-200000000}"
VALID_TOKENS="${VALID_TOKENS:-300000}"
EVAL_TOKENS="${EVAL_TOKENS:-1000000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-128}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"

echo "Running on host $(hostname)"
echo "Started on $(date)"
echo "SLURM_JOBID=${SLURM_JOBID}"
echo "VASAE_OUT=${VASAE_OUT}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "LAYER_IDX=${LAYER_IDX}"
echo "DTYPE=${DTYPE}"
echo "DIM_SPARSE=${DIM_SPARSE:-auto-vocab}"
echo "CORPUS_DIR=${CORPUS_DIR}"
echo "RUN_DIR=${RUN_DIR}"
echo "TRAIN_TOKENS=${TRAIN_TOKENS}"
echo "VALID_TOKENS=${VALID_TOKENS}"
echo "EVAL_TOKENS=${EVAL_TOKENS}"
printf "\n"

uv sync --frozen

if [ -z "$DIM_SPARSE" ]; then
    DIM_SPARSE="$(MODEL_NAME="$MODEL_NAME" uv run --no-sync python -c 'import os; from transformers import AutoConfig; print(AutoConfig.from_pretrained(os.environ["MODEL_NAME"]).vocab_size)')"
    echo "Resolved DIM_SPARSE=${DIM_SPARSE}"
fi

uv run --no-sync python scripts/collect/validate_corpus.py \
    --out-dir "$CORPUS_DIR"

MODEL_ARGS=(
    --model-name "$MODEL_NAME"
    --layer-idx "$LAYER_IDX"
)
if [ -n "$DTYPE" ]; then
    MODEL_ARGS+=(--dtype "$DTYPE")
fi

if [ ! -f "${RUN_DIR}/results.json" ]; then
    uv run --no-sync python scripts/training/train_sae_online.py \
        --data-source jsonl \
        "${MODEL_ARGS[@]}" \
        --corpus-dir "$CORPUS_DIR" \
        --train-tokens "$TRAIN_TOKENS" \
        --valid-tokens "$VALID_TOKENS" \
        --train-batchsize "$BATCH_SIZE" \
        --valid-batchsize "$BATCH_SIZE" \
        --max-length "$MAX_LENGTH" \
        --dim-sparse "$DIM_SPARSE" \
        --sparsity-type topk \
        --k 32 \
        --nonneg-latents \
        --anchor-coeff 1e-4 \
        --anchor-mode hard \
        --num-epochs "$NUM_EPOCHS" \
        --wandb-group "023_Dataset" \
        --save-dir "$(dirname "$RUN_DIR")" \
        --exp-name "$(basename "$RUN_DIR")"
else
    echo "Training results already exist at ${RUN_DIR}/results.json; skipping train."
fi

for CORPUS in fineweb dclm pile; do
    if [ ! -f "${RUN_DIR}/results_eval_${CORPUS}.json" ]; then
        uv run --no-sync python scripts/eval/eval_sae_online.py \
            --data-source jsonl \
            --sae-path "$RUN_DIR" \
            "${MODEL_ARGS[@]}" \
            --corpus "$CORPUS" \
            --eval-tokens "$EVAL_TOKENS" \
            --corpus-dir "$CORPUS_DIR" \
            --test-batchsize "$EVAL_BATCH_SIZE" \
            --max-length "$MAX_LENGTH"
    else
        echo "Eval results already exist for ${CORPUS}; skipping."
    fi
done

uv run --no-sync python scripts/aggregate/summarize_dataset_results.py \
    --run-dir "$RUN_DIR"

echo "done"
