#!/bin/bash

#SBATCH --job-name=023_fetch_corpus
#SBATCH --output=exp/023_Dataset/logs/%x_%j.log
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=00:20:00

cd ~/work/VASAE

echo "Running on host $(hostname)"
echo "Started on $(date)"
echo "Directory is $(pwd)"
echo "Slurm job ID is ${SLURM_JOBID}"
echo "This job runs on the following machines:"
echo "${SLURM_JOB_NODELIST}"
printf "\n\n"

mkdir -p exp/023_Dataset/logs
mkdir -p "$VASAE_OUT"

CORPORA=(fineweb dclm pile)

TRAIN_TOKENS="${TRAIN_TOKENS:-66666667}"
HELDOUT_TOKENS="${HELDOUT_TOKENS:-1000000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
TOKENIZER="${TOKENIZER:-gpt2}"
OUT_DIR="${OUT_DIR:-${VASAE_OUT}/Dataset/data}"

echo "=== 023_Dataset fetch corpora: ${CORPORA[*]} ==="
echo "HF_HOME=${HF_HOME}"
echo "VASAE_OUT=${VASAE_OUT}"
echo "OUT_DIR=${OUT_DIR}"
echo "TRAIN_TOKENS=${TRAIN_TOKENS}"
echo "HELDOUT_TOKENS=${HELDOUT_TOKENS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "TOKENIZER=${TOKENIZER}"
printf "\n\n"

uv sync --frozen

declare -A PIDS

for CORPUS in "${CORPORA[@]}"; do
    (
        echo "[$(date)] starting ${CORPUS}"
        if uv run --no-sync python scripts/collect/fetch_corpus.py "$CORPUS" \
            --out-dir "$OUT_DIR" \
            --tokenizer "$TOKENIZER" \
            --train-tokens "$TRAIN_TOKENS" \
            --heldout-tokens "$HELDOUT_TOKENS" \
            --batch-size "$BATCH_SIZE"; then
            echo "[$(date)] finished ${CORPUS}"
        else
            rc=$?
            echo "[$(date)] failed ${CORPUS} with exit code ${rc}"
            exit "$rc"
        fi
    ) &
    PIDS["$CORPUS"]=$!
done

status=0
for CORPUS in "${CORPORA[@]}"; do
    if wait "${PIDS[$CORPUS]}"; then
        echo "${CORPUS} completed successfully."
    else
        echo "${CORPUS} failed."
        status=1
    fi
done

if [ "$status" -ne 0 ]; then
    echo "At least one corpus fetch failed."
    exit "$status"
fi

echo "done"
