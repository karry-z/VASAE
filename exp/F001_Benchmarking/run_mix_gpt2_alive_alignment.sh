#!/bin/bash

#SBATCH --job-name=001Fmix_gpt2_alive
#SBATCH --output=exp/F001_Benchmarking/logs/%x_%j_%a.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-14

set -euo pipefail

cd "${VASAE_HOME:-$HOME/work/VASAE}"

export HF_HOME="${HF_HOME:-${PROJECTDIR:-/projects/b5bq}/hf_cache}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TQDM_DISABLE=1

LAYERS=(0 3 6 9 11)
CORPORA=(fineweb dclm pile)

RESULTS_DIR="${RESULTS_DIR:-/projects/b5bq/VASAE/F001_Benchmarking_mix}"
CORPUS_DIR="${CORPUS_DIR:-/projects/b5bq/VASAE/Dataset/data}"
OUTPUT_DIR="${OUTPUT_DIR:-exp/F001_Benchmarking/alive_alignment/gpt2_mix}"
EVAL_TOKENS="${EVAL_TOKENS:-1000000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LENGTH="${MAX_LENGTH:-128}"
ALIGNMENT_THRESHOLD="${ALIGNMENT_THRESHOLD:-0.8}"

mkdir -p exp/F001_Benchmarking/logs "$OUTPUT_DIR"

if [ "${AGGREGATE_ONLY:-0}" = "1" ]; then
    if [ "${SLURM_ARRAY_TASK_ID:-0}" != "0" ]; then
        echo "AGGREGATE_ONLY=1; only array task 0 aggregates."
        exit 0
    fi
    uv run --no-sync python scripts/analyze/alignment/eval_alive_alignment.py \
        --output-dir "$OUTPUT_DIR" \
        --aggregate-only
    exit 0
fi

task_id=${SLURM_ARRAY_TASK_ID:-0}
num_tasks=$(( ${#LAYERS[@]} * ${#CORPORA[@]} ))

if [ "$task_id" -lt 0 ] || [ "$task_id" -ge "$num_tasks" ]; then
    echo "Invalid task_id=${task_id}; expected 0-$(( num_tasks - 1 ))."
    exit 1
fi

layer_idx=$(( task_id / ${#CORPORA[@]} ))
corpus_idx=$(( task_id % ${#CORPORA[@]} ))
layer=${LAYERS[$layer_idx]}
corpus=${CORPORA[$corpus_idx]}

echo "Running on host $(hostname)"
echo "Started on $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Directory is $(pwd)"
echo "SLURM_JOBID=${SLURM_JOBID:-local}"
echo "SLURM_ARRAY_TASK_ID=${task_id}"
echo "Layer=${layer}"
echo "Corpus=${corpus}"
echo "RESULTS_DIR=${RESULTS_DIR}"
echo "CORPUS_DIR=${CORPUS_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "EVAL_TOKENS=${EVAL_TOKENS}"
printf "\n"

uv sync --frozen

VENV_SITE="$(uv run --no-sync python -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="${VENV_SITE}/nvidia/cusparselt/lib:${VENV_SITE}/nvidia/cusparse/lib:${LD_LIBRARY_PATH:-}"
nvidia-smi --list-gpus

FORCE_ARGS=()
if [ "${FORCE:-0}" = "1" ]; then
    FORCE_ARGS=(--force)
fi

uv run --no-sync python scripts/analyze/alignment/eval_alive_alignment.py \
    --results-dir "$RESULTS_DIR" \
    --model-name gpt2 \
    --layer-idx "$layer" \
    --corpus "$corpus" \
    --corpus-dir "$CORPUS_DIR" \
    --eval-tokens "$EVAL_TOKENS" \
    --alignment-threshold "$ALIGNMENT_THRESHOLD" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --max-length "$MAX_LENGTH" \
    --no-aggregate \
    "${FORCE_ARGS[@]}"

echo "done"
