#!/bin/bash

#SBATCH --job-name=001AF_abl_gpt2
#SBATCH --output=exp/001A_F_AblationSoft/logs/%x_%j_%a.log
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --array=0-35

cd ~/work/VASAE
export HF_HOME=/scratch/b5bq/pu22650.b5bq/hf_cache
VENV_SITE="$(uv run python -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="${VENV_SITE}/nvidia/cusparselt/lib:${VENV_SITE}/nvidia/cusparse/lib:${LD_LIBRARY_PATH}"

echo "Running on host $(hostname)"
echo "Started on $(date)"
echo "Slurm job ID is ${SLURM_JOBID}, array task ID is ${SLURM_ARRAY_TASK_ID}"
printf "\n"

SCRATCH="/scratch/b5bq/pu22650.b5bq/VASAE_out/001A_F_AblationSoft"
mkdir -p "$SCRATCH"

LAYERS=(0 5 11)

# Task layout (36 tasks total):
#   0-17:  Exp 1a — lambda sweep (6 lambda x 3 layers)
#   18-23: Exp 1b — mode comparison (2 modes x 3 layers)
#   24-35: Exp 2  — k sweep (4 k values x 3 layers)
task_id=${SLURM_ARRAY_TASK_ID}

if [ "$task_id" -lt 18 ]; then
    # Exp 1a: lambda sweep, hard mode, k=32
    LAMBDAS=(0 1e-5 1e-4 5e-4 1e-3 5e-3)
    layer_idx=$(( task_id / 6 ))
    lambda_idx=$(( task_id % 6 ))
    layer=${LAYERS[$layer_idx]}
    lambda=${LAMBDAS[$lambda_idx]}

    exp_name="001AF_gpt2_lambda_L${layer}_a${lambda}"
    echo "=== Exp1a: layer=${layer}, lambda=${lambda} ==="

    VARIANT_ARGS=(
        --dim-sparse 50257
        --k 32
        --anchor-coeff "$lambda"
        --anchor-mode hard
    )

elif [ "$task_id" -lt 24 ]; then
    # Exp 1b: mode comparison at lambda=1e-4, k=32
    MODES=(logsumexp softmax)
    idx=$(( task_id - 18 ))
    layer_idx=$(( idx / 2 ))
    mode_idx=$(( idx % 2 ))
    layer=${LAYERS[$layer_idx]}
    mode=${MODES[$mode_idx]}

    exp_name="001AF_gpt2_mode_L${layer}_${mode}"
    echo "=== Exp1b: layer=${layer}, mode=${mode} ==="

    VARIANT_ARGS=(
        --dim-sparse 50257
        --k 32
        --anchor-coeff 1e-4
        --anchor-mode "$mode"
        --anchor-topk 10
    )

else
    # Exp 2: k sweep at lambda=1e-4, hard mode
    KS=(8 16 64 128)
    idx=$(( task_id - 24 ))
    layer_idx=$(( idx / 4 ))
    k_idx=$(( idx % 4 ))
    layer=${LAYERS[$layer_idx]}
    k=${KS[$k_idx]}

    exp_name="001AF_gpt2_k_L${layer}_k${k}"
    echo "=== Exp2: layer=${layer}, k=${k} ==="

    VARIANT_ARGS=(
        --dim-sparse 50257
        --k "$k"
        --anchor-coeff 1e-4
        --anchor-mode hard
    )
fi

# Reuse 001_F results for overlapping configs:
#   Exp 1a lambda=1e-4 (hard, k=32) = 001_F soft
BENCH="/scratch/b5bq/pu22650.b5bq/VASAE_out/001_F_Benchmarking"
if [ "$task_id" -lt 18 ] && [ "$lambda" = "1e-4" ]; then
    src="${BENCH}/001F_gpt2_L${layer}_soft"
    if [ -f "${src}/results.json" ] && [ ! -e "${SCRATCH}/${exp_name}" ]; then
        ln -s "$src" "${SCRATCH}/${exp_name}"
        echo "Symlinked from 001_F: ${src} -> ${SCRATCH}/${exp_name}"
    fi
fi

RUN_DIR="${SCRATCH}/${exp_name}"

# Skip if both train and eval outputs already exist
if [ -f "${RUN_DIR}/results.json" ] && [ -f "${RUN_DIR}/results_eval.json" ]; then
    echo "Train and eval results already exist for ${exp_name}, skipping."
    exit 0
fi

if [ ! -f "${RUN_DIR}/results.json" ]; then
    uv run python scripts/training/train_sae_online.py \
        --model-name gpt2 \
        --layer-idx "$layer" \
        --dataset wikitext \
        --dataset-config wikitext-103-raw-v1 \
        --max-length 128 \
        --train-batchsize 32 \
        --valid-batchsize 32 \
        --train-samples 50000 \
        --eval-samples 10000 \
        --test-samples 5000 \
        --sparsity-type topk \
        --nonneg-latents \
        --num-epochs 20 \
        --patience 3 \
        --lr 1e-3 \
        --wandb-group "001AF_abl_gpt2" \
        --save-dir "$SCRATCH" \
        --exp-name "$exp_name" \
        "${VARIANT_ARGS[@]}"
fi

if [ ! -f "${RUN_DIR}/results_eval.json" ]; then
    uv run python scripts/eval/eval_sae_online.py \
        --sae-path "$RUN_DIR" \
        --model-name gpt2 \
        --layer-idx "$layer" \
        --test-batchsize 32 \
        --max-length 128 \
        --dataset wikitext \
        --dataset-config wikitext-103-raw-v1 \
        --device cuda
fi

echo "done"
