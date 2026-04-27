#!/bin/bash

#SBATCH --job-name=001F_bench_gpt2
#SBATCH --output=exp/001_F_Benchmarking/logs/%x_%j_%a.log
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --array=0-35%12

echo "Started on $(date)"
echo "Running on host $(hostname)"
cd $VASAE_HOME
echo "running on working dir: $(pwd)"
mkdir -p $VASAE_OUT/F001_Benchmarking
echo "output stores at: $VASAE_OUT/F001_Benchmarking"
echo "Slurm job ID is ${SLURM_JOBID}"
echo "Slurm array task ID is ${SLURM_ARRAY_TASK_ID}"
echo "This jobs runs on the following machines:"
echo "${SLURM_JOB_NODELIST}"
printf "\n\n"

# 12 layers x 3 variants (plain, hard, soft) = 36 tasks
# task_id = layer * 3 + variant
task_id=${SLURM_ARRAY_TASK_ID}
layer=$(( task_id / 3 ))
variant=$(( task_id % 3 ))

# Common arguments
COMMON_ARGS=(
    --model-name gpt2
    --layer-idx "$layer"
    --dataset wikitext
    --dataset-config wikitext-103-raw-v1
    --max-length 128
    --train-batchsize 32
    --valid-batchsize 32
    --train-samples 50000
    --eval-samples 10000
    --test-samples 5000
    --sparsity-type topk
    --k 32
    --nonneg-latents
    --num-epochs 20
    --patience 3
    --lr 1e-3
    --wandb-group "001F_bench_gpt2"
    --save-dir $VASAE_OUT/F001_Benchmarking
)

if [ "$variant" -eq 0 ]; then
    # Plain SAE: untied decoder, no anchor, dim_sparse = vocab_size (50257)
    variant_name="plain"
    VARIANT_ARGS=(--dim-sparse 50257)
elif [ "$variant" -eq 1 ]; then
    # VASAE-Hard: tied decoder, dim_sparse = vocab_size (auto)
    variant_name="hard"
    VARIANT_ARGS=(--tied-decoder --freeze-decoder)
elif [ "$variant" -eq 2 ]; then
    # VASAE-Soft: untied decoder, dim_sparse = vocab_size, anchor regularizer
    variant_name="soft"
    VARIANT_ARGS=(--dim-sparse 50257 --anchor-coeff 1e-4 --anchor-mode hard)
fi

exp_name="F001_gpt2_L${layer}_${variant_name}"
RUN_DIR="$VASAE_OUT/F001_Benchmarking/${exp_name}"
echo "=== Train: layer=${layer}, variant=${variant_name} ==="

# Skip if both train and eval outputs already exist
if [ -f "${RUN_DIR}/results.json" ] && [ -f "${RUN_DIR}/results_eval.json" ]; then
    echo "Train and eval results already exist for ${exp_name}, skipping."
    exit 0
fi

if [ ! -f "${RUN_DIR}/results.json" ]; then
    uv run python scripts/training/train_sae_online.py \
        "${COMMON_ARGS[@]}" \
        "${VARIANT_ARGS[@]}" \
        --exp-name "$exp_name"
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
