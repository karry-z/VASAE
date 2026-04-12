#!/bin/bash

#SBATCH --job-name=001F_bench_gpt2
#SBATCH --output=exp/001_F_Benchmarking/logs/%x_%j_%a.log
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --array=0-35%12

cd ~/work/VASAE
export HF_HOME=/scratch/b5bq/pu22650.b5bq/hf_cache
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

# 12 layers x 3 variants (plain, hard, soft) = 36 tasks
# task_id = layer * 3 + variant
task_id=${SLURM_ARRAY_TASK_ID}
layer=$(( task_id / 3 ))
variant=$(( task_id % 3 ))

SCRATCH="/scratch/b5bq/pu22650.b5bq/VASAE_out/001_F_Benchmarking"
mkdir -p "$SCRATCH"

# Common arguments
COMMON_ARGS=(
    --model-name gpt2
    --layer-idx "$layer"
    --dataset wikitext
    --dataset-config wikitext-103-raw-v1
    --max-length 128
    --train-batchsize 32
    --eval-batchsize 32
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
    --save-dir "$SCRATCH"
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

exp_name="001F_gpt2_L${layer}_${variant_name}"
echo "=== Train: layer=${layer}, variant=${variant_name} ==="

# Skip if results already exist
if [ -f "${SCRATCH}/${exp_name}/results.json" ]; then
    echo "Results already exist for ${exp_name}, skipping."
    exit 0
fi

uv run python scripts/training/train_sae_online.py \
    "${COMMON_ARGS[@]}" \
    "${VARIANT_ARGS[@]}" \
    --exp-name "$exp_name"

echo "done"
