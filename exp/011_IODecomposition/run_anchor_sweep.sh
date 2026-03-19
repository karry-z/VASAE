#!/bin/bash

#SBATCH --job-name=011_anchor_sweep
#SBATCH --output=exp/011_p_IODecomposition/logs/%x_%j_%a.log
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --array=0-11

cd ~/work/VASAE
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

# Sweep: 4 anchor_coeffs x 3 layers = 12 tasks
anchors=(0 0.01 0.1 1.0)
layers=(2 6 11)

task_id=${SLURM_ARRAY_TASK_ID}
n_layers=${#layers[@]}

anchor_idx=$(( task_id / n_layers ))
layer_idx=$(( task_id % n_layers ))

anchor="${anchors[$anchor_idx]}"
layer="${layers[$layer_idx]}"
anchor_tag="${anchor//./p}"

SCRATCH="/scratch/b5bq/pu22650.b5bq/VASAE_out/011_anchor_sweep"
exp_name="011_anchor_gpt2_L${layer}_k32_a${anchor_tag}"

mkdir -p "$SCRATCH"

echo "=== Train anchor sweep SAE: layer=${layer}, k=32, anchor_coeff=${anchor} ==="

uv run python scripts/train_sae_online.py \
    --model-name gpt2 \
    --layer-idx "$layer" \
    --dataset wikitext \
    --dataset-config wikitext-103-raw-v1 \
    --max-length 128 \
    --train-batchsize 32 \
    --eval-batchsize 32 \
    --train-samples 4000 \
    --eval-samples 1000 \
    --test-samples 1000 \
    --sparsity-type topk \
    --k 32 \
    --nonneg-latents \
    --anchor-coeff "$anchor" \
    --num-epochs 5 \
    --lr 1e-3 \
    --wandb-group "011_anchor_sweep" \
    --exp-name "$exp_name" \
    --save-dir "$SCRATCH"

echo "done"
