#!/bin/bash

#SBATCH --job-name=009_online_sweep
#SBATCH --output=exp/009_p_OnlineSAESweep/logs/%x_%j_%a.log
#SBATCH --gpus=1
#SBATCH --time=16:00:00
#SBATCH --array=0-71%10

nvidia-smi --list-gpus
cd ~/work/VASAE

echo "Running on host $(hostname)"
echo "Started on $(date)"
echo "Directory is $(pwd)"
echo "Slurm job ID is ${SLURM_JOBID}"
echo "Slurm array task ID is ${SLURM_ARRAY_TASK_ID}"
echo "This jobs runs on the following machines:"
echo "${SLURM_JOB_NODELIST}"
printf "\n\n"

# Sweep: 12 layers x 3 k values x 2 anchor coeffs = 72 tasks
layers=(0 1 2 3 4 5 6 7 8 9 10 11)
ks=(8 16 32)
anchors=(0 1e-4)

task_id=${SLURM_ARRAY_TASK_ID}
layer_idx=$(( task_id / 6 ))
rem=$(( task_id % 6 ))
k_idx=$(( rem / 2 ))
anchor_idx=$(( rem % 2 ))

layer="${layers[$layer_idx]}"
k="${ks[$k_idx]}"
anchor="${anchors[$anchor_idx]}"

# Use a filesystem-safe tag in exp name.
anchor_tag="${anchor//./p}"

SCRATCH="/scratch/b5bq/pu22650.b5bq/VASAE_out/009_online_sweep"
exp_name="009_online_gpt2_L${layer}_k${k}_a${anchor_tag}"

mkdir -p "$SCRATCH"

echo "=== Train online SAE: layer=${layer}, k=${k}, anchor_coeff=${anchor} ==="

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
    --k "$k" \
    --nonneg-latents \
    --tied-decoder \
    --anchor-coeff "$anchor" \
    --num-epochs 5 \
    --lr 1e-3 \
    --wandb-group "009_online_sweep" \
    --exp-name "$exp_name" \
    --save-dir "$SCRATCH"

echo "done"
