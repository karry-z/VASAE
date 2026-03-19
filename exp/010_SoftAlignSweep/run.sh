#!/bin/bash

#SBATCH --job-name=010_soft_align
#SBATCH --output=exp/010_p_SoftAlignSweep/logs/%x_%j_%a.log
#SBATCH --gpus=1
#SBATCH --time=16:00:00
#SBATCH --array=0-107%10

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

# Sweep: 12 layers x 3 k values x 3 anchor coeffs = 108 tasks
# No --tied-decoder: decoder is learned freely, anchor loss softly aligns it.
layers=(0 1 2 3 4 5 6 7 8 9 10 11)
ks=(8 16 32)
anchors=(1e-3 1e-4 1e-5)

task_id=${SLURM_ARRAY_TASK_ID}
n_anchors=${#anchors[@]}
n_ks=${#ks[@]}
combos=$(( n_ks * n_anchors ))

layer_idx=$(( task_id / combos ))
rem=$(( task_id % combos ))
k_idx=$(( rem / n_anchors ))
anchor_idx=$(( rem % n_anchors ))

layer="${layers[$layer_idx]}"
k="${ks[$k_idx]}"
anchor="${anchors[$anchor_idx]}"

anchor_tag="${anchor//./p}"

SCRATCH="/scratch/b5bq/pu22650.b5bq/VASAE_out/010_soft_align"
exp_name="010_soft_gpt2_L${layer}_k${k}_a${anchor_tag}"

mkdir -p "$SCRATCH"

echo "=== Train soft-align SAE: layer=${layer}, k=${k}, anchor_coeff=${anchor} ==="

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
    --anchor-coeff "$anchor" \
    --num-epochs 5 \
    --lr 1e-3 \
    --wandb-group "010_soft_align" \
    --exp-name "$exp_name" \
    --save-dir "$SCRATCH"

echo "done"
