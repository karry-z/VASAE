#!/bin/bash

#SBATCH --job-name=011_eval_009
#SBATCH --output=exp/011_p_IODecomposition/logs/%x_%j_%a.log
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --array=0-23

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

# Sweep: 12 layers x 2 anchor values (0, 1e-4), k=32 only = 24 tasks
layers=(0 1 2 3 4 5 6 7 8 9 10 11)
anchors=(0 1e-4)

task_id=${SLURM_ARRAY_TASK_ID}
n_anchors=${#anchors[@]}

layer_idx=$(( task_id / n_anchors ))
anchor_idx=$(( task_id % n_anchors ))

layer="${layers[$layer_idx]}"
anchor="${anchors[$anchor_idx]}"
anchor_tag="${anchor//./p}"

SCRATCH="/scratch/b5bq/pu22650.b5bq/VASAE_out/009_online_sweep"
sae_path="${SCRATCH}/009_online_gpt2_L${layer}_k32_a${anchor_tag}"

echo "=== Eval 009 SAE: layer=${layer}, k=32, anchor=${anchor} ==="
echo "SAE path: ${sae_path}"

if [ ! -d "$sae_path" ]; then
    echo "ERROR: SAE directory not found: $sae_path"
    exit 1
fi

uv run python scripts/eval_sae_online.py \
    --sae-path "$sae_path" \
    --model-name gpt2 \
    --layer-idx "$layer" \
    --n-samples 1000 \
    --batch-size 32 \
    --max-length 128 \
    --device cuda

echo "done"
