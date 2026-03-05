#!/bin/bash

#SBATCH --job-name=sweep_lowrank
#SBATCH --output=%x_%j_%a.log
#SBATCH --gpus=1
#SBATCH --time=09:00:00         # Hours:Mins:Secs
#SBATCH --array=0-71%12         # Sweep 12 layers × 6 lowrank_coeff values


nvidia-smi --list-gpus
cd ~/work/VASAE


# record some potentially useful details about the job:
echo "Running on host $(hostname)"
echo "Started on $(date)"
echo "Directory is $(pwd)"
echo "Slurm job ID is ${SLURM_JOBID}"
echo "Slurm array task ID is ${SLURM_ARRAY_TASK_ID}"
echo "This jobs runs on the following machines:"
echo "${SLURM_JOB_NODELIST}"
printf "\n\n"

source ~/miniforge3/bin/activate
conda activate qwen

echo $(which python)

# Define lowrank coeffs to sweep (0.0 to 1.0 in steps of 0.1)
lowrank_coeffs=(0.0 0.2 0.4 0.6 0.8 1.0)

# Define layers to sweep (0-11)
layers=(0 1 2 3 4 5 6 7 8 9 10 11)

# Convert linear task ID to 2D (layer_idx, coeff_idx)
# Total tasks = 12 layers × 6 coeffs = 72
task_id=$SLURM_ARRAY_TASK_ID
num_coeffs=${#lowrank_coeffs[@]}
layer_idx=$((task_id / num_coeffs))
coeff_idx=$((task_id % num_coeffs))

# Get the current layer and coeff using the indices
layer="${layers[$layer_idx]}"
coeff="${lowrank_coeffs[$coeff_idx]}"
exp_name="vasae_lowrank_layer${layer}_coeff${coeff}"
echo "Running experiment: $exp_name with layer transformer.h.$layer and --lowrank-coeff $coeff"

python scripts/train_sae_gpt2_hf.py \
    --exp-name "$exp_name" \
    --layer-name "transformer.h.$layer" \
    --lowrank-coeff "$coeff" \
    --wandb-group "sweep_lowrank"


echo "done"
