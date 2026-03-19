#!/bin/bash

#SBATCH --job-name=llava_sae
#SBATCH --output=exp/llava_coco/log/%x_%j_%a.log
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --array=0-8%4

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

source ~/miniforge3/bin/activate
conda activate qwen

echo $(which python)

layers=(0 4 8 12 16 20 24 28 31)
layer=${layers[$SLURM_ARRAY_TASK_ID]}
exp_name="vasae_llava_layer${layer}"
echo "Running experiment: $exp_name with layer model.language_model.layers.$layer"

python scripts/train_sae_llava.py \
    --exp-name "$exp_name" \
    --layer-name "model.language_model.layers.$layer" \
    --sae-save-path "/scratch/b5bq/pu22650.b5bq/VASAE_out/sae_llava_layer${layer}.pth" \
    --wandb-group "llava_coco"

echo "done"
