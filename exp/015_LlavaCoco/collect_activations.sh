#!/bin/bash

#SBATCH --job-name=llava_collect
#SBATCH --output=exp/llava_coco/log/%x_%j.log
#SBATCH --gpus=1
#SBATCH --time=24:00:00

nvidia-smi --list-gpus
cd ~/work/VASAE

echo "Running on host $(hostname)"
echo "Started on $(date)"
echo "Directory is $(pwd)"
echo "Slurm job ID is ${SLURM_JOBID}"
echo "This jobs runs on the following machines:"
echo "${SLURM_JOB_NODELIST}"
printf "\n\n"

source ~/miniforge3/bin/activate
conda activate qwen

echo $(which python)

python scripts/collect_llava_activations.py \
    --max-length 768 \
    --num-examples 20000 \
    --layers "0,4,8,12,16,20,24,28,31"

echo "done"
