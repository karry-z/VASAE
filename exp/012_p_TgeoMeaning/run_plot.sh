#!/bin/bash

#SBATCH --job-name=012d_plot
#SBATCH --output=exp/012_p_TgeoMeaning/logs/%x_%j.log
#SBATCH --time=0:30:00

cd ~/work/VASAE
echo "Running on host $(hostname)"
echo "Started on $(date)"
printf "\n\n"

echo "=== 012d: Aggregate + visualize ==="

uv run python scripts/plot_tgeo_meaning.py \
    --weight-dir exp/012_p_TgeoMeaning/weight_only \
    --data-dir exp/012_p_TgeoMeaning/data \
    --io-dir exp/012_p_TgeoMeaning/io_full \
    --output-dir exp/012_p_TgeoMeaning/figures \
    --layers 0-11 \
    --data-layers 2,6,11

echo "done"
