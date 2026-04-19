#!/bin/bash

#SBATCH --job-name=F002_diag_cs
#SBATCH --output=exp/F002_AlignmentAnalysis/logs/%x_%j.log
#SBATCH --gpus=1
#SBATCH --time=0:15:00

cd ~/work/VASAE
VENV_SITE="$(uv run python -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="${VENV_SITE}/nvidia/cusparselt/lib:${VENV_SITE}/nvidia/cusparse/lib:${LD_LIBRARY_PATH}"

uv run python scripts/analyze/alignment/diagnose_casestudy.py
