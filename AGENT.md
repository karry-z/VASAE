# Agent Guide

## Project Overview

VASAE (Vocab-Aligned Sparse Auto-Encoder) is a research codebase for training Sparse Auto-Encoders whose decoder can align with token vocabulary embeddings. The goal is to make latent SAE features interpretable through vocabulary-space semantics while preserving useful reconstruction and downstream behavior.

The main workflow is:

1. Collect or stream language-model activations.
2. Train an offline or online SAE.
3. Evaluate reconstruction, logit lens behavior, causal effects, and feature activity.
4. Analyze alignment and generate experiment reports, plots, or aggregate tables.

## Environment

- Python 3.12, managed with `uv`.
- Dependencies are declared in `pyproject.toml`; PyTorch and torchvision use the configured CUDA 12.8 index.
- Jobs normally run on Isambard-AI (`aarch64`) through Slurm.
- Runtime experiment tracking may use wandb, but most entrypoints support `--no-wandb`.


## Storage And Jobs

- Use the project env vars from `.bashrc` instead of hard-coded machine paths:
  - `VASAE_HOME` points to the repository checkout, for example `~/work/VASAE`.
  - `VASAE_OUT` points to project output storage, for example `$PROJECTDIR/VASAE`.
- Keep source code, small configs, Slurm scripts, and experiment reports under `$VASAE_HOME`.
- Put checkpoints, cached datasets, generated activations, intermediate files, and bulky results under `$VASAE_OUT`.
- Use `$PROJECTDIR` for project-shared input datasets, environments, or container images.
- Do not add large generated files, logs, checkpoints, `.pt` tensors, cached datasets, or PDFs to version control unless the user explicitly asks for that artifact.
- Experiment directories under `exp/` are part of the research record. Preserve existing `report.md` files and Slurm scripts unless the task is specifically about that experiment.

For job-oriented scripts, avoid progress bars because output usually goes to logs. Follow existing patterns such as setting `HF_HUB_DISABLE_PROGRESS_BARS=1`, `TQDM_DISABLE=1`, and using dataset/transformers progress disabling where applicable.

## Architecture

Core package: `src/vasae/`

- `models/` contains `SAEModel`, `SAEConfig`, encoder variants, sparsity modules, and factory helpers for loading model components.
- `data/` contains activation sources, memmap datasets, corpus helpers, and schema/config types.
- `engine/` contains trainer classes, train/evaluate loops, intervention utilities, and training config types.
- `metrics/` contains the `IMetric` interface, metric composition/aggregation, logit lens accuracy, cross-entropy recovery, activity, and variance explained metrics.
- `losses/` contains anchor and cosine-similarity loss helpers.
- `analysis/` contains alignment, hook, SAE loading, I/O, and tensor-stat utilities used by scripts and notebooks.
- `utils/` contains logging and seed helpers.

Supporting packages:

- `src/shared_utils/` provides shared logging and seed utilities used by scripts and experiments.
- `src/easy_transformer/` contains IOI dataset and Redwood adapter code.

Script layout:

- `scripts/training/` trains offline SAEs from stored activations or online SAEs from streamed HuggingFace model activations.
- `scripts/eval/` evaluates trained SAEs and IOI causal/feature-sweep behavior.
- `scripts/collect/` collects GPT-2 activations and fetches or validates corpora.
- `scripts/plot/` produces analysis figures and case-study outputs.
- `scripts/aggregate/` collects experiment result JSON into summary CSVs/tables.

Experiment layout:

- `exp/F001_Benchmarking/` and `exp/F001A_AblationSoft/` cover formal benchmark and ablation experiments.
- `exp/F002_AlignmentAnalysis/` covers alignment-analysis runs and reports.
- `exp/021_IOI/` and `exp/022_IOI_casestudy/` cover IOI feature-sweep and case-study work.
- `exp/023_Dataset/` contains dataset-mixture collection, training, heldout evaluation, and summarization scripts.

## Agent Working Rules

- Prefer `uv run ...` for Python commands so the project environment is used consistently.
- Prefer `$VASAE_HOME`, `$VASAE_OUT`, and `$PROJECTDIR` in docs, scripts, and job files when paths should be portable across checkout/output locations.
- Use `vasae.utils.log.get_logger` or `shared_utils.log.get_logger` in runtime scripts instead of adding bare `print` output, except for small CLI summary scripts that already follow a print-based pattern.
- Keep code changes close to existing module boundaries and patterns.
- Add or update focused pytest coverage when changing models, losses, metrics, trainer behavior, data schemas, or evaluation logic.
- For documentation-only changes, reading the updated file is sufficient verification unless the user asks for tests.
- Do not rewrite unrelated experiment history, generated reports, or local worktree changes.
