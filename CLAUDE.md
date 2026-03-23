# CLAUDE.md

## Project Overview

VASAE (Vocab-Aligned Sparse Auto-Encoder) is a research project for training Sparse Auto-Encoders whose decoder aligns with token vocabulary embeddings, enabling interpretable latent representations of neural network activations.

## HPC Environment

Runs on **Isambard-AI** (aarch64). Jobs are submitted via **Slurm**.

### Storage Layout

- **`/home/b5bq/pu22650.b5bq/work/proj_temp/`** — Project directory (in `$HOME`). User-specific storage for configuration files, submission scripts, job output files. Accessible to user and project group members. Not intended for large data.
  - **`scripts/`** — Entry points
  - **`src/`** — Packages. Shared utilities (e.g. logger) live here — use `shared_utils.log` logger rather than `print`.
  - **`exp/`** — Experiments. Each one contains Slurm job scripts.
- **`/scratch/b5bq/pu22650.b5bq/`** — User-specific working data (checkpoints, intermediate I/O, container images). Short-lived data for running jobs.
- **`/projects/b5bq/`** — Project-specific shared storage (input datasets, shared Conda environments, shared container images). Accessible only to project members.

## Setup & Dependencies
- Python 3.12, managed with **uv**

## Rules
- Do **not** show any progress bar — output goes to log files where progress bars are not expected.

## Architecture

- **`scripts/`** — Entry point scripts (training, evaluation, analysis)
- **`exp/`** — Experiment directories, each containing Slurm job scripts and logs
- **`notebooks/`** — Jupyter notebooks
- **`tests/`** — Tests
- **`/scratch/b5bq/pu22650.b5bq/VASAE_out`** — Output storage (checkpoints, results)

### Source Layout: `src/`

#### `src/vasae/` — Core VASAE package

- **`models/sae.py`** — Core SAE model (HuggingFace `PreTrainedModel`). `SAEModel` and `SAEConfig` with encoder variants, sparsity modules, and optional low-rank decoder decomposition. Decoder can be "tied" to GPT-2 vocab embeddings.
- **`models/dualpath_sae.py`** — Dual-path SAE variant.
- **`models/decompose_sae.py`** — Decomposition SAE variant.
- **`models/encoders.py`** — Encoder architectures (Linear, MLP).
- **`models/sparsity.py`** — Sparsity modules (TopK, BatchTopK, Identity/L1).
- **`models/factory.py`** — Factory functions for creating SAE models and loading GPT-2 components.
- **`data/dataset.py`** — `GPT2LayerActivations` dataset reads memory-mapped `.dat` files. `get_dataloader()` creates train/valid/test splits (70/20/10).
- **`data/activation_source.py`** — Activation source abstraction.
- **`data/schema.py`** — `Meta` and `LayerMeta` types for activation file metadata.
- **`engine/trainer.py`** — Trainer class.
- **`engine/train.py`** — Training loop with Adam optimizer, MSE + L1 loss, logit lens metrics, optional wandb logging.
- **`engine/evaluate.py`** — Evaluation loop (no gradients), aggregates metrics across batches.
- **`engine/intervention.py`** — Causal intervention / ablation utilities.
- **`engine/config.py`** — Dataclass configs for data and training.
- **`metrics/base.py`** — `IMetric` abstract interface, `MetricComposer`, `Aggregator` for batch-weighted accumulation.
- **`metrics/logitlens.py`** — Logit lens accuracy: measures if reconstructed activations predict the same tokens as originals via the unembedding matrix.
- **`metrics/ce_loss.py`** — Cross-entropy loss metric.
- **`metrics/variance_explained.py`** — Variance explained metric.
- **`utils/log.py`** — Logger setup. Use this logger, not `print`.
- **`utils/seed.py`** — Random seed utilities.

#### `src/shared_utils/` — Shared utilities across projects

- **`log.py`** — Shared logger.
- **`seed.py`** — Shared seed utilities.

#### `src/easy_transformer/` — Easy-Transformer modules

- **`ioi_dataset.py`** — IOI (Indirect Object Identification) dataset.
- **`ioi_redwood_adapter.py`** — Redwood adapter for IOI.



