# CLAUDE.md

## Project Overview

VASAE (Vocab-Aligned Sparse Auto-Encoder) is a research project for training Sparse Auto-Encoders whose decoder aligns with token vocabulary embeddings, enabling interpretable latent representations of neural network activations.

## Setup & Dependencies
- Python 3.12, managed with **uv** 




## Architecture
scripts/ for entry points
"/scratch/b5bq/pu22650.b5bq/VASAE_out" for output storage
exp/ for job script and log, run job in hpc with slurm

### Source Layout: `src/vasae/`

- **`models/sae_hf.py`** — Core SAE model (HuggingFace `PreTrainedModel`). Defines `SAEModel` and `SAEConfig` with encoder variants (Linear, MLP), sparsity modules (TopK, BatchTopK, Identity/L1), and optional low-rank decoder decomposition. Decoder can be "tied" to GPT-2 vocab embeddings.
- **`models/factory.py`** — Factory functions for creating SAE models and loading GPT-2 components (embeddings, unembeddings). Handles legacy model class names.
- **`data/dataset.py`** — `GPT2LayerActivations` dataset reads memory-mapped `.dat` files of pre-extracted activations. `get_dataloader()` creates train/valid/test splits (70/20/10).
- **`data/data_schema.py`** — `Meta` and `LayerMeta` types for activation file metadata.
- **`engine/train.py`** — Training loop (`train_one_epoch`) with Adam optimizer, MSE + L1 loss, logit lens metrics, optional wandb logging.
- **`engine/evaluate.py`** — Evaluation loop (no gradients), aggregates metrics across batches.
- **`metrics/logitlens.py`** — Logit lens accuracy: measures if reconstructed activations predict the same tokens as originals via the unembedding matrix.
- **`metrics/interface.py`** — `IMetric` abstract interface, `MetricComposer` for combining metrics, `Aggregator` for batch-weighted accumulation.
- **`configs/`** — Dataclass configs for data (`DataConfig`) and training (`TrainConfig`).



