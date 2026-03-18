# VASAE: Vocab-Aligned Sparse Auto-Encoder

**VASAE** is a research framework for training Sparse Auto-Encoders (SAEs) whose decoder aligns with token vocabulary embeddings, enabling interpretable latent representations of neural network activations.

The approach ties the SAE decoder to GPT-2's unembedding matrix, so each learned feature corresponds directly to directions in vocabulary space — making features inherently interpretable through their token-level semantics.

## Installation

The code requires `python>=3.12` and the following dependencies. We recommend using [uv](https://docs.astral.sh/uv/) for environment management.

```bash
git clone https://github.com/<your-org>/VASAE.git
cd VASAE
uv sync
```

## Getting Started

### Collecting Activations

Extract GPT-2 layer activations to memory-mapped files:

```bash
uv run python scripts/collect_gpt2_activations.py
```

### Training

Train an offline SAE on pre-collected activations:

```bash
uv run python scripts/train_sae_offline.py
```

Train an online SAE with activations computed on the fly:

```bash
uv run python scripts/train_sae_online.py
```

Train model variants:

```bash
# Dual-path SAE
uv run python scripts/train_dualpath_sae.py

# Decomposition SAE
uv run python scripts/train_decompose_sae.py
```

### Evaluation

```bash
# Evaluate loss recovered
uv run python scripts/eval_loss_recovered.py

# Evaluate with online SAE
uv run python scripts/eval_sae_online.py

# IOI causal intervention
uv run python scripts/eval_ioi_causal.py
```

### Analysis

A suite of analysis scripts is provided for feature-level interpretability:

```bash
uv run python scripts/analyze_feature_vocab_alignment.py
uv run python scripts/analyze_feature_io_decomposition.py
uv run python scripts/analyze_context_interpretability.py
uv run python scripts/analyze_logit_attribution_sparsity.py
uv run python scripts/analyze_token_frequency.py
```

## Model Overview

<table>
<tr>
<th>Component</th>
<th>Description</th>
</tr>
<tr>
<td>SAEModel</td>
<td>Core sparse auto-encoder with configurable encoder, sparsity, and decoder. HuggingFace <code>PreTrainedModel</code> compatible.</td>
</tr>
<tr>
<td>DualPathSAE</td>
<td>Dual-path variant separating vocab-aligned and residual subspaces.</td>
</tr>
<tr>
<td>DecomposeSAE</td>
<td>Decomposition variant for analyzing feature structure.</td>
</tr>
</table>

**Encoder variants:** Linear, MLP

**Sparsity modules:** TopK, BatchTopK, Identity/L1

**Decoder modes:** Standard, Vocab-Tied (to GPT-2 unembedding), Low-Rank decomposition

## Metrics

| Metric | Description |
| --- | --- |
| Logit Lens Accuracy | Whether reconstructed activations predict the same tokens as originals via the unembedding matrix |
| CE Loss | Cross-entropy loss of reconstructed vs. original activations |
| Variance Explained | Fraction of activation variance captured by the reconstruction |

## Project Structure

```
VASAE/
├── scripts/                # Entry points: training, evaluation, analysis
├── src/
│   ├── vasae/              # Core package
│   │   ├── models/         # SAE, DualPathSAE, DecomposeSAE, encoders, sparsity
│   │   ├── data/           # Dataset, activation sources, schema
│   │   ├── engine/         # Trainer, train/eval loops, intervention, configs
│   │   ├── metrics/        # Logit lens, CE loss, variance explained
│   │   └── utils/          # Logger, seed
│   ├── shared_utils/       # Shared logger and seed utilities
│   └── easy_transformer/   # IOI dataset and adapters
├── exp/                    # Experiment configs and Slurm job scripts
├── notebooks/              # Jupyter notebooks
├── tests/                  # Tests
└── pyproject.toml
```

## License

TBD

## Citing VASAE

If you use VASAE in your research, please cite:

```bibtex
@misc{vasae2025,
  title   = {VASAE: Vocab-Aligned Sparse Auto-Encoder},
  year    = {2025},
  url     = {https://github.com/<your-org>/VASAE}
}
```
