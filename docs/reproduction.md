# Reproduction

These commands are templates for the final paper-facing workflow. Replace model, layer, sample count, and output directory values as needed.

## Install

```bash
uv sync
```

For figure generation or W&B logging:

```bash
uv sync --extra figures
uv sync --extra wandb
```

## Train A Plain SAE

```bash
uv run python scripts/train_vasae.py \
  --model-name gpt2 \
  --layer-idx 11 \
  --method plain \
  --dataset wikitext \
  --dataset-config wikitext-103-raw-v1 \
  --train-samples 8000 \
  --eval-samples 2000 \
  --test-samples 1000 \
  --batch-size 32 \
  --dim-sparse vocab \
  --k 32 \
  --num-epochs 5 \
  --no-wandb \
  --save-dir experiments/paper/runs
```

## Train VASAE-Soft

```bash
uv run python scripts/train_vasae.py \
  --model-name gpt2 \
  --layer-idx 11 \
  --method vasae_soft \
  --anchor-coeff 0.1 \
  --anchor-every 1 \
  --dataset wikitext \
  --dataset-config wikitext-103-raw-v1 \
  --train-samples 8000 \
  --eval-samples 2000 \
  --test-samples 1000 \
  --batch-size 32 \
  --dim-sparse vocab \
  --k 32 \
  --num-epochs 5 \
  --no-wandb \
  --save-dir experiments/paper/runs
```

## Optional Fixed-Decoder Baseline

Use this only as a baseline or ablation:

```bash
uv run python scripts/train_vasae.py \
  --model-name gpt2 \
  --layer-idx 11 \
  --method hard_tied_baseline \
  --dataset wikitext \
  --dataset-config wikitext-103-raw-v1 \
  --dim-sparse vocab \
  --k 32 \
  --no-wandb \
  --save-dir experiments/paper/runs
```

## Evaluate Reconstruction

```bash
uv run python scripts/eval_reconstruction.py \
  --checkpoint experiments/paper/runs/gpt2_L11_vasae_soft \
  --model-name gpt2 \
  --layer-idx 11 \
  --dataset wikitext \
  --dataset-config wikitext-103-raw-v1 \
  --samples 1000 \
  --batch-size 32
```

Add optional language-model metrics when the environment can run the extra forward passes:

```bash
uv run python scripts/eval_reconstruction.py \
  --checkpoint experiments/paper/runs/gpt2_L11_vasae_soft \
  --model-name gpt2 \
  --layer-idx 11 \
  --logit-lens \
  --ce-recovery
```

## Analyze Vocabulary Alignment

```bash
uv run python scripts/analyze_alignment.py \
  --checkpoint experiments/paper/runs/gpt2_L11_vasae_soft \
  --model-name gpt2 \
  --top-k 5
```

The output assigns nearest-token geometric labels to decoder directions.

## Generate Paper Figures

```bash
uv run --extra figures python scripts/make_paper_figures.py \
  --paper-dir experiments/paper \
  --formats png,pdf
```

Cleaned paper-facing summaries and generated figures live under `experiments/paper/`.
