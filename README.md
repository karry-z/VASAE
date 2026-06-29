# VASAE: Naming SAE Dictionary Directions with Vocabulary-Aligned Anchoring

VASAE trains sparse autoencoders with learnable decoder directions softly anchored to fixed token embeddings. Each feature receives an intrinsic nearest-token name by cosine similarity between its decoder direction and the vocabulary embedding space. These names are geometric labels, not full semantic explanations or causal claims.

## Installation

This project requires Python 3.12 or newer, as specified in `pyproject.toml`.

```bash
uv sync
```

No license has been specified for this repository.

## Minimal Usage

Train a plain TopK SAE baseline:

```bash
uv run python scripts/train_vasae.py --method plain
```

Train VASAE-soft:

```bash
uv run python scripts/train_vasae.py --method vasae_soft --anchor-coeff 0.1
```

Evaluate reconstruction metrics for a saved run:

```bash
uv run python scripts/eval_reconstruction.py --checkpoint outputs/runs/gpt2_L11_vasae_soft
```

Analyze vocabulary alignment and nearest-token names:

```bash
uv run python scripts/analyze_alignment.py --checkpoint outputs/runs/gpt2_L11_vasae_soft --top-k 5
```

## Method Boundary

VASAE-soft is the main method: a sparse autoencoder with a learnable decoder and a vocabulary-anchor regularization term.

The `hard_tied_baseline` option is a fixed-decoder baseline or ablation. It is not the main VASAE method.

Nearest-token names are assigned from decoder geometry. They should be read as vocabulary-anchor labels, not as complete feature interpretations.

## Project Structure

```text
scripts/
src/vasae/
  analysis.py
  data.py
  engine.py
  metrics.py
  models.py
  utils.py
```

## Reproduction Boundary

This release keeps the core implementation and command-line entry points only. It does not track experiment results, generated summaries, paper figures, notebooks, or plotting programs. Local runs write checkpoints and metrics under ignored output directories such as `outputs/runs`.

## Limitations

Nearest-token names are not semantic explanations, not causal evidence, and not guaranteed to be context-invariant. They identify nearby vocabulary directions in the embedding space used for anchoring.

## Citation

No citation metadata has been specified in this repository.
