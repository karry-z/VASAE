# Method

VASAE trains a TopK sparse autoencoder on language-model activations. For an activation vector `x`, the encoder produces sparse activations `z`, and the decoder reconstructs `x_hat`.

The reconstruction objective is mean squared error:

```text
L_recon = MSE(x_hat, x)
```

## VASAE-Soft

VASAE-soft keeps the decoder learnable. It adds a vocabulary-anchor loss that encourages each decoder direction to stay close to at least one fixed token embedding direction.

For normalized decoder directions `d_i` and normalized token embeddings `e_j`, the default anchor term uses nearest-token cosine similarity:

```text
L_anchor = - mean_i max_j cosine(d_i, e_j)
L_total = L_recon + lambda * L_anchor
```

`lambda` is controlled by `--anchor-coeff` in `scripts/train_vasae.py`.

## Naming

After training, each decoder direction is labeled with its nearest token embeddings by cosine similarity. These names are intrinsic geometric labels in the chosen embedding space. They are not full semantic explanations, not causal evidence, and not guaranteed to remain stable across contexts.

## Baselines

The plain baseline is a TopK SAE with a learnable decoder and no vocabulary-anchor loss.

The `hard_tied_baseline` option fixes decoder directions to vocabulary embeddings. It is only a baseline or ablation and is not the main VASAE method.

## Metrics

The retained evaluation path reports reconstruction MSE and variance explained. Optional language-model forward-pass metrics include CE recovery and logit-lens agreement.
