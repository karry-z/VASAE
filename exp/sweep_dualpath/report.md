# Dual-Path SAE Experiment Report

## 1. Experiment Overview

**Goal**: Validate the proposal's Dual-Path SAE (Section 3.4) on GPT-2. The architecture decomposes residual stream activations into:
- **Sparse token path**: `z @ W_E`, decoder frozen to embedding matrix, L1 sparsity
- **Dense path**: `y @ P_k^T`, decoder frozen to PCA basis of **embedding-orthogonal** residuals, L1 sparsity

**Key differences from prior `DecomposeSAEModel`**:

| | DecomposeSAEModel (old) | DualPathSAE (this exp) |
|---|---|---|
| Sparse sparsity | TopK (k=8) | L1 penalty (λ_z) |
| Dense sparsity | None | L1 penalty (λ_y) |
| PCA target | Token-reconstruction residuals (h − z_s @ E^T) | Embedding-orthogonal residuals ((I − P_E) @ h) |
| Bias | Learned bias vector | mean_r (mean of emb-orthogonal residuals) |
| Phase 1 | Train full sparse SAE first | No pre-training; compute P_E analytically |

**Model**: GPT-2 (117M, absolute position encoding, d=768, V=50257)

**Sweep config**:
- Layers: 0–11 (all 12 transformer blocks)
- d_pca: {2, 4, 8}
- λ_z = 1e-3, λ_y = 1e-4
- lr = 1e-3 (Adam), 20 epochs, batch_size = 128
- Slurm job: `2807073`, 36 array tasks (12 layers × 3 d_pca)

---

## 2. Phase 1: Embedding-Orthogonal Subspace Analysis

### 2.1 W_E Rank and Projector

| Property | Value |
|---|---|
| W_E shape | (50257, 768) |
| Numerical rank of W_E | 760 |
| Null space dimension | 8 |
| ‖P_E − I‖_F | 2.828 |

**Interpretation**: GPT-2's embedding matrix has rank 760, leaving an 8-dimensional subspace orthogonal to all token embeddings. ‖P_E − I‖_F = 2√2 ≈ 2.83 is consistent with 8 missing dimensions (each contributes eigenvalue 1 to the difference).

### 2.2 PCA of Embedding-Orthogonal Residuals

Eigenvalues of Cov((I − P_E) @ h) for layer 5 (representative):

| PC | Eigenvalue | Cumulative Explained Var (%) |
|---|---|---|
| 1 | _____ | _____ |
| 2 | _____ | _____ |
| 3 | _____ | _____ |
| 4 | _____ | _____ |
| 5 | _____ | _____ |
| 6 | _____ | _____ |
| 7 | _____ | _____ |
| 8 | _____ | _____ |

**Orthogonality check** ‖P_k^T @ W_E^T‖_F per layer:

| Layer | d_pca=2 | d_pca=4 | d_pca=8 |
|---|---|---|---|
| 0 | _____ | _____ | _____ |
| 1 | _____ | _____ | _____ |
| 2 | _____ | _____ | _____ |
| 3 | _____ | _____ | _____ |
| 4 | _____ | _____ | _____ |
| 5 | _____ | _____ | _____ |
| 6 | _____ | _____ | _____ |
| 7 | _____ | _____ | _____ |
| 8 | _____ | _____ | _____ |
| 9 | _____ | _____ | _____ |
| 10 | _____ | _____ | _____ |
| 11 | _____ | _____ | _____ |

**Expected**: Values ≈ 0 (numerical noise only), confirming P_k directions are orthogonal to all token embeddings. If values are large, pinv precision may be an issue.

### 2.3 Eigenvalue Spectrum Across Layers

| Layer | λ_1 | λ_2 | λ_3 | λ_4 | λ_5 | λ_6 | λ_7 | λ_8 | λ_9 (should ≈ 0) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | _____ | _____ | _____ | _____ | _____ | _____ | _____ | _____ | _____ |
| 5 | _____ | _____ | _____ | _____ | _____ | _____ | _____ | _____ | _____ |
| 11 | _____ | _____ | _____ | _____ | _____ | _____ | _____ | _____ | _____ |

**Questions to answer**:
- Does λ_1 dominate (position encoding)?
- How does the spectrum change from early to late layers?
- Is λ_9 truly ≈ 0, confirming the null space is exactly 8-dim?

---

## 3. Phase 2: Training Results

### 3.1 Final Training Metrics (epoch 20)

| Layer | d_pca | Train Loss | Train Recon | Train L1_z | Train L1_y | Train L0_z | Train Acc (%) |
|---|---|---|---|---|---|---|---|
| 0 | 2 | _____ | _____ | _____ | _____ | _____ | _____ |
| 0 | 4 | _____ | _____ | _____ | _____ | _____ | _____ |
| 0 | 8 | _____ | _____ | _____ | _____ | _____ | _____ |
| 1 | 2 | _____ | _____ | _____ | _____ | _____ | _____ |
| 1 | 4 | _____ | _____ | _____ | _____ | _____ | _____ |
| 1 | 8 | _____ | _____ | _____ | _____ | _____ | _____ |
| 2 | 2 | _____ | _____ | _____ | _____ | _____ | _____ |
| 2 | 4 | _____ | _____ | _____ | _____ | _____ | _____ |
| 2 | 8 | _____ | _____ | _____ | _____ | _____ | _____ |
| 3 | 2 | _____ | _____ | _____ | _____ | _____ | _____ |
| 3 | 4 | _____ | _____ | _____ | _____ | _____ | _____ |
| 3 | 8 | _____ | _____ | _____ | _____ | _____ | _____ |
| 4 | 2 | _____ | _____ | _____ | _____ | _____ | _____ |
| 4 | 4 | _____ | _____ | _____ | _____ | _____ | _____ |
| 4 | 8 | _____ | _____ | _____ | _____ | _____ | _____ |
| 5 | 2 | _____ | _____ | _____ | _____ | _____ | _____ |
| 5 | 4 | _____ | _____ | _____ | _____ | _____ | _____ |
| 5 | 8 | _____ | _____ | _____ | _____ | _____ | _____ |
| 6 | 2 | _____ | _____ | _____ | _____ | _____ | _____ |
| 6 | 4 | _____ | _____ | _____ | _____ | _____ | _____ |
| 6 | 8 | _____ | _____ | _____ | _____ | _____ | _____ |
| 7 | 2 | _____ | _____ | _____ | _____ | _____ | _____ |
| 7 | 4 | _____ | _____ | _____ | _____ | _____ | _____ |
| 7 | 8 | _____ | _____ | _____ | _____ | _____ | _____ |
| 8 | 2 | _____ | _____ | _____ | _____ | _____ | _____ |
| 8 | 4 | _____ | _____ | _____ | _____ | _____ | _____ |
| 8 | 8 | _____ | _____ | _____ | _____ | _____ | _____ |
| 9 | 2 | _____ | _____ | _____ | _____ | _____ | _____ |
| 9 | 4 | _____ | _____ | _____ | _____ | _____ | _____ |
| 9 | 8 | _____ | _____ | _____ | _____ | _____ | _____ |
| 10 | 2 | _____ | _____ | _____ | _____ | _____ | _____ |
| 10 | 4 | _____ | _____ | _____ | _____ | _____ | _____ |
| 10 | 8 | _____ | _____ | _____ | _____ | _____ | _____ |
| 11 | 2 | _____ | _____ | _____ | _____ | _____ | _____ |
| 11 | 4 | _____ | _____ | _____ | _____ | _____ | _____ |
| 11 | 8 | _____ | _____ | _____ | _____ | _____ | _____ |

### 3.2 Validation Metrics (epoch 20)

| Layer | d_pca | Valid Loss | Valid Recon | Valid Acc (%) |
|---|---|---|---|---|
| 0 | 8 | _____ | _____ | _____ |
| 5 | 8 | _____ | _____ | _____ |
| 11 | 8 | _____ | _____ | _____ |

(Fill representative layers; full table in supplementary.)

---

## 4. Phase 3: Test Evaluation

### 4.1 Variance Explained

| Layer | d_pca | VE (full) | VE_sparse | VE_dense | MSE_full | MSE_sparse | MSE_dense |
|---|---|---|---|---|---|---|---|
| 0 | 2 | _____ | _____ | _____ | _____ | _____ | _____ |
| 0 | 4 | _____ | _____ | _____ | _____ | _____ | _____ |
| 0 | 8 | _____ | _____ | _____ | _____ | _____ | _____ |
| 1 | 2 | _____ | _____ | _____ | _____ | _____ | _____ |
| 1 | 4 | _____ | _____ | _____ | _____ | _____ | _____ |
| 1 | 8 | _____ | _____ | _____ | _____ | _____ | _____ |
| 2 | 2 | _____ | _____ | _____ | _____ | _____ | _____ |
| 2 | 4 | _____ | _____ | _____ | _____ | _____ | _____ |
| 2 | 8 | _____ | _____ | _____ | _____ | _____ | _____ |
| 3 | 2 | _____ | _____ | _____ | _____ | _____ | _____ |
| 3 | 4 | _____ | _____ | _____ | _____ | _____ | _____ |
| 3 | 8 | _____ | _____ | _____ | _____ | _____ | _____ |
| 4 | 2 | _____ | _____ | _____ | _____ | _____ | _____ |
| 4 | 4 | _____ | _____ | _____ | _____ | _____ | _____ |
| 4 | 8 | _____ | _____ | _____ | _____ | _____ | _____ |
| 5 | 2 | _____ | _____ | _____ | _____ | _____ | _____ |
| 5 | 4 | _____ | _____ | _____ | _____ | _____ | _____ |
| 5 | 8 | _____ | _____ | _____ | _____ | _____ | _____ |
| 6 | 2 | _____ | _____ | _____ | _____ | _____ | _____ |
| 6 | 4 | _____ | _____ | _____ | _____ | _____ | _____ |
| 6 | 8 | _____ | _____ | _____ | _____ | _____ | _____ |
| 7 | 2 | _____ | _____ | _____ | _____ | _____ | _____ |
| 7 | 4 | _____ | _____ | _____ | _____ | _____ | _____ |
| 7 | 8 | _____ | _____ | _____ | _____ | _____ | _____ |
| 8 | 2 | _____ | _____ | _____ | _____ | _____ | _____ |
| 8 | 4 | _____ | _____ | _____ | _____ | _____ | _____ |
| 8 | 8 | _____ | _____ | _____ | _____ | _____ | _____ |
| 9 | 2 | _____ | _____ | _____ | _____ | _____ | _____ |
| 9 | 4 | _____ | _____ | _____ | _____ | _____ | _____ |
| 9 | 8 | _____ | _____ | _____ | _____ | _____ | _____ |
| 10 | 2 | _____ | _____ | _____ | _____ | _____ | _____ |
| 10 | 4 | _____ | _____ | _____ | _____ | _____ | _____ |
| 10 | 8 | _____ | _____ | _____ | _____ | _____ | _____ |
| 11 | 2 | _____ | _____ | _____ | _____ | _____ | _____ |
| 11 | 4 | _____ | _____ | _____ | _____ | _____ | _____ |
| 11 | 8 | _____ | _____ | _____ | _____ | _____ | _____ |

### 4.2 Logit Lens Accuracy (test set)

| Layer | d_pca=2 | d_pca=4 | d_pca=8 |
|---|---|---|---|
| 0 | _____ | _____ | _____ |
| 1 | _____ | _____ | _____ |
| 2 | _____ | _____ | _____ |
| 3 | _____ | _____ | _____ |
| 4 | _____ | _____ | _____ |
| 5 | _____ | _____ | _____ |
| 6 | _____ | _____ | _____ |
| 7 | _____ | _____ | _____ |
| 8 | _____ | _____ | _____ |
| 9 | _____ | _____ | _____ |
| 10 | _____ | _____ | _____ |
| 11 | _____ | _____ | _____ |

---

## 5. Comparison with DecomposeSAE (sweep_decompose)

### 5.1 VE Comparison (d_pca=8)

| Layer | DualPath VE | DualPath VE_sparse | Decompose VE | Decompose VE_sparse |
|---|---|---|---|---|
| 0 | _____ | _____ | _____ | _____ |
| 1 | _____ | _____ | _____ | _____ |
| 2 | _____ | _____ | _____ | _____ |
| 3 | _____ | _____ | _____ | _____ |
| 4 | _____ | _____ | _____ | _____ |
| 5 | _____ | _____ | _____ | _____ |
| 6 | _____ | _____ | _____ | _____ |
| 7 | _____ | _____ | _____ | _____ |
| 8 | _____ | _____ | _____ | _____ |
| 9 | _____ | _____ | _____ | _____ |
| 10 | _____ | _____ | _____ | _____ |
| 11 | _____ | _____ | _____ | _____ |

**Key question**: Does the DualPath VE_sparse >> Decompose VE_sparse? This would confirm that the proposal's approach (embedding-orthogonal PCA + L1) fixes the VE_sparse ≈ 0 problem of the old design.

### 5.2 Logit Lens Accuracy Comparison

| Layer | DualPath Acc (%) | Decompose Acc (%) |
|---|---|---|
| 0 | _____ | _____ |
| 5 | _____ | _____ |
| 11 | _____ | _____ |

---

## 6. Analysis

### 6.1 Effect of d_pca

**Hypothesis**: Since the embedding-orthogonal subspace is exactly 8-dim, d_pca=8 should capture essentially all non-vocab signal. d_pca=2 and d_pca=4 provide ablation points.

| Metric | d_pca=2 (avg across layers) | d_pca=4 | d_pca=8 |
|---|---|---|---|
| VE (full) | _____ | _____ | _____ |
| VE_sparse | _____ | _____ | _____ |
| VE_dense | _____ | _____ | _____ |
| Δ(VE − VE_sparse) | _____ | _____ | _____ |

**Expected**: VE should plateau at d_pca=8 (no more orthogonal directions to capture). VE_sparse should be approximately constant across d_pca (sparse path doesn't depend on dense dimensions).

### 6.2 Layer-wise Trends

**Questions**:
- Layer 0: Is VE_sparse ≈ 1.0? (activations = embeddings, so token path should reconstruct perfectly)
- Deep layers (10–11): Does VE_dense become more important? (activations diverge from embedding space)
- Which layer has the largest gap VE − VE_sparse? (where the dense path matters most)

### 6.3 Sparsity Analysis

| Layer | Mean L0 of z (d_pca=8) | Mean |z|_1 | Mean |y|_1 |
|---|---|---|---|
| 0 | _____ | _____ | _____ |
| 5 | _____ | _____ | _____ |
| 11 | _____ | _____ | _____ |

**Questions**:
- Is the sparse code genuinely sparse (L0 << V=50257)?
- Does λ_z = 1e-3 give a reasonable sparsity level, or does it need tuning?

---

## 7. Sanity Checks

### 7.1 P_E Projector Verification

- [  ] Rank of W_E = 760 for all layers (same embedding matrix)
- [  ] ‖P_E − I‖_F ≈ 2.83 (= √8, one per missing dimension)
- [  ] P_k directions are orthogonal to W_E (‖P_k^T @ W_E^T‖_F < 1e-3)

### 7.2 Training Convergence

- [  ] Loss decreasing monotonically for all runs
- [  ] No NaN/Inf in any run
- [  ] All 36 jobs completed successfully

### 7.3 Layer 0 Check

- [  ] VE_sparse (layer 0) ≈ 1.0 — since layer 0 output = W_E[token_ids] + positional_emb, the sparse path through W_E should reconstruct most of the embedding component
- [  ] VE_dense (layer 0) should capture positional encoding component

### 7.4 Consistency

- [  ] VE (full) > VE_sparse for all layers and d_pca
- [  ] VE (full) > VE_dense for all layers and d_pca
- [  ] VE increases (or stays flat) with d_pca

---

## 8. Failure Modes to Watch For

| Symptom | Likely Cause | Fix |
|---|---|---|
| VE_sparse ≈ 0 everywhere | Sparse encoder not learning; λ_z too large | Reduce λ_z by 10× |
| VE ≈ VE_dense, VE_sparse ≈ 0 | Dense path absorbs everything; sparse path unused | Reduce λ_y, increase λ_z |
| VE < 0 (on test set) | Model not converged | More epochs, check lr |
| L0_z ≈ V (not sparse) | λ_z too small, ReLU not killing enough | Increase λ_z |
| L0_z ≈ 0 (too sparse) | λ_z too large | Decrease λ_z |
| ‖P_k^T @ W_E^T‖ >> 0 | pinv numerical issues | Use SVD-based projector with explicit rank cutoff |
| Training diverges | lr too high for this architecture | Reduce lr to 3e-4 |

---

## 9. Next Steps (post-experiment)

- [ ] Fill in all _____ values from `results.json` files
- [ ] Plot VE vs layer curves (one line per d_pca)
- [ ] Plot VE_sparse vs layer to confirm token path is meaningful
- [ ] If VE_sparse ≈ 0: diagnose with failure mode table above, re-run with adjusted λ_z
- [ ] Compare with sweep_decompose results
- [ ] If results are good: run CE recovery evaluation (`eval_loss_recovered.py` adapted for DualPathSAE)
- [ ] If results are good: tune λ_z and λ_y with finer sweep
- [ ] Write up findings for paper Section 4.5
