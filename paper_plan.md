# The Geometry of What Sparse Autoencoders Miss
## NeurIPS Paper Plan (Revised)

---

## One-Sentence Story

> Language model residual streams have a low-dimensional, vocabulary-orthogonal substructure that encodes systematic meta-information (position, frequency, wordform); this structure is invisible to sparse dictionary methods, persists across scales and architectures, and can be explicitly separated to build better autoencoders.

---

## Core Thesis

**Sparse methods fail on residual streams not because they need more atoms, but because part of the signal lives in a low-dimensional subspace that is geometrically misaligned with the vocabulary basis.**

This is a geometric claim, not a sparsity claim. The evidence:
- PCA@8 achieves lower reconstruction error than OMP@128 (8 learned directions beat 128 vocabulary atoms)
- The misaligned subspace encodes interpretable, systematic features (position, token frequency, BPE word boundaries)
- Removing these features improves OMP reconstruction dramatically (F6+OMP@8 reduces error from ~90% to ~5%)
- Yet ~83% of the residual after feature removal remains geometrically misaligned — the structure runs deeper than known features

The question is whether this is a quirk of GPT-2 or a fundamental property of transformer residual streams.

---

## Logical Chain

```
Observation (GPT-2):
  OMP@k with vocab dictionary E performs poorly (relative error 85-95%)
  But PCA@8 massively outperforms OMP@128
  → The "missing" component is LOW-DIMENSIONAL, not high-dimensional noise

Identification (GPT-2):
  Dense residual encodes position, frequency, word-start via linear probes
  Removing these: error drops from ~90% to ~5% (F6+OMP@8)
  But geometric mismatch persists at ~83% even after feature removal
  → Known features explain FUNCTION but not GEOMETRY of the dense component

Generalization question:
  Is this a GPT-2 artifact (small model, absolute position encoding)?
  Or a structural property of transformer residual streams?
  Does position encoding type (absolute vs RoPE) change the picture?
  Does model scale change the picture?

Answer (this paper):
  Test on GPT-2 (117M, absolute PE), Llama-3-8B (RoPE), LLaVA-1.5-7B, Qwen2-Audio-7B
  → Low-dim vocab-orthogonal structure exists in ALL models
  → Content varies: absolute PE → strong position signal; RoPE → position signal absent,
     but frequency/wordform persist; multimodal → modality-specific features emerge
  → Geometric mismatch ratio is remarkably stable across architectures (~75-85%)
  → Dual-Path SAE exploits this structure: E^T @ z_sparse + V @ z_dense

Architectural contribution:
  Dual-Path SAE with EXPLICIT low-rank dense path
  → Not just "add a low-rank residual" (which is an engineering trick)
  → Principled: V is initialized from identified dense subspace directions
  → Interpretable: V columns correspond to position/frequency/modality axes
  → Closes the gap: vocab-tied + 16-dim dense path ≈ free decoder performance
```

---

## Section 3: Method

### 3.1 Sparse Decomposition and Its Geometric Failure Mode

**Content**: Given residual stream activation h in R^d and token embedding matrix E in R^{V x d} (V >> d, so span(E) = R^d), we use OMP@k to find the best k-sparse linear combination of vocabulary vectors.

**Key definitions**:
- Relative reconstruction error: ||h - h_recon||^2 / ||h||^2
- Dense residual: r = h - h_recon (note: r is NOT orthogonal to span(E), it is IN span(E) but requires dense combinations)

**Key empirical fact** (motivating the paper):
- OMP@k error decreases slowly with k (OMP@8: ~90%, OMP@64: ~80%)
- PCA@k error decreases FAST (PCA@8 < OMP@128)
- This gap is the signature of a low-dimensional, vocab-misaligned subspace

**Robustness**: Reproduce with Lasso/L1 regression across k = 4, 8, 16, 32, 64 to confirm the pattern is not OMP-specific.

**Why this matters**: Standard SAEs implicitly assume the decoder basis (whether free or vocab-tied) can capture all structure via sparse codes. The PCA >> OMP gap shows this assumption fails — there exists structure that is inherently low-rank and dense.

### 3.2 Characterizing the Dense Subspace

**Content**: Systematic protocol to identify what information the dense residual r encodes.

**Steps**:
1. Compute OMP@k residuals r for a corpus of activations
2. PCA on r → extract top-K principal components
3. Per-PC linear correlation with candidate features (Pearson r)
4. MLP probe on r as information upper bound
5. Iterative feature removal: subtract best linear fit of identified features, re-measure OMP error

**Candidate features** (applicable across all models):
- Sequence position (scalar, normalized)
- Token unigram log-frequency
- BPE word-start indicator (Ġ prefix)
- Function word indicator
- Token string length

For multimodal models, add:
- Modality label (text vs visual vs audio)
- Spatial position x, y (visual tokens only, from patch grid)
- Temporal position (audio tokens only)

This is not a novel method — it is standard probing applied to a specific, well-motivated target (OMP residuals rather than raw activations). The contribution is in WHAT we probe and WHAT we find, not the probing technique itself.

### 3.3 Measuring Geometric Mismatch

**Content**: Quantify how misaligned the activation covariance is with the vocabulary geometry, independent of any specific sparse method.

**Metrics**:
- **Subspace angle**: Principal angles between top-k eigenvectors of activation covariance C and top-k right singular vectors of E
- **Trace alignment**: tr(C @ E^T E) / (||C||_F ||E^T E||_F)
- **Condition number**: condition number of E restricted to activation principal subspace
- **Residual mismatch ratio**: After removing ALL identified features from r, what fraction of ||r||^2 remains? This measures the "unexplained geometric gap"

These metrics are computed per-layer and compared across models to test universality.

### 3.4 Dual-Path SAE

**Content**: Architecture that explicitly separates sparse vocab-aligned and dense low-rank channels.

```
h_recon = E^T @ z_sparse + V @ z_dense + b
```

- E^T in R^{d x V}: frozen text vocabulary embeddings (sparse channel)
- V in R^{d x r}: learnable low-rank basis, r = 8~32 (dense channel)
- z_sparse: k-sparse via TopK
- z_dense: dense, low-dimensional (just r numbers)
- b: learnable bias

**Key difference from vanilla low-rank residual**: V can be initialized from the identified dense subspace directions (Section 3.2), making the architecture a direct operationalization of the geometric finding. After training, V's columns remain interpretable and can be compared with probing results — closing the loop.

**Training**: MSE + sparsity penalty on z_sparse. No explicit regularization on z_dense (it's already low-dimensional).

---

## Section 4: Experiments

### 4.1 Setup

**Models**:
| Model | Type | Params | Position Encoding | Why |
|-------|------|--------|-------------------|-----|
| GPT-2 | Text | 117M | Absolute (learned) | Baseline, full pre-experiment results |
| Llama-3-8B | Text | 8B | RoPE | Scale + PE type variation |
| LLaVA-1.5-7B | Vision-Language | 7B | RoPE | Non-text input modality |
| Qwen2-Audio-7B | Audio-Language | 7B | RoPE | Second non-text modality |

**Data**: OpenWebText (text), COCO Captions (VLM), AudioCaps (ALM). ~50K token activations per model, 6 equally-spaced layers each.

**Key design choice**: ALL models are analyzed using E_text (text vocabulary) as the sparse dictionary. For multimodal models, this means visual/audio token positions will naturally have higher OMP error — this is expected and acknowledged, not presented as a finding. The interesting question is what structure the dense residual contains beyond this trivial modality effect.

### 4.2 Exp 1: Is the Low-Rank Misaligned Structure Universal?

**Question**: Does the PCA >> OMP gap exist beyond GPT-2?

**Method**:
- For each model, each layer: compute OMP@k error (k = 4, 8, 16, 32, 64) and PCA@k error (k = 4, 8, 16, 32)
- Plot reconstruction error curves: if PCA@k consistently beats OMP@(many * k), the low-rank misaligned structure exists
- Compute geometric mismatch metrics (Section 3.3) per model per layer

**Expected outcomes**:
- PCA >> OMP gap exists in all 4 models (main claim)
- Gap magnitude may vary with model scale and architecture
- RoPE models may show different subspace orientation (position signal should be weaker or absent since position is encoded in attention, not in residual stream directly)

**Key figure**: 4-panel plot (one per model) of OMP@k vs PCA@k reconstruction error across layers. The persistent gap IS the visual proof of the dual structure.

### 4.3 Exp 2: What Does the Dense Subspace Encode?

**Question**: Across models, what features live in the vocab-orthogonal subspace?

**Method**: Apply probing protocol (Section 3.2) to OMP residuals in each model.

**Sub-questions by model**:

*GPT-2 (reproduce + validate)*: Confirm position/frequency/word-start. This is the reference point.

*Llama-3-8B (scale + PE effect)*:
- Does position signal persist? RoPE encodes position in attention, not directly in residual stream → hypothesis: position signal in dense subspace is WEAKER
- Do frequency and word-start persist? These are independent of PE type → hypothesis: YES
- What replaces position if it's weaker? Possible: layer-norm scale, attention pattern statistics

*LLaVA-1.5-7B (visual modality)*:
- For TEXT tokens: similar to Llama-3 (same base model)
- For VISUAL tokens: probe for spatial position (x, y from patch grid), patch region type
- Across modalities: probe modality label in dense subspace → is modality identity encoded here?
- CRITICAL: separate trivial modality effect (visual tokens ≠ text tokens, so of course OMP with E_text is worse) from non-trivial structure (spatial layout encoded in dense subspace even after controlling for modality)

*Qwen2-Audio-7B*:
- For AUDIO tokens: probe temporal position, energy, pitch
- Same structure as VLM analysis

**Feature removal cascade** (per model):
- F0: baseline OMP@8 error
- +position → +frequency → +word_start → +modality_features → F_all
- Measure error after each removal
- Report residual mismatch ratio after removing all identified features

**Key figure**: Table showing (model × feature) contribution to dense subspace variance. Highlights universal features (frequency, word-start) vs architecture-specific (position) vs modality-specific (spatial layout, temporal position).

### 4.4 Exp 3: Functional Importance of the Dense Subspace

**Question**: Is the dense subspace functionally important, or just a geometric artifact?

**Method**: Ablation study — zero out top-K PCs of OMP residual, measure impact on downstream behavior.

**Metrics**:
- Logit lens accuracy (does ablated activation still predict the correct next token?)
- KL divergence between original and ablated logit distributions
- For VLM: VQA accuracy on ablated activations (if tractable)

**Comparisons**:
- Ablate r_dense PCs vs ablate activation PCs (same number of dimensions) → if r_dense PCs have outsized impact relative to their variance share, the dense subspace is functionally privileged
- Ablate at shallow vs mid vs deep layers → which layers' dense subspace matters most?

**Expected**: Deep-layer dense subspace ablation causes large KL divergence (analogous to GPT-2 L11 PC1 finding). This confirms the dense subspace is not noise — it carries information the model needs for prediction.

### 4.5 Exp 4: Dual-Path SAE

**Question**: Can we build a better SAE by explicitly modeling the dual structure?

**Variants** (trained on Llama-3-8B mid-layer AND LLaVA-1.5-7B mid-layer):
1. **Vanilla SAE**: free decoder (upper bound on reconstruction)
2. **Vocab-tied SAE**: decoder = E_text^T (interpretable but constrained)
3. **Dual-Path SAE (random init)**: E_text^T + V (r=16), V randomly initialized
4. **Dual-Path SAE (informed init)**: same, but V initialized from top PCs of OMP residual

**Metrics**:
- Reconstruction MSE
- Logit lens accuracy
- Sparsity of z_sparse (L0 norm)
- Interpretability of V: cosine similarity between V columns and probing-identified directions

**Key comparisons**:
- (2) vs (1): the gap is what vocab-tying loses → quantifies the dense channel's contribution
- (3) vs (2): the gain from adding 16 dense dimensions → does a tiny dense path recover most of the gap?
- (4) vs (3): does informed initialization help? → validates the probing analysis
- V columns after training: do they converge to interpretable directions regardless of initialization? → strongest evidence that the dual structure is real

**Key figure**: Bar chart showing MSE for 4 variants. The gap between (2) and (1) should be largely closed by (3), proving that vocab-tying fails specifically because of the low-rank dense structure, and that 16 dimensions suffice to fix it.

---

## Section 5: Discussion

### 5.1 The Dual Structure is Geometric, Not Modality-Specific

Summary table across 4 models: PCA>>OMP gap size, mismatch ratio, identified features. The geometric mismatch ratio is remarkably stable (~75-85%) even though feature content varies by architecture and modality. This suggests the dual structure is a property of how transformers learn representations, not of what they represent.

### 5.2 Implications for Sparse Autoencoders

Current SAEs assume all meaningful structure is sparse. Our finding shows ~15-20% of residual stream energy is inherently low-rank and dense. This isn't a failure of sparsity — it's a different KIND of information (meta-information about position, frequency, modality) that is naturally dense. SAE research should explicitly account for this rather than forcing it through sparse codes.

### 5.3 What We Don't Know About the 83%

Even after removing all identified features, ~83% of the geometric mismatch remains. This is the biggest open question. Possibilities:
- Nonlinear encoding of known features (but MLP probes only recover modestly more)
- Features we haven't thought to probe for
- Intrinsic geometric properties of the transformer computation (layer-norm-induced structure, attention pattern residuals)

This is an honest limitation, not a weakness — it defines the frontier for future work.

### 5.4 Limitations

- Analysis is linear (OMP, PCA, linear probes). Nonlinear structure may exist but is harder to characterize.
- OMP is one sparse coding method. Results should be validated with pursuit algorithms, Lasso, and learned dictionaries.
- Only 4 models. Claims of universality are preliminary.
- Multimodal analysis uses text vocabulary only. A fairer analysis would require defining sparse dictionaries for visual/audio modalities, which is an open problem.
- Activation collection scale (~50K tokens) may miss rare phenomena.

---

## Section 1: Introduction

1. **Hook**: Sparse Autoencoders are the dominant tool for LLM interpretability, built on the assumption that neural activations decompose into sparse features. How well does this assumption hold?

2. **Observation**: We show that in GPT-2, PCA with just 8 directions achieves lower reconstruction error than OMP with 128 vocabulary atoms. This means a significant part of the residual stream lives in a low-dimensional subspace that is geometrically misaligned with the token vocabulary — invisible to any sparse method using vocabulary-based dictionaries.

3. **Identification**: This subspace is not noise. It encodes systematic meta-information: sequence position, token frequency, BPE word boundaries. Removing these features dramatically improves sparse reconstruction. Yet a large geometric gap (~83%) persists beyond known features.

4. **Generalization**: We test whether this "dual structure" — sparse vocabulary-aligned content plus dense vocabulary-orthogonal meta-information — is universal. Across GPT-2 (117M), Llama-3-8B, LLaVA-1.5-7B, and Qwen2-Audio-7B, we find: (a) the PCA >> OMP gap exists in every model, (b) the content of the dense subspace varies by architecture and modality, but (c) the geometric mismatch ratio is remarkably stable.

5. **Application**: We propose Dual-Path SAE, which adds a small learnable low-rank basis alongside the vocabulary-tied decoder. With just 16 dense dimensions, it recovers most of the performance lost by vocabulary tying — validating that the dual structure is real and exploitable.

**Contributions**:
1. Identify the PCA >> OMP gap as a signature of low-dimensional vocab-orthogonal structure in transformer residual streams
2. Characterize this structure's content across 4 models spanning different scales, architectures, and modalities
3. Show the dense subspace is functionally important through targeted ablations
4. Propose and validate Dual-Path SAE as a principled architecture exploiting this structure

---

## Section 2: Related Work

- **Sparse Autoencoders**: Bricken et al., Templeton et al., Gao et al. — assume sparse decomposition suffices; we show it doesn't for ~15-20% of the signal
- **Residual stream geometry**: Elhage et al. (mathematical framework), logit lens (Nostalgebraist), residual stream as communication channel
- **Dictionary learning**: OMP, Lasso, sparse coding theory — we use these as analysis tools, not as the contribution itself
- **Probing representations**: Standard technique; our application to OMP residuals (rather than raw activations) is the novelty
- **Multimodal representations**: Cross-modal alignment studies, visual/audio token processing in VLMs/ALMs

---

## Key Figures

1. **Figure 1 (Hero figure)**: PCA@k vs OMP@k reconstruction error curves for GPT-2 mid-layer. The gap between the two curves IS the paper's core phenomenon. Simple, visual, immediately compelling.

2. **Figure 2**: Same plot for all 4 models (2x2 grid), confirming universality.

3. **Figure 3**: Feature removal cascade — bar chart showing OMP error after progressively removing position/frequency/word-start. Dramatic drop, but large residual remains.

4. **Figure 4**: Cross-model comparison table — dense subspace features, mismatch ratio, PCA>>OMP gap size. Universal features highlighted.

5. **Figure 5**: Ablation impact — KL divergence when ablating dense subspace PCs vs random PCs of same dimensionality. Dense PCs have outsized impact.

6. **Figure 6**: Dual-Path SAE reconstruction MSE — 4 variants, showing the dense path closes the vocab-tying gap.

---

## What Changed from the Original Plan and Why

### Removed

1. **"Modality translation pipeline" narrative**: Reframed. The observation that visual tokens become more text-aligned in deeper layers is expected (the model is trained to output text). Presenting this as a discovery would invite immediate rejection. Instead, we acknowledge it as expected and focus on what's in the dense residual BEYOND this trivial effect.

2. **Cross-layer evolution heatmap as core figure**: Replaced by PCA vs OMP gap as the hero figure. The heatmap shows a trivial phenomenon; the PCA>>OMP gap shows a surprising one.

3. **"Dense Channel Probing Protocol" as a named method**: It's standard probing applied to a specific target. Naming it suggests false novelty. Instead, we describe it plainly.

4. **Elaborate multi-step probing (Step 1-4)**: Simplified. The method is standard; the contribution is the finding, not the technique.

5. **Cross-modal binding experiment**: Too speculative, insufficient evidence to support strong claims.

### Restructured

1. **Core thesis shifted**: From "dual-channel structure is universal across modalities" (an observation) to "PCA >> OMP gap reveals geometric misalignment in ALL transformer residual streams" (a geometric claim with clear evidence criteria).

2. **Multimodal role downgraded**: From the central narrative to one axis of variation in a generalization study. The paper's core finding (PCA>>OMP gap, interpretable dense subspace) stands on text-only models. Multimodal models provide additional evidence and show how the dense subspace content varies.

3. **Dual-Path SAE motivation clarified**: Not "we discovered dual channels, so let's build a dual-path SAE" (which sounds post-hoc). Instead: "vocab-tied SAEs lose performance precisely because they can't represent the low-rank dense structure; adding 16 learnable dimensions fixes this."

4. **Exp 4 expanded to 2 models**: Training on both Llama-3-8B and LLaVA-1.5-7B to show the architecture works across settings.

### Added

1. **Honest framing of the 83% residual**: Elevated from a bullet point to a discussion subsection. This is the most intriguing finding and the clearest invitation for future work.

2. **Explicit acknowledgment of tautologies**: The plan now explicitly states that visual tokens having high OMP error with text vocab is expected, not a finding. This preempts the obvious reviewer objection.

3. **Lasso/L1 robustness check**: To prevent "this is just an OMP artifact" criticism.

4. **Clear evidence criteria**: Each experiment has a stated question and what would constitute a positive/negative answer, rather than "expected story lines" that assume the answer.
