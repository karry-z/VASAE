# Experiment 004: Context-based Interpretability Analysis

## 实验目的

对 002 的高对齐 feature，提取 top activating contexts，验证语义相关性。统计 feature usage 和 dead feature 比例。验证高对齐是否等于真正的可解释性提升。

## 实验设置

| 参数 | 值 |
|---|---|
| 模型来源 | Exp 002 训练的 SAE |
| 配置 | 2 layers (6, 11) x 4 lambda (0, 1e-4, 1e-3, 1e-2) |
| 跟踪 features | Top-200 高对齐 features |
| Top contexts | 每个 feature 的 top-20 激活位置 |
| Context window | 前后各 10 token |

## 分析方法

1. 加载 002 训练好的 SAE + alignment_results.json
2. 遍历数据集，forward encoder 得到 sparse activations
3. 对 top-200 高对齐 features，记录 top-20 激活位置的上下文
4. 统计 dead features、激活频率、激活强度分布
5. Encoder-decoder consistency 检查

## 实验结果

### A. Feature Usage Statistics

| Layer | lambda | Dead Features | Dead % | Alive Features | Mean Act Freq | Median Act Freq |
|---|---|---|---|---|---|---|
| 6 | 0 | 45,717 | 90.97% | 4,540 | 1.59e-4 | 0.0 |
| 6 | 1e-4 | 45,639 | 90.81% | 4,618 | 1.59e-4 | 0.0 |
| 6 | 1e-3 | 45,548 | 90.63% | 4,709 | 1.59e-4 | 0.0 |
| 6 | 1e-2 | 45,616 | 90.77% | 4,641 | 1.59e-4 | 0.0 |
| 11 | 0 | 49,532 | 98.56% | 725 | 1.59e-4 | 0.0 |
| 11 | 1e-4 | 49,610 | 98.71% | 647 | 1.59e-4 | 0.0 |
| 11 | 1e-3 | 49,686 | 98.86% | 571 | 1.59e-4 | 0.0 |
| 11 | 1e-2 | 49,676 | 98.84% | 581 | 1.59e-4 | 0.0 |

Activation frequency histogram (layer 6 vs 11, lambda=0):
- Layer 6: 3,822 rare (<0.1%), 557 low, 153 medium, 8 high (>=10%)
- Layer 11: 592 rare, 39 low, 76 medium, 18 high (>=10%)

Layer 11 has far more dead features but its alive features are more frequently active (18 high-freq vs 8).

### B. Consistency Analysis (Tracked Top-200 Features)

| Layer | lambda | Tracked | With Activations | Dead |
|---|---|---|---|---|
| 6 | 0 | 200 | 199 | 1 |
| 6 | 1e-4 | 200 | 29 | 171 |
| 6 | 1e-3 | 200 | 31 | 169 |
| 6 | 1e-2 | 200 | 29 | 171 |
| 11 | 0 | 200 | 193 | 7 |
| 11 | 1e-4 | 200 | 1 | 199 |
| 11 | 1e-3 | 200 | 0 | 200 |
| 11 | 1e-2 | 200 | 0 | 200 |

Key finding: Top-200 high-alignment features from 002 are overwhelmingly dead in anchored models (lambda>0). Only baseline (lambda=0) keeps most tracked features alive. This means the features that scored highest on vocab alignment in exp 002 are almost entirely inactive when anchoring is applied -- anchoring shifts which features are used, not just how aligned they are.

### C. Context Interpretability Examples

#### Layer 6, Lambda 0 (baseline) -- high activation, low interpretability

Features have very high activation magnitudes (max ~1300) but poor token-context alignment. The top features fire on generic/unrelated text:

- **Feature 3003** (aligned token: 'where'), max_act=1381.39, n_activations=24,168
  - Top context: "Games and Symbols of Hate..." (no 'where' present)
  - 2nd context: "NSA Leaker Edward Snowden was supposed to be on Aer..." (no 'where' present)
- **Feature 28138** (aligned token: ' adjustable'), max_act=1377.86, n_activations=27,709
  - Fires on the exact same top contexts as Feature 3003 -- suggests polysemantic/entangled representations
- **Feature 5357** (aligned token: 'and'), max_act=96.15, n_activations=77,248
  - Very frequent (6% of positions). Context does contain 'and' in some cases but the feature is too broad to be interpretable.

Only 6 of 200 tracked features had their aligned token appear in any top-5 context.

#### Layer 6, Lambda 1e-4 (weak anchoring) -- sparse, low-magnitude activations

Activation magnitudes drop dramatically (max ~6 vs ~1300). Only 29 of 200 tracked features active.

- **Feature 25941** (token: 'thren'), max_act=6.38, n_activations=2
  - Context: "...Speaker Paul Ryan Paul Davis Ryan Kenosha will be a good bell..." -- no obvious 'thren' connection
- **Feature 40786** (token: 'Setup'), max_act=5.38, n_activations=3
  - Context: "But in 2014, he became the world's longest-serving death row inmate..." -- no 'Setup' relevance
- **Feature 4683** (token: ' existing'), max_act=4.72, n_activations=1
  - Context: "Update: Creed's Scott Stapp Is Not the New Singer for Stone Temple Pil..." -- no 'existing' match

Zero token-aligned features found among the 29 active ones.

#### Layer 6, Lambda 1e-3 (moderate anchoring)

Similar pattern: 31 active features, very low magnitudes (max ~6.8).

- **Feature 316** (token: 'et'), max_act=5.68, n_activations=3 -- 'et' substring found in "Kucinich" context
- **Feature 33098** (token: ' Crosby'), max_act=5.43, n_activations=2 -- no 'Crosby' in context

1 of 200 tracked features had token-in-context match (the short substring 'et').

#### Layer 11, Lambda 0 (baseline) -- broad high-frequency features

- **Feature 14545** (token: ' GA'), max_act=693.96, n_activations=21,076
  - Context: URLs and links -- fires on punctuation/special characters
- **Feature 46036** (token: 'こ'), max_act=621.50, n_activations=45,896
  - Extremely frequent (36% of positions). Fires on numbers/percentages. No Japanese text present.
- **Feature 38345** (token: 'immune'), max_act=612.78, n_activations=1,195
  - Context: "Department of Transportation. Statewide, 1,295..." -- fires on numbers, not immune-related text

7 of 200 had token-in-context matches, but mostly coincidental (short substrings).

#### Layer 11, Lambda > 0 (anchored) -- near-total feature death

- **Lambda 1e-4**: Only 1 tracked feature active (Feature 50184 '!!!!!', max_act=15.32, n=1)
- **Lambda 1e-3**: 0 tracked features active
- **Lambda 1e-2**: 0 tracked features active

The top-200 high-alignment features are completely dead at layer 11 when any anchoring is applied.

## 结论

1. **Dead feature rates are uniformly high** (~91% at layer 6, ~99% at layer 11) regardless of anchoring lambda. Anchoring does not significantly change the total number of alive features (~4500-4700 at L6, ~570-725 at L11).

2. **Anchoring kills the specific high-alignment features**. The top-200 features that were identified as highly aligned to vocab tokens in exp 002 become overwhelmingly dead when anchoring is applied (171/200 dead at L6, 199-200/200 dead at L11). This suggests anchoring forces the model to reorganize which features are active, rather than simply improving alignment of existing features.

3. **Context interpretability is poor across all conditions**. Even in the baseline (lambda=0), the top features show:
   - Polysemantic behavior (multiple unrelated features fire on the same contexts)
   - Very high magnitudes but no semantic correspondence to aligned tokens
   - Only 3-3.5% of tracked features show even substring matches between aligned token and activation context

4. **Activation magnitude scales inversely with anchoring strength**. Baseline max activations are ~1300 (L6) and ~700 (L11), dropping to ~5-6 (L6) and ~15 (L11) with anchoring. This dramatic magnitude reduction suggests the anchoring constraint fundamentally changes the encoding scheme.

5. **Implication for VASAE**: High cosine similarity between decoder columns and embedding vectors (as measured in exp 002) does not translate to interpretable feature-token correspondences. The "alignment" is geometric but not semantic -- features aligned to token X do not preferentially activate on text containing X. A different approach to interpretability evaluation is needed.
