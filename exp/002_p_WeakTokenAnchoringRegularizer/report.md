# Experiment 002: Weak Token-Anchoring Regularizer — Does It Improve Decoder-Vocab Alignment?

## 实验目的

测试弱 token-anchoring regularizer 能否在不崩坏重构质量的前提下，让 plain SAE 的 decoder features 更好地对齐 GPT-2 vocab embeddings。

基于 Exp 001 结论（plain SAE 几乎零自然对齐），本实验显式加弱约束，观察 alignment / reconstruction trade-off。

## 实验设置

| 参数 | 值 |
|---|---|
| 黑盒模型 | GPT-2 (768-dim) |
| 激活层 | `transformer.h.6`, `transformer.h.11`（两层） |
| 数据集 | OpenWebText2 预提取激活 |
| SAE 架构 | `SAEModel` (HuggingFace), Linear encoder |
| dim_sparse | 50257 (= vocab size，decoder 不绑定 embedding) |
| Sparsity | TopK, k=8 |
| tied_decoder | **False** |
| nonneg_latents | True |
| use_lowrank | False |
| Epochs | 20 |
| LR | 1e-3 |
| Optimizer | Adam |

### 正则项

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_{\text{anchor}} \mathcal{L}_{\text{anchor}}$$

其中：

$$\mathcal{L}_{\text{anchor}} = -\frac{1}{m}\sum_{i=1}^{m} \max_j \cos(d_i, e_j)$$

对每个 decoder feature $d_i$，计算其与所有 token embedding $e_j$ 的 cosine similarity 的最大值，取负均值。鼓励每个 feature 靠近至少一个 token direction，但不强迫等于它。

### Sweep 参数

| `lambda_anchor` | 含义 |
|---|---|
| 0 | baseline（= plain SAE，复现 Exp 001） |
| 1e-4 | 极弱约束 |
| 1e-3 | 弱约束 |
| 1e-2 | 中等约束 |

### 训练命令

```bash
# SLURM array job: 2 layers × 4 lambda = 8 tasks
sbatch exp/002_p_WeakTokenAnchoringRegularizer/run.sh
```

## 分析方法

对每个 (layer, lambda) 组合，复用 `scripts/analyze_feature_vocab_alignment.py` 计算：

### 1. 重构质量

- MSE Loss（test set）
- LogitLens Accuracy（test set）

### 2. Max Cosine Similarity 分布统计

$$\text{max\_sim}_i = \max_j \cos(d_i, e_j)$$

报告 mean, median, std, P25, P75, P95。

### 3. 对齐类别分布

| 类别 | max_sim 阈值 | 含义 |
|---|---|---|
| 强对齐 | ≥ 0.8 | feature 几乎等价于某个 token embedding 方向 |
| 中等对齐 | [0.5, 0.8) | feature 与某些 tokens 有明显相关性 |
| 弱对齐 | [0.3, 0.5) | 存在一定关联但不强 |
| 无对齐 | < 0.3 | feature 方向与 vocab space 基本正交 |

### 4. Top-k Token 示例

对高对齐 features，查看 nearest tokens 是否语义集中、是否只吸到 function words。

---

## 实验结果

### A. 训练指标（Test Set）

| Layer | lambda_anchor | MSE Loss | LogitLens Acc |
|---|---|---|---|
| 6 | 0 | 2.1666 | 77.59% |
| 6 | 1e-4 | 2.1665 | 77.57% |
| 6 | 1e-3 | 2.1656 | 77.34% |
| 6 | 1e-2 | 2.1892 | 77.63% |
| 11 | 0 | 22.2717 | 87.76% |
| 11 | 1e-4 | 22.2750 | 87.73% |
| 11 | 1e-3 | 22.2539 | 87.30% |
| 11 | 1e-2 | 22.3221 | 87.40% |

**重构质量几乎不受影响。** Layer 6 的 MSE 波动 < 0.03（< 1.1%），LogitLens Acc 波动 < 0.3pp。Layer 11 的 MSE 波动 < 0.07（< 0.3%），Acc 波动 < 0.5pp。即使 lambda=1e-2，重构也没有崩坏。

### B. Cosine Similarity 分布统计

| Layer | lambda_anchor | Mean | Median | Std | P25 | P75 | P95 |
|---|---|---|---|---|---|---|---|
| 6 | 0 | 0.1265 | 0.1256 | 0.0231 | 0.1116 | 0.1401 | 0.1620 |
| 6 | 1e-4 | 0.5336 | 0.5665 | 0.2629 | 0.2631 | 0.6916 | 0.9160 |
| 6 | 1e-3 | 0.7708 | 0.8114 | 0.2098 | 0.6828 | 0.9307 | 0.9994 |
| 6 | 1e-2 | 0.8307 | 0.8903 | 0.2024 | 0.7663 | 0.9830 | 0.9978 |
| 11 | 0 | 0.1293 | 0.1287 | 0.0255 | 0.1146 | 0.1429 | 0.1644 |
| 11 | 1e-4 | 0.8163 | 0.9046 | 0.2416 | 0.8927 | 0.9142 | 0.9262 |
| 11 | 1e-3 | 0.9199 | 0.9988 | 0.2054 | 0.9924 | 0.9993 | 0.9997 |
| 11 | 1e-2 | 0.8954 | 0.9959 | 0.1969 | 0.8810 | 0.9976 | 0.9986 |

Mean cosine similarity 从 baseline 的 ~0.13 跃升至 lambda=1e-3 时的 0.77（layer 6）/ 0.92（layer 11）。**分布发生了数量级的右移。**

### C. 对齐度分布

| Layer | lambda_anchor | 强对齐 (≥0.8) | 中等对齐 [0.5, 0.8) | 弱对齐 [0.3, 0.5) | 无对齐 (<0.3) |
|---|---|---|---|---|---|
| 6 | 0 | 0 (0.0%) | 1 (0.0%) | 26 (0.1%) | 50,230 (99.9%) |
| 6 | 1e-4 | 10,673 (21.2%) | 21,861 (43.5%) | 3,426 (6.8%) | 14,297 (28.5%) |
| 6 | 1e-3 | 26,390 (52.5%) | 18,768 (37.3%) | 2,402 (4.8%) | 2,697 (5.4%) |
| 6 | 1e-2 | 34,831 (69.3%) | 11,606 (23.1%) | 1,407 (2.8%) | 2,413 (4.8%) |
| 11 | 0 | 0 (0.0%) | 14 (0.0%) | 72 (0.1%) | 50,171 (99.8%) |
| 11 | 1e-4 | 44,024 (87.6%) | 60 (0.1%) | 293 (0.6%) | 5,880 (11.7%) |
| 11 | 1e-3 | 44,086 (87.7%) | 1,851 (3.7%) | 2,512 (5.0%) | 1,808 (3.6%) |
| 11 | 1e-2 | 38,727 (77.1%) | 8,160 (16.2%) | 2,199 (4.4%) | 1,171 (2.3%) |

Layer 11 对 anchoring 的响应极为敏感：仅 lambda=1e-4 就将强对齐 features 从 0 推到 87.6%。Layer 6 的响应更渐进，从 21.2%（1e-4）到 69.3%（1e-2）。

### D. Top-k Token 示例

#### Layer 11, lambda=1e-3（强对齐 87.7%）

| Feature ID | max_sim | Top Tokens (cos sim) |
|---|---|---|
| 16657 | 0.9999 | 'Interstitial' (0.9999), 'stitial' (0.5341) |
| 35774 | 0.9999 | 'soDeliveryDate' (0.9999), 'isSpecialOrderable' (0.5233) |
| 29712 | 0.9999 | 'yip' (0.9999) |
| 46002 | 0.9999 | 'enegger' (0.9999), ' Schwarzenegger' (0.5178) |
| 39249 | 0.9999 | 'BuyableInstoreAndOnline' (0.9999) |

#### Layer 6, lambda=1e-3（强对齐 52.5%）

| Feature ID | max_sim | Top Tokens (cos sim) |
|---|---|---|
| 29712 | 0.9999 | 'yip' (0.9999) |
| 29434 | 0.9999 | '覚醒' (0.9999), ' サーティ' (0.5446) |
| 29953 | 0.9999 | 'ulhu' (0.9999), ' Cthulhu' (0.5829) |
| 9386 | 0.9999 | 'ovember' (0.9999) |
| 19655 | 0.9999 | 'glers' (0.9999) |

**高对齐 features 并非只吸到 function words**，而是分散在各种 content tokens（包括稀有 subwords、专有名词片段、多语种 token）。每个 feature 与其 top-1 token 的 cosine 极高（~1.0），与 top-2 的差距很大（~0.5），说明形成了 token-specific 对齐，而非吸到某个共同方向。

### E. 直方图

各组直方图保存在 `/scratch/b5bq/pu22650.b5bq/VASAE_out/002_anchor/layer_*/analysis/max_sim_histogram.png`。

---

## 结论判断标准

| 结果 | 判断 | 后续行动 |
|---|---|---|
| max cosine 显著右移 + reconstruction 基本不崩 | 弱 anchoring 有效 | 继续探索更精细的 regularizer 设计 |
| 一加 anchor loss 重构就崩 | token embedding geometry 不适合约束 SAE feature | 停止这条线 |
| cosine 上去但高对齐 features 全是 function words | anchoring 没带来可解释性增益 | 停止这条线 |

## 结论

**弱 token-anchoring regularizer 效果极为显著，且重构质量几乎不受影响。**

### 核心发现

1. **Trade-off 非常有利**：即使最强的 lambda=1e-2，MSE 增幅 < 1.1%（layer 6）/ < 0.3%（layer 11），LogitLens Acc 波动 < 0.5pp。代价极小。

2. **对齐提升是数量级的**：
   - Layer 11：baseline 强对齐 0% → lambda=1e-4 即达 87.6%，mean cosine 从 0.13 → 0.82
   - Layer 6：baseline 强对齐 0% → lambda=1e-3 达 52.5%，lambda=1e-2 达 69.3%

3. **对齐是 token-specific 的，不是 function word collapse**：高对齐 features 分散在 content tokens（Interstitial, Schwarzenegger, Cthulhu, 覚醒 等），top-1 与 top-2 的 cosine gap 很大（~0.5），说明每个 feature 确实锚定到了特定 token 方向。

4. **Layer 11 对 anchoring 的响应远强于 Layer 6**：可能因为 layer 11 的激活本身就更接近 unembedding space，decoder features 更容易被引导向 vocab directions。

### 判定

**属于"max cosine 显著右移 + reconstruction 基本不崩"**。弱 anchoring 这条线可以继续。

### 后续方向

- 推荐 lambda 范围：layer 11 用 1e-4 ~ 1e-3，layer 6 用 1e-3 ~ 1e-2
- 需进一步验证：高对齐 features 的激活模式是否真的可解释（需要上下文级分析，不只看 nearest token）
- 可以探索：是否能在 tied decoder 的基础上加 anchor loss 作为 fine-tuning signal
