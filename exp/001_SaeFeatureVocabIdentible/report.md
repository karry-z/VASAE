# Experiment 001: Do Plain SAE Decoder Features Naturally Align with Vocab Embeddings?

## 实验目的

验证在无任何 vocab 约束条件下，普通 SAE 学出的 decoder features 是否自然对齐 GPT-2 的 token embedding 空间。

## 实验设置

| 参数 | 值 |
|---|---|
| 黑盒模型 | GPT-2 (768-dim) |
| 激活层 | `transformer.h.0` ~ `transformer.h.11`（全部 12 层） |
| 数据集 | OpenWebText2 预提取激活 |
| SAE 架构 | `SAEModel` (HuggingFace), Linear encoder |
| dim_sparse | 50257 (= vocab size，但 decoder 不绑定 embedding) |
| Sparsity | TopK, k=8 |
| tied_decoder | **False** |
| nonneg_latents | True |
| use_lowrank | False |
| Epochs | 20 |
| LR | 1e-3 |
| Optimizer | Adam |

### 训练命令

```bash
# SLURM array job, 每层一个 task
sbatch exp/001_p_SaeFeatureVocabIdentible/run.sh
```

## 分析方法

### 1. Cosine Similarity 计算

对每个 decoder feature $d_i$（decoder 权重矩阵的第 $i$ 列，shape `(768,)`），计算其与所有 vocab embeddings $e_j$ 的 cosine similarity：

$$s_{ij} = \cos(d_i, e_j) = \frac{d_i \cdot e_j}{\|d_i\| \|e_j\|}$$

取每个 feature 的最大 cosine similarity 作为其"对齐度"：

$$\text{max\_sim}_i = \max_j s_{ij}$$

### 2. Token Matching

对每个 feature $d_i$，找到 cosine similarity 最高的 top-k tokens，观察：
- nearest tokens 是否语义集中
- 不同 feature 是否覆盖不同的词汇区域

### 3. 分类标准

| 类别 | max_sim 阈值 | 含义 |
|---|---|---|
| 强对齐 | ≥ 0.8 | feature 几乎等价于某个 token embedding 方向 |
| 中等对齐 | [0.5, 0.8) | feature 与某些 tokens 有明显相关性 |
| 弱对齐 | [0.3, 0.5) | 存在一定关联但不强 |
| 无对齐 | < 0.3 | feature 方向与 vocab space 基本正交 |

### 4. 分析脚本

```bash
python scripts/analyze_feature_vocab_alignment.py \
    --model-path /scratch/b5bq/pu22650.b5bq/VASAE_out/001_plain_sae/layer_11/sae.pth \
    --blackbox-model-dir /scratch/b5bq/pu22650.b5bq/VASAE_out/BlackBoxModels/gpt2 \
    --output-dir /scratch/b5bq/pu22650.b5bq/VASAE_out/001_plain_sae/layer_11/analysis \
    --top-k 10
```

---

## 实验结果

### A. 训练指标（Test Set）

| Layer | MSE Loss | LogitLens Acc |
|---|---|---|
| 0 | 0.4506 | 81.53% |
| 1 | 0.5220 | 81.80% |
| 2 | 0.6985 | 82.30% |
| 3 | 0.9783 | 80.87% |
| 4 | 1.2767 | 80.42% |
| 5 | 1.6540 | 78.48% |
| 6 | 2.1666 | 77.59% |
| 7 | 2.9490 | 75.94% |
| 8 | 4.2233 | 74.77% |
| 9 | 6.5488 | 73.88% |
| 10 | 11.5045 | 76.74% |
| 11 | 22.2717 | 87.76% |

MSE 随层数递增显著增大（浅层激活范数小，深层大）。LogitLens accuracy 在最后一层最高（87.76%），说明 layer 11 重建质量在语义层面最好。

### B. Cosine Similarity 分布统计（所有层）

| Layer | Mean | Median | Std | Min | Max | P25 | P75 | P95 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.1311 | 0.1304 | 0.0220 | 0.0348 | 0.3559 | 0.1167 | 0.1444 | 0.1658 |
| 1 | 0.1305 | 0.1297 | 0.0216 | 0.0391 | 0.3613 | 0.1161 | 0.1438 | 0.1655 |
| 2 | 0.1294 | 0.1287 | 0.0215 | 0.0409 | 0.4736 | 0.1151 | 0.1428 | 0.1642 |
| 3 | 0.1296 | 0.1288 | 0.0215 | 0.0378 | 0.4639 | 0.1154 | 0.1430 | 0.1642 |
| 4 | 0.1286 | 0.1278 | 0.0217 | 0.0523 | 0.4714 | 0.1142 | 0.1420 | 0.1633 |
| 5 | 0.1273 | 0.1265 | 0.0224 | 0.0468 | 0.4679 | 0.1126 | 0.1408 | 0.1629 |
| 6 | 0.1265 | 0.1256 | 0.0231 | 0.0324 | 0.5064 | 0.1116 | 0.1401 | 0.1620 |
| 7 | 0.1264 | 0.1252 | 0.0249 | 0.0211 | 0.5326 | 0.1107 | 0.1402 | 0.1643 |
| 8 | 0.1249 | 0.1230 | 0.0271 | -0.0129 | 0.5606 | 0.1084 | 0.1383 | 0.1646 |
| 9 | 0.1256 | 0.1235 | 0.0305 | -0.0879 | 0.7032 | 0.1076 | 0.1398 | 0.1689 |
| 10 | 0.1287 | 0.1275 | 0.0277 | -0.0853 | 0.6396 | 0.1127 | 0.1426 | 0.1668 |
| 11 | 0.1293 | 0.1287 | 0.0255 | -0.1648 | 0.7227 | 0.1146 | 0.1429 | 0.1644 |

### C. 对齐度分布

| Layer | 强对齐 (≥0.8) | 中等对齐 [0.5, 0.8) | 弱对齐 [0.3, 0.5) | 无对齐 (<0.3) |
|---|---|---|---|---|
| 0 | 0 (0.0%) | 0 (0.0%) | 16 (0.0%) | 50241 (100.0%) |
| 1 | 0 (0.0%) | 0 (0.0%) | 5 (0.0%) | 50252 (100.0%) |
| 2 | 0 (0.0%) | 0 (0.0%) | 7 (0.0%) | 50250 (100.0%) |
| 3 | 0 (0.0%) | 0 (0.0%) | 9 (0.0%) | 50248 (100.0%) |
| 4 | 0 (0.0%) | 0 (0.0%) | 11 (0.0%) | 50246 (100.0%) |
| 5 | 0 (0.0%) | 0 (0.0%) | 14 (0.0%) | 50243 (100.0%) |
| 6 | 0 (0.0%) | 1 (0.0%) | 26 (0.1%) | 50230 (99.9%) |
| 7 | 0 (0.0%) | 1 (0.0%) | 50 (0.1%) | 50206 (99.9%) |
| 8 | 0 (0.0%) | 2 (0.0%) | 86 (0.2%) | 50169 (99.8%) |
| 9 | 0 (0.0%) | 7 (0.0%) | 145 (0.3%) | 50105 (99.7%) |
| 10 | 0 (0.0%) | 16 (0.0%) | 93 (0.2%) | 50148 (99.8%) |
| 11 | 0 (0.0%) | 14 (0.0%) | 72 (0.1%) | 50171 (99.8%) |

**所有层中，强对齐 feature 数量为零，中等对齐不超过 16 个（0.03%），99.7%+ 的 features 完全无对齐。**

### D. Top-k Token 示例（Layer 11）

#### 中等对齐 Features

| Feature ID | max_sim | Top-5 Tokens (cos sim) |
|---|---|---|
| 43290 | 0.7227 | ' of' (0.723), ' the' (0.453), ' in' (0.420), ' a' (0.405), ' and' (0.390) |
| 12667 | 0.6996 | ' to' (0.700), ' the' (0.447), ' for' (0.423), ' a' (0.415), ' in' (0.409) |
| 31143 | 0.6271 | ' the' (0.627), ' a' (0.473), ' and' (0.390), ' that' (0.368), ' in' (0.363) |

这些极少数"中等对齐" feature 对应的都是超高频 function words（of, to, the），top tokens 之间 cosine 差距不大，说明这些 feature 实际是在编码高频 token 的共同方向（"function word subspace"），并非 token-specific 对齐。

#### 弱对齐 Features

| Feature ID | max_sim | Top-5 Tokens (cos sim) |
|---|---|---|
| 31264 | 0.4907 | ' the' (0.491), ' a' (0.443), ' "' (0.426), ' in' (0.425), ' and' (0.417) |
| 27252 | 0.4786 | '\n' (0.479), ' and' (0.407), ',' (0.383), ' in' (0.381), ' the' (0.380) |
| 18169 | 0.4723 | ' with' (0.472), ' With' (0.343), 'with' (0.310), 'With' (0.292), ' WITH' (0.287) |

弱对齐 features 同样以 function words 和标点为主，top tokens 之间的 sim 差异很小，不构成"单 token 对齐"。

#### 无对齐 Features（绝大多数）

| Feature ID | max_sim | Top-5 Tokens (cos sim) |
|---|---|---|
| 13393 | 0.2987 | ' in' (0.299), ' a' (0.262), ' the' (0.260), ' and' (0.232), ' with' (0.232) |
| 38345 | 0.2935 | ' the' (0.293), ' a' (0.274), '000' (0.266), ' and' (0.223), ' in' (0.222) |
| 48083 | 0.2912 | ' York' (0.291), ' the' (0.245), ' and' (0.240), ',' (0.237), '-' (0.234) |

### E. 直方图

各层直方图已保存在 `/scratch/b5bq/pu22650.b5bq/VASAE_out/001_plain_sae/layer_*/analysis/max_sim_histogram.png`。

分布形态：所有层的 max_sim 分布均呈尖锐的单峰分布，峰值集中在 0.12-0.13 附近，右尾极短，几乎没有 feature 超过 0.3。

---

## 结论判断标准

| 结果 | 判断 | 后续行动 |
|---|---|---|
| 强对齐 ≥ 30% | 自然对齐显著 | Vocab-tied decoder 有较强先验支撑，值得进一步训练验证 |
| 中等+强对齐 ≥ 50% | 存在明显对齐趋势 | 尝试 soft alignment loss 或 tied decoder，看能否提升 |
| 总体对齐（≥0.3）< 30% | 基本无自然对齐 | vocab-tying 作为归纳偏置的理论基础不足，需要重新审视 VASAE 方向 |

## 结论

**普通 SAE 学出的 decoder features 与 GPT-2 vocab embeddings 之间几乎不存在自然对齐。**

- 全部 12 层中，强对齐（≥0.8）feature 数量为 **零**。
- 即使在对齐最好的 layer 9/10/11，中等+弱对齐 features 也仅占 **0.1%-0.3%**，且这些 features 对应的是高频 function words 的共同方向，不是 token-specific 的对齐。
- 50257 个 features 中，**99.7%+ 的 max cosine similarity 低于 0.3**，与随机方向在 768 维空间中的 expected cosine similarity（约 $1/\sqrt{768} \approx 0.036$）相比只高出约 3 倍，说明 decoder features 所编码的方向与 vocab embedding 空间基本正交。

**判定：总体对齐（≥0.3）远低于 30%，属于"基本无自然对齐"。**

这一结果说明，在无约束条件下，SAE 倾向于学习与 vocab embedding 正交的方向来最优重建激活。vocab-tying 作为归纳偏置并非自然涌现的性质，而是一种外部强加的结构约束。后续 VASAE 的价值论证需要从"对齐是否提升可解释性或下游性能"的角度出发，而非"SAE 本身就会对齐 vocab"。
