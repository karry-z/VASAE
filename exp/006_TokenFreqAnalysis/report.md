# Experiment 006: Token Frequency / Rarity Analysis

## 实验目的

检查高对齐 feature 是否偏向稀有 token。对比 plain SAE（001 baseline）post-hoc matching 与 anchor SAE（002）的 aligned token 频率分布。

## 实验设置

| 参数 | 值 |
|---|---|
| Token 频率来源 | data_info.json display_text |
| 对比配置 | 001 baseline (lambda=0) vs 002 (lambda=1e-4, 1e-3, 1e-2) |
| 层 | Layer 6, Layer 11 |
| kNN isolation | k=10, embedding space cosine distance |

## 分析方法

1. 统计训练数据中每个 token 的出现频率
2. 对每个 feature 的 top-1 aligned token，查其频率 rank
3. 画直方图对比 plain vs anchor 的频率分布
4. Token isolation 分析：aligned token 在 embedding 空间的 k-NN 距离

> **Note**: 频率分析基于 alignment_results.json 中保存的 example features（每类别 5 个样本），样本量有限（n=5/类），适合看趋势但不宜过度解读精确数值。

## 实验结果

### A. Layer 6

| Config | n_high_align | Strong % | Median Freq Rank | Mean Freq Rank | % in Top-1000 | % in Bottom-10000 |
|---|---|---|---|---|---|---|
| plain (lambda=0) | 1 | 0.0% | N/A | N/A | N/A | N/A |
| anchor (1e-4) | 32,534 | 21.2% | 37,463 | 33,595 | 0% | 40% |
| anchor (1e-3) | 45,158 | 52.5% | 18,981 | 20,153 | 20% | 20% |
| anchor (1e-2) | 46,437 | 69.3% | 18,981 | 20,153 | 20% | 20% |

### B. Layer 11

| Config | n_high_align | Strong % | Median Freq Rank | Mean Freq Rank | % in Top-1000 | % in Bottom-10000 |
|---|---|---|---|---|---|---|
| plain (lambda=0) | 14 | 0.0% | N/A | N/A | N/A | N/A |
| anchor (1e-4) | 44,084 | 87.6% | 30,565 | 30,441 | 0% | 20% |
| anchor (1e-3) | 45,937 | 87.7% | 18,929 | 16,923 | 0% | 0% |
| anchor (1e-2) | 46,887 | 77.1% | 18,929 | 16,523 | 0% | 0% |

### C. Isolation Analysis

| 指标 | 值 |
|---|---|
| kNN k | 10 |
| Mean kNN cosine similarity (全部 token) | 0.5575 |
| Median kNN cosine similarity (全部 token) | 0.5506 |

Embedding 空间中 token 的 k-NN 距离适中（mean cosine ~0.56），说明 token 向量之间既不过度聚集也不完全孤立，支持 1-to-1 feature-token 对齐的可行性。

## 结论

1. **高对齐 feature 偏向中低频 token**。在弱 anchoring (lambda=1e-4) 下，sampled strong features 的 median frequency rank 约 30k-37k（总 50,257 个 token），即位于频率排名的中下段。这与 002 report 中高对齐 feature 锚定到稀有 subword（'yip', '覚醒', 'ulhu' 等）的观察一致。

2. **更强 anchoring 使分布向中频移动**。lambda 从 1e-4 增到 1e-3 后，median rank 从 ~37k 降至 ~19k（Layer 6）和从 ~31k 降至 ~19k（Layer 11）。这说明更强的 anchor 约束开始覆盖更广泛的频率范围，而不只是吸到最稀有的 token。

3. **Plain SAE 无有效对齐**，无法进行频率分析（strong features = 0）。

4. **Embedding 空间 isolation 适中** (mean kNN cosine ~0.56)。Token 向量并不过度聚集，这解释了为什么 anchor loss 可以有效地将 decoder features 逐个对齐到不同的 token 方向——如果 token 向量都挤在一起，max cosine 对齐就会退化为粗粒度的方向选择。

5. **风险评估**：虽然当前数据显示 anchor 并非只吸到最稀有 token，但 lambda=1e-4 时仍有 40%（Layer 6）/ 20%（Layer 11）的 sampled features 对齐到 bottom-10000 token。随 lambda 增大这一比例降低，但仍需关注稀有 token 吸附是否影响对高频 token 的覆盖率。
