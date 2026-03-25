---
title: Ablation Study on VASAE-Soft
date:
---

# 目的

对 VASAE-Soft 的关键设计选择进行消融，验证各超参数的敏感性和合理性。

# 方法

## 实验一：Anchor 配置消融（λ + mode）

### 1a. Anchor 系数 $\lambda$ 消融

预实验 Exp 007 在 GPT-2 L6/L11（tied decoder, k=8）上发现 $\lambda \le 3\text{e-}4$ 几乎是 free lunch，$1\text{e-}3$ 是重构开始退化的拐点。本实验在正式配置（untied decoder, k=32）下验证该结论，并扩展到 Llama。

固定 anchor mode = hard，$k = 32$，sweep：

$$\lambda \in \{0,\ 1\text{e-}5,\ 1\text{e-}4,\ 5\text{e-}4,\ 1\text{e-}3,\ 5\text{e-}3\}$$

代表层：GPT-2 L0/5/11，Llama L0/15/31。共 6λ × 6 层 = **36 tasks**。

### 1b. Anchor 模式消融

Exp 008 在 GPT-2 L6/L11 上证明 hard max 对齐效果远优于 logsumexp/softmax（L11: 87.6% vs 4.4% strong alignment），但未报告 VE 和 CE Recovery。本实验在正式配置下验证。

固定 $\lambda = 1\text{e-}4$（001_F 所用值），$k = 32$，比较三种 mode：

| 模式 | 公式 |
| -- | -- |
| hard | $\max_j \cos(d_i, e_j)$ |
| logsumexp | $\text{logsumexp}(\text{top-}k_j \cos(d_i, e_j))$，$k=10$ |
| softmax | $\sum_j w_j \cdot \cos(d_i, e_j)$，$w = \text{softmax}(\text{top-}k \cos)$，$k=10$ |

其中 hard@λ=1e-4 已在 1a 中包含，新增 2 mode × 6 层 = **12 tasks**。

## 实验二：稀疏度帕累托曲线

固定 $\lambda = 1\text{e-}4$（hard mode），sweep $k$：

$$k \in \{8,\ 16,\ 32,\ 64,\ 128\}$$

由于 TopK 前有 ReLU（nonneg_latents），当正值不足 $k$ 个时实际 L0 < $k$。

其中 $k = 32$ 已在实验一中包含，新增 4k × 6 层 = **24 tasks**。

帕累托曲线以实际 L0 为横轴，VE / CE Recovery 为纵轴，展示 VASAE-Soft 的稀疏度-质量 trade-off，并观察不同 $k$ 下 dead feature rate 的变化。

## 实验三：Anchor 计算频率消融（仅 Llama）

Llama 词表 128256，每 batch 计算 anchor loss 代价过高（~8.5s/batch vs 正常 ~0.2s/batch）。001_F 中引入 `anchor_every` 每 N 步计算一次。需验证降低频率是否影响性能。

$$\texttt{anchor\_every} \in \{1,\ 10,\ 50,\ 100,\ 500\}$$

固定 $\lambda = 1\text{e-}4$，hard mode，$k = 32$。仅 Llama L0/15/31。

共 5 × 3 = **15 tasks**（其中部分与实验一重叠）。

## 执行顺序

1. **先跑实验三**：确认 anchor_every 的安全值，决定后续 Llama 实验使用的频率参数
2. **再并行跑实验一 + 二**

## 共享配置

与 001_F_Benchmarking 的 VASAE-Soft 一致：

- dim_sparse = vocab_size（GPT-2: 50257, Llama: 128256）
- decoder: 独立可学习（untied）
- encoder: linear, TopK sparsity, nonneg latents
- Adam (lr=1e-3), max 20 epochs, early stopping (patience=3)
- WikiText-103, max_length=128
- GPT-2: float32, batch_size=32, train/eval/test = 50000/10000/5000
- Llama: bfloat16, batch_size=8, train/eval/test = 20000/2000/5000
- 与 001_F_Benchmarking 配置完全一致，重叠配置点（Exp1a λ=1e-4, Exp3 anchor_every=50）直接复用 001_F 结果

## 评估指标

- **VE**（Variance Explained）：归一化重构质量
- **CE Recovery**：功能性重构质量
- **Dead Feature Rate**：测试集上激活次数为 0 的 feature 占比（实验一、二报告）
- **L0**：每个输入的平均非零激活 feature 数，TopK + ReLU 下 L0 ≤ $k$（实验二报告）

# 流程

```bash
# 0. 实验三：Anchor 计算频率消融（先跑，确认 anchor_every 安全值）
sbatch exp/001A_F_AblationSoft/run_frequency_llama.sh

# 1. 实验一 + 二：Anchor 配置 + 稀疏度帕累托（并行）
sbatch exp/001A_F_AblationSoft/run_gpt2.sh
sbatch exp/001A_F_AblationSoft/run_llama.sh

# 2. 汇总结果
uv run python scripts/collect_ablation_results.py \
    --results-dir /scratch/b5bq/pu22650.b5bq/VASAE_out/001A_F_AblationSoft \
    --output-dir exp/001A_F_AblationSoft
```

# 结果

## 实验一：Anchor 配置消融

### 1a. Anchor 系数 $\lambda$

#### GPT-2

| layer | $\lambda$ | VE | CE Recovery | Dead Rate |
| -- | -- | -- | -- | -- |

#### Llama-3.1-8B

| layer | $\lambda$ | VE | CE Recovery | Dead Rate |
| -- | -- | -- | -- | -- |

### 1b. Anchor 模式

#### GPT-2

| layer | mode | VE | CE Recovery | Dead Rate |
| -- | -- | -- | -- | -- |

#### Llama-3.1-8B

| layer | mode | VE | CE Recovery | Dead Rate |
| -- | -- | -- | -- | -- |

## 实验二：稀疏度帕累托曲线

### GPT-2

| layer | $k$ | L0 | VE | CE Recovery | Dead Rate |
| -- | -- | -- | -- | -- | -- |

### Llama-3.1-8B

| layer | $k$ | L0 | VE | CE Recovery | Dead Rate |
| -- | -- | -- | -- | -- | -- |

帕累托曲线图：以实际 L0 为横轴，VE（或 CE Recovery）为纵轴，每层一条线。

## 实验三：Anchor 计算频率消融 (Llama-3.1-8B)

| layer | anchor_every | VE | CE Recovery | wall time / epoch |
| -- | -- | -- | -- | -- |

## 分析

待结果完成后补充。
