# Experiment 011: Feature Input-Output Decomposition & Token Alignment Diagnostic

## 实验目的

通过四组平行实验，系统性地回答一个核心问题：**Token-level alignment 到底能不能 work？**

010 的 sweep 表明 anchor coeff 在 [1e-5, 1e-3] 范围内几乎无差异，004 的报告显示只有 3.5% 的高对齐 feature 的激活上下文包含对齐 token。011 从四个角度量化这个 gap，排除或确认 token alignment 的可行性。

## 实验设计

四组独立实验并行运行：

| 子实验 | 目的 | 方法 | 规模 |
|---|---|---|---|
| A. I/O Decomposition | 量化 feature 的 input/output/geometric 三视角一致性 | 对 010 (k=32, a=1e-4) 模型，计算 t_geo/t_input/t_causal | L2, L6, L11 |
| B. Logit Attribution Sparsity | 检查 W_U @ d_i 是 peaked 还是 diffuse | 纯权重分析，不需要数据 | L0-L11 全部 |
| C. 009 Tied-Decoder Eval | 评估 tied decoder (dim_sparse=50257) 的重建质量 | 对 009 模型补 eval | 12 layers × 2 anchors |
| D. Anchor Strength Sweep | 测试极端 anchor coeff (0, 0.01, 0.1, 1.0) | 训练新 SAE | L2, L6, L11 × 4 anchors |

## 子实验 A：I/O Decomposition

### 方法

对每个 SAE feature i，从三个（四个）视角确定"它代表什么 token"：

1. **t_geo(i)**：`argmax_t cos(d_i, e_t)` — decoder 列与 embedding 的余弦相似度（几何对齐）
2. **t_logit(i)**：`argmax_t (W_U @ d_i)_t` — logit attribution 最大的 token（输出效果）
3. **t_input(i)**：在 500 个 wikitext 样本上统计，哪个 token 出现时该 feature 激活值之和最高
4. **t_causal(i)**：ablate 该 feature 后，哪个 token 的预测概率变化最大（因果效果）

注：GPT-2 使用 weight tying (W_E = W_U)，因此 t_geo 和 t_logit 虽然计算方式不同（cosine vs raw dot product），但高度相关。

### 结果

#### 一致性矩阵

| 指标 | L2 | L6 | L11 |
|---|---|---|---|
| alive features | 1128/6144 (18.4%) | 1600/6144 (26.0%) | 504/6144 (8.2%) |
| geo = logit (top-1) | 67.3% | 67.2% | 87.1% |
| **geo = input (top-1)** | **1.2%** | **0.7%** | **2.0%** |
| **geo = causal (top-1)** | **0.1%** | **1.9%** | **10.5%** |
| logit = input (top-1) | 0.4% | 0.4% | 1.8% |
| **input = causal (top-1)** | **0.0%** | **0.0%** | **0.0%** |
| all four agree | 0.0% | 0.0% | 0.0% |
| geo-input top-5 overlap | 0.02 | 0.01 | 0.02 |
| geo-causal top-5 overlap | 0.02 | 0.15 | 0.61 |

#### Feature 类别分布

基于一致性将 alive features 分为五类：

| 类别 | 定义 | L2 | L6 | L11 |
|---|---|---|---|---|
| **Token feature** | geo=input=causal | 0 (0%) | 0 (0%) | 1 (0.2%) |
| **Output feature** | geo=causal ≠ input | 1 (0.1%) | 30 (1.9%) | 53 (10.5%) |
| **Input feature** | geo=input ≠ causal | 0 (0%) | 0 (0%) | 0 (0%) |
| **Context feature** | geo ≠ input ≠ causal | 432 (38.3%) | 488 (30.5%) | 293 (58.1%) |
| **Unaligned** | geo max_sim < 0.3 | 695 (61.6%) | 1082 (67.6%) | 157 (31.2%) |

#### 典型案例

**Output features（L11，geo = causal ≠ input）— 唯一"部分 work"的类别：**

| Feature | geo_sim | geo token | input token | causal token |
|---|---|---|---|---|
| F4734 | 0.998 | ' nearly' | ' such' | ' nearly' |
| F3006 | 0.998 | ' Pale' | ' St' | ' Pale' |
| F1317 | 0.998 | 'iction' | ' to' | 'iction' |
| F887 | 0.997 | 'Esc' | ' the' | 'Esc' |

这些 feature 的几何对齐确实反映了它们的因果效果（ablate 后影响最大的 token），但它们的激活条件（input token）完全不同。说明它们是"推动预测某 token"的功能单元，但不是"检测某 token"的感知单元。

**典型不一致案例（L2）：**

| Feature | geo_sim | geo token | input token | causal token |
|---|---|---|---|---|
| F3409 | 0.999 | 'click' | ' just' | N/A |
| F6055 | 0.998 | ' NH' | 's' | N/A |
| F4056 | 0.998 | ' unm' | '@' | 'senal' |

三个视角指向完全不相关的 token。

### 解读

1. **geo = input ≈ 1%**：几何对齐与输入语义完全无关。一个 feature 的 decoder 列跟 token "cache" 高度对齐，不代表它会在包含 "cache" 的文本中激活。
2. **geo = causal 随层递增（0.1% → 10.5%）**：越靠近 unembedding 层，decoder 方向越能预测因果效果。这符合预期——L11 的输出直接被 W_U 读取。
3. **input = causal = 0%**：最惊人的发现。激活某 feature 的输入 token 和该 feature 影响的输出 token **完全不是同一个**。Feature 不是"token detector"也不是"token promoter"，而是 context-dependent 的计算单元。
4. **0 个 token feature**：没有任何 feature 在三个视角上一致。Token-level 的 feature 解释不成立。

---

## 子实验 B：Logit Attribution Sparsity

### 方法

对 010 (k=32, a=1e-4) 的全 12 层模型，计算每个 feature 的 logit attribution 向量 `la_i = d_i @ W_U^T`（shape: vocab_size），然后度量其稀疏性：

- **Entropy**：`H(softmax(la_i))`，均匀分布时 H_max = ln(50257) = 10.82
- **Top-1 concentration**：softmax(la_i) 最大值占总概率的比例
- **Top-5 concentration**：softmax(la_i) top-5 值之和
- **Max/Mean ratio**：|la_i| 的最大值与均值之比

### 结果

| Layer | Entropy | Ent/Max | Max/Mean | Top-1% | Top-5% | Geo Sim | Corr(Ent,Geo) |
|---|---|---|---|---|---|---|---|
| L0 | 10.82 | 0.999 | 4.5 | 0.008 | 0.023 | 0.691 | -0.622 |
| L1 | 10.82 | 0.999 | 4.5 | 0.008 | 0.023 | 0.649 | -0.583 |
| L2 | 10.82 | 0.999 | 4.6 | 0.008 | 0.023 | 0.626 | -0.597 |
| L3 | 10.82 | 0.999 | 4.7 | 0.008 | 0.022 | 0.579 | -0.611 |
| L4 | 10.82 | 0.999 | 4.7 | 0.007 | 0.022 | 0.552 | -0.622 |
| L5 | 10.82 | 0.999 | 4.8 | 0.007 | 0.021 | 0.514 | -0.642 |
| L6 | 10.82 | 1.000 | 4.8 | 0.007 | 0.021 | 0.491 | -0.644 |
| L7 | 10.82 | 1.000 | 4.8 | 0.007 | 0.021 | 0.474 | -0.645 |
| L8 | 10.82 | 0.999 | 4.8 | 0.007 | 0.021 | 0.477 | -0.629 |
| L9 | 10.82 | 0.999 | 4.7 | 0.007 | 0.022 | 0.521 | -0.595 |
| L10 | 10.82 | 0.999 | 4.4 | 0.009 | 0.024 | 0.715 | -0.524 |
| L11 | 10.82 | 0.999 | 4.3 | 0.010 | 0.026 | 0.867 | -0.532 |

### 解读

1. **Entropy/Max = 0.999 everywhere**：softmax(W_U @ d_i) 在所有层上都几乎等于均匀分布。这意味着**每个 feature 的 logit contribution 分散在全部 50257 个 token 上**，无法被解释为"推动某个 token"。

2. **Top-1 concentration = 0.008%**：即使 decoder 列跟某个 token embedding 有 0.99 的 cosine similarity，它通过 W_U 投射到 logit 空间后仍然是 diffuse 的。

3. **Correlation = -0.6**：geo alignment 越高的 feature，entropy 越低（稍微更 peaked），但绝对值仍在 10.81-10.82 范围内，几乎没有实际意义。

4. **根本原因**：W_E (= W_U) 的行向量之间有大量共线性。即使 d_i 跟 e_t 完全平行，d_i 也同时跟许多其他 e_j 有非零投影，导致 d_i @ W_U^T 不 peaked。Token embedding 空间中的方向不具有 vocab-level 的稀疏选择性。

---

## 子实验 C：009 Tied-Decoder Evaluation

### 方法

009 训练的 SAE 使用 tied decoder：decoder 固定为 W_E^T（GPT-2 的 token embedding 转置），dim_sparse = 50257（vocab size），k=32。对其补充 evaluation，计算 VE、loss recovered 等指标。

### 结果

| Layer | VE | MSE | Loss Recovered | CE(sae) |
|---|---|---|---|---|
| L0 | -21.67 | 38.2 | -0.67 | 11.13 |
| L1 | -16.77 | 102.0 | -1.15 | 10.55 |
| L2 | -11.37 | 837.4 | 0.14 | 4.34 |
| L3 | -9.67 | 829.3 | 0.37 | 4.24 |
| L4 | -9.18 | 879.7 | -0.05 | 4.42 |
| L5 | -8.98 | 922.6 | -0.52 | 4.61 |
| L6 | -9.15 | 980.2 | -0.34 | 4.51 |
| L7 | -8.99 | 998.5 | -0.23 | 4.46 |
| L8 | -7.98 | 925.8 | 0.06 | 4.38 |
| L9 | -8.52 | 1021.6 | 0.10 | 4.19 |
| L10 | -12.73 | 1553.0 | -0.46 | 4.43 |
| L11 | -22.52 | 1295.5 | -14.35 | 6.34 |

### 解读

1. **VE 全部为大负数**：重建后的激活比用全零还差。用 50257 个 token embedding 做字典，选 32 个线性组合，完全无法逼近原始激活。

2. **对比 010**：010 (free decoder, dim_sparse=6144, k=32) 在相同层上 VE > 0.95（中间层）。说明 **decoder 必须自由学习，而不能绑定到 token embeddings**。

3. **L0 的 MSE=38 最低，但 VE 最差之一**：因为 L0 的激活方差本身就低，MSE 不大但相对方差极高。

4. **结论**：用 token embedding 做固定字典的 VASAE 方案（tied decoder）是不可行的。

---

## 子实验 D：Anchor Strength Sweep

### 设置

| 参数 | 值 |
|---|---|
| anchor_coeff | 0 (baseline), 0.01, 0.1, 1.0 |
| layers | 2, 6, 11 |
| k | 32 |
| decoder | free (not tied) |
| 其他 | 同 010 |

### 结果

| Layer | Anchor | Loss Recovered | VE | LogitLens Acc | CE(sae) |
|---|---|---|---|---|---|
| L2 | 0 | **0.887** | 0.989 | 0.842 | 4.015 |
| L2 | 0.01 | 0.874 | 0.988 | 0.833 | 4.019 |
| L2 | 0.1 | 0.871 | 0.988 | 0.835 | 4.021 |
| L2 | 1.0 | 0.857 | 0.988 | 0.823 | 4.026 |
| L6 | 0 | **0.897** | 0.978 | 0.807 | 4.010 |
| L6 | 0.01 | 0.894 | 0.977 | 0.807 | 4.012 |
| L6 | 0.1 | 0.896 | 0.977 | 0.808 | 4.011 |
| L6 | 1.0 | 0.890 | 0.977 | 0.802 | 4.014 |
| L11 | 0 | **0.484** | 0.507 | 0.879 | 4.057 |
| L11 | 0.01 | 0.459 | 0.497 | 0.888 | 4.061 |
| L11 | 0.1 | 0.470 | 0.498 | 0.889 | 4.059 |
| L11 | 1.0 | 0.466 | 0.501 | 0.891 | 4.060 |

### 解读

1. **anchor=0（无对齐）始终是最优或接近最优**：强推 alignment 不会提升性能，只会略微降低。

2. **anchor=1.0 vs anchor=0 的差距很小**（L2: 0.887→0.857, L6: 0.897→0.890）：即使把 anchor loss 的系数提高 1000 倍（从 1e-3 到 1.0），对重建质量的影响仍然有限。说明 anchor loss 没有实质性地改变 decoder 的学习方向。

3. **L11 的 LogitLens Acc 随 anchor 增大反而提升**（0.879→0.891）：这看似矛盾，但原因是 anchor 强制 decoder 列更接近 embedding，而 L11 层的激活本就接近 embedding 空间，因此 logit lens（通过 W_U 投影）的 top-1 准确率稍高。但 loss_recovered 并没有提升。

---

## 综合结论

### Token-level alignment 不可行的三重证据

| 层面 | 证据 | 子实验 |
|---|---|---|
| **代数层面** | softmax(W_U @ d_i) 的 entropy = 0.999 × H_max，任何 decoder 方向对 logit 的贡献都是 near-uniform 的 | B |
| **功能层面** | geo vs input 一致性 ≈ 1%，geo vs causal ≈ 0-10%，input vs causal = 0%。没有 feature 同时是输入检测器和输出推动器 | A |
| **重建层面** | Tied decoder (dim_sparse=vocab, k=32) 的 VE 全部为大负数，无法用 token embeddings 稀疏重建 activations | C |

### 补充发现

1. **anchor loss 效果微弱**：即使系数从 0 提高到 1.0（跨 3 个数量级），对 decoder alignment 和重建质量的影响都很小。Anchor loss 没有实质性地改变 decoder 的学习方向。（子实验 D）

2. **Layer 11 是唯一的例外**：geo = causal 达到 10.5%（其他层 < 2%），top-5 overlap 达到 0.61。53 个 output feature 的几何对齐确实反映了因果效果。这合理——L11 的输出直接被 W_U 读取，因此 decoder 方向在 unembedding 空间中有部分功能意义。但即使 L11 也没有 token feature（三视角一致的 feature 仅 1 个）。（子实验 A）

3. **Feature 是计算单元而非 token 标签**：每个 feature 的激活条件（什么输入触发它）和效果（它影响什么输出）指向完全不同的 token。Feature 编码的是 context-dependent 的转换规则（如"在某种上下文中，推动对某 token 的预测"），而非单个 token 的语义。

### 对 VASAE 方向的影响

Token-level alignment（无论是 tied decoder 还是 anchor loss）不是正确的可解释性策略。可能的替代方向：

1. **放弃单 token 对齐**，转向理解 feature 的 (input context → output effect) 映射
2. **Cluster-level alignment**：对齐到 token cluster 而非单个 token
3. **Functional interpretability**：直接用 feature 的因果效果定义其含义，而非几何相似度
4. **Layer-specific 策略**：L11 的 output feature 模式表明最后一层的 decoder 方向有部分输出语义，可以只在此层尝试对齐

## 文件清单

```
exp/011_p_IODecomposition/
├── report.md                          # 本报告
├── run.sh                             # IO decomposition SLURM script
├── eval_009.sh                        # 009 eval SLURM script
├── run_anchor_sweep.sh                # Anchor sweep SLURM script
├── L2_k32/
│   ├── io_decomposition_results.json  # 完整结果
│   └── io_tensors.pt                  # 原始 tensor 数据
├── L6_k32/
│   └── ...
├── L11_k32/
│   └── ...
├── logit_attr/
│   ├── logit_attribution_stats.json   # Logit attribution sparsity 结果
│   ├── logit_attribution_tensors.pt   # 原始 tensor
│   └── entropy_distribution_by_layer.png  # 可视化
└── logs/                              # SLURM 日志
```

**关键脚本：**
- `scripts/analyze_feature_io_decomposition.py` — IO decomposition 分析
- `scripts/analyze_logit_attribution_sparsity.py` — Logit attribution sparsity 分析
- `scripts/eval_sae_online.py` — SAE 在线评估
