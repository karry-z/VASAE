# Experiment 005: Random Dictionary Control

## 实验目的

验证 anchor loss 的效果是否依赖真实 vocab embedding 的几何结构。将 W_E 替换为随机字典（shuffle / gaussian），重跑训练，如果随机字典也能产生同样的对齐效果，说明 anchor loss 只是"几何贴脸"而非利用了 vocab 的语义结构。

## 实验设置

| 参数 | 值 |
|---|---|
| 黑盒模型 | GPT-2 (768-dim) |
| 激活层 | `transformer.h.6`, `transformer.h.11` |
| SAE 架构 | 同 002 |
| Lambda | 1e-4, 1e-3 |
| Random anchor 类型 | shuffle (行排列), gaussian (同 norm 随机方向) |

### Random Anchor 定义

- **shuffle**: 对 W_E 的行做随机排列，保持 embedding 的个体向量不变但打乱 token-方向对应关系
- **gaussian**: 生成同维度随机向量，归一化到与 W_E 各行相同的 norm

### 分析

每个 task 训练后做两次 alignment 分析：
1. vs random dict（检查是否对齐到随机字典）
2. vs real W_E（检查是否也对齐到真实 vocab）

## 实验结果

### A. Alignment vs Random Dict

| Layer | lambda | Random Type | Mean max_sim (vs random) | Strong % (vs random) |
|---|---|---|---|---|
| 6 | 1e-4 | shuffle | 0.5500 | 25.80% |
| 6 | 1e-4 | gaussian | 0.5492 | 21.51% |
| 6 | 1e-3 | shuffle | 0.7980 | 58.04% |
| 6 | 1e-3 | gaussian | 0.7966 | 58.33% |
| 11 | 1e-4 | shuffle | 0.8271 | 88.70% |
| 11 | 1e-4 | gaussian | 0.8229 | 86.84% |
| 11 | 1e-3 | shuffle | 0.9305 | 88.24% |
| 11 | 1e-3 | gaussian | 0.9297 | 87.55% |

### B. Alignment vs Real W_E

| Layer | lambda | Random Type | Mean max_sim (vs real) | Strong % (vs real) |
|---|---|---|---|---|
| 6 | 1e-4 | shuffle | 0.5500 | 25.80% |
| 6 | 1e-4 | gaussian | 0.1265 | 0.00% |
| 6 | 1e-3 | shuffle | 0.7980 | 58.04% |
| 6 | 1e-3 | gaussian | 0.1271 | 0.00% |
| 11 | 1e-4 | shuffle | 0.8271 | 88.70% |
| 11 | 1e-4 | gaussian | 0.1294 | 0.00% |
| 11 | 1e-3 | shuffle | 0.9305 | 88.24% |
| 11 | 1e-3 | gaussian | 0.1296 | 0.00% |

> **Note**: shuffle 类型 vs_random 和 vs_real 结果完全一致。这是因为 shuffle 只是对 W_E 行做排列，analysis_vs_real 实际比较的是同一组向量（只是 token 对应关系不同），cosine similarity 的 max 操作会找到相同的最大值。

### C. Reconstruction Quality

| Layer | lambda | Random Type | Test MSE Loss | Test LogitLens Acc |
|---|---|---|---|---|
| 6 | 1e-4 | shuffle | 2.181 | 0.7752 |
| 6 | 1e-4 | gaussian | 2.163 | 0.7742 |
| 6 | 1e-3 | shuffle | 2.180 | 0.7758 |
| 6 | 1e-3 | gaussian | 2.166 | 0.7764 |
| 11 | 1e-4 | shuffle | 22.237 | 0.8760 |
| 11 | 1e-4 | gaussian | 22.280 | 0.8776 |
| 11 | 1e-3 | shuffle | 22.162 | 0.8752 |
| 11 | 1e-3 | gaussian | 22.267 | 0.8784 |

## 结论

1. **Anchor loss 确实能将 decoder 对齐到任意目标字典** -- 无论是 shuffle 还是 gaussian 随机字典，vs_random 的 alignment 都很高（尤其 lambda=1e-3 和 layer 11），说明 anchor loss 的对齐机制是纯几何效应，不依赖 vocab 的语义结构。

2. **Gaussian 随机字典不会对齐到真实 W_E** -- gaussian 条件下 vs_real 的 mean max_sim 仅约 0.13（接近随机基线），strong% = 0%。这确认了 anchor loss 不是在"发现"隐藏的 vocab 结构，而是在强制对齐到给定的目标。

3. **Shuffle 条件下 vs_real = vs_random** -- 因为 shuffle 保留了 W_E 的所有行向量（仅打乱顺序），max cosine similarity 在两种字典上完全一致。这意味着 shuffle 对齐后 decoder 的每一列仍然是某个 vocab embedding 方向，只是 token 对应关系是错的。

4. **重建质量几乎不受影响** -- 所有 random anchor 条件下的 test MSE 和 LogitLens Acc 与 baseline 水平相近（layer 6: MSE ~2.17, Acc ~0.775; layer 11: MSE ~22.2, Acc ~0.876），说明 anchor loss 在引导 decoder 方向的同时不显著损害自编码能力。

5. **结论**: Anchor loss 的对齐效果是几何效应而非语义效应。它可以将 decoder 拉向任何目标字典。真正赋予 VASAE 可解释性的是目标字典本身（真实 W_E）的语义内容，而非 anchor loss 机制本身。

### 预期结论模式

- 如果 random dict 也能对齐 → anchor loss 是几何效应，不依赖 vocab 语义
- 如果 random dict 不能对齐 → vocab geometry 确实特殊，anchor loss 利用了语义结构
- 如果 shuffle 能对齐但 gaussian 不能 → 重要的是向量本身而非对应关系
