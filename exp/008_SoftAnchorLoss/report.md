# Experiment 008: Soft Anchor Loss Variant

## 实验目的

用 top-k logsumexp / softmax 替代 hard max anchor loss，减少对孤立 token 的吸附效应。

## 实验设置

| 参数 | 值 |
|---|---|
| 黑盒模型 | GPT-2 (768-dim) |
| 激活层 | `transformer.h.6`, `transformer.h.11` |
| SAE 架构 | 同 002 |
| Lambda | 1e-4, 1e-3 |
| Anchor mode | logsumexp, softmax |
| Anchor top-k | 10 |

### Soft Anchor Loss 定义

- **hard** (baseline): $\max_j \cos(d_i, e_j)$ -- 只看最近的一个 token
- **logsumexp**: $\text{logsumexp}(\text{topk}_j \cos(d_i, e_j))$ -- 平滑的 soft-max
- **softmax**: $\sum_j w_j \cdot \cos(d_i, e_j)$ where $w = \text{softmax}(\text{topk } \cos)$ -- 加权平均

## 实验结果

### A. Reconstruction Quality

| Layer | lambda | Mode | MSE Loss | LogitLens Acc |
|---|---|---|---|---|
| 6 | 1e-4 | logsumexp | 2.1675 | 77.49% |
| 6 | 1e-4 | softmax | 2.1644 | 77.50% |
| 6 | 1e-3 | logsumexp | 2.1677 | 77.45% |
| 6 | 1e-3 | softmax | 2.1678 | 77.48% |
| 11 | 1e-4 | logsumexp | 22.2608 | 87.74% |
| 11 | 1e-4 | softmax | 22.2607 | 87.73% |
| 11 | 1e-3 | logsumexp | 22.2683 | 87.13% |
| 11 | 1e-3 | softmax | 22.2675 | 87.13% |

重构质量几乎不受影响。所有 MSE 波动 < 0.01（layer 6）/ < 0.01（layer 11）。LogitLens Acc 在 lambda=1e-4 时波动 < 0.1pp，lambda=1e-3 时最大下降约 0.6pp（layer 11）。

### B. Alignment Statistics

| Layer | lambda | Mode | Mean | Median | Std | Strong (>=0.8) count | Strong % |
|---|---|---|---|---|---|---|---|
| 6 | 1e-4 | logsumexp | 0.3783 | 0.3680 | 0.2003 | 546 | 1.09% |
| 6 | 1e-4 | softmax | 0.3794 | 0.3685 | 0.2016 | 560 | 1.11% |
| 6 | 1e-3 | logsumexp | 0.6515 | 0.7169 | 0.2460 | 18,512 | 36.83% |
| 6 | 1e-3 | softmax | 0.6525 | 0.7188 | 0.2456 | 18,364 | 36.54% |
| 11 | 1e-4 | logsumexp | 0.6152 | 0.6696 | 0.1883 | 2,191 | 4.36% |
| 11 | 1e-4 | softmax | 0.6158 | 0.6705 | 0.1884 | 2,200 | 4.38% |
| 11 | 1e-3 | logsumexp | 0.8333 | 0.9062 | 0.2215 | 44,108 | 87.76% |
| 11 | 1e-3 | softmax | 0.8337 | 0.9058 | 0.2211 | 44,115 | 87.78% |

### C. Comparison with Hard Anchor (Exp 002)

#### Layer 6

| lambda | Mode | Mean | Strong % | 002 Hard Mean | 002 Hard Strong % |
|---|---|---|---|---|---|
| 1e-4 | logsumexp | 0.3783 | 1.09% | 0.5336 | 21.2% |
| 1e-4 | softmax | 0.3794 | 1.11% | 0.5336 | 21.2% |
| 1e-3 | logsumexp | 0.6515 | 36.83% | 0.7708 | 52.5% |
| 1e-3 | softmax | 0.6525 | 36.54% | 0.7708 | 52.5% |

#### Layer 11

| lambda | Mode | Mean | Strong % | 002 Hard Mean | 002 Hard Strong % |
|---|---|---|---|---|---|
| 1e-4 | logsumexp | 0.6152 | 4.36% | 0.8163 | 87.6% |
| 1e-4 | softmax | 0.6158 | 4.38% | 0.8163 | 87.6% |
| 1e-3 | logsumexp | 0.8333 | 87.76% | 0.9199 | 87.7% |
| 1e-3 | softmax | 0.8337 | 87.78% | 0.9199 | 87.7% |

#### 对比要点

1. **Soft anchor 在相同 lambda 下对齐效果弱于 hard anchor。** 在 layer 6 lambda=1e-4，hard anchor 已达 mean 0.53 / strong 21.2%，而 soft anchor 仅 mean 0.38 / strong 1.1%。Layer 11 lambda=1e-4 差距更大：hard 87.6% strong vs soft 4.4% strong。

2. **logsumexp 与 softmax 几乎没有差异。** 在所有 8 组实验中，两种 mode 的 mean、median、strong% 差别均在 0.5pp 以内。两种 smooth 方式在 top-k=10 的设定下行为几乎等价。

3. **Soft anchor 需要更大的 lambda 才能达到 hard anchor 的效果。** Layer 11 需要 lambda=1e-3 才能达到 hard anchor lambda=1e-4 的水平（87.8% vs 87.6% strong）。Layer 6 需要 lambda=1e-3 才到 36.8%，hard anchor 在 lambda=1e-3 已达 52.5%。

4. **重构质量方面两者相当。** Soft anchor 在 lambda=1e-3 的 MSE / LogitLens Acc 与 hard anchor 相似，没有明显优势。

## 结论

**Soft anchor loss（logsumexp / softmax）对齐效果系统性弱于 hard max anchor loss。**

- 在相同 lambda 下，soft 版本的 alignment 明显低于 hard 版本（尤其 lambda=1e-4 差距巨大）。
- logsumexp 和 softmax 两种 soft 变体之间没有实质区别。
- Soft anchor 并未带来重构质量的改善，即"更温和"并不等于"更好的 trade-off"。
- 若要使用 soft anchor 达到 hard anchor 的对齐水平，需将 lambda 提高约 1 个数量级，但这会带来更大的重构代价。

### 建议

- **Hard max anchor 仍是更优的 baseline。** Soft 变体没有展示出足够的优势来替代 hard max。
- 如果希望缓解 hard max 的 token collapse 问题，可以考虑其他方向（如 diverse anchor、contrastive loss），而非简单地 smooth 化 max 操作。
