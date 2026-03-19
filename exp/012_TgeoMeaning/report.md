# Experiment 012: t_geo 意义诊断 + 全层 Feature 功能分析

## 实验目标

011 证明 token-level alignment 不 work（t_geo != t_input, t_geo != t_causal），但 t_geo 不可能是随机的。本实验回答两个问题：
1. **t_geo 到底反映了什么？** 是 embedding 几何 artifact，还是 decoder 学到的真实结构？
2. **不同层的 feature 功能有何差异？** 扩展 011 的 3 层分析到全部 12 层。

## 实验结构

| 子实验 | 内容 | 层数 | 数据需求 |
|--------|------|------|----------|
| 012a | Weight-only: margin + null baseline + hub + 层间 Jaccard | L0-L11 | 无 |
| 012b | Data-dependent: mean activation direction + context position | L2/L6/L11 | 500 samples |
| 012c | Full IO decomposition（扩展 011） | L0-L11 | 500+100 samples |
| 012d | 汇总可视化 + 假设判定 | — | — |
| 012e | **按 token 类型条件分析** (content/function/punct/subword) | 全层 | CPU only，复用已有 tensor |

---

## 核心发现：按 Token 类型分层后 t_geo 的含义清晰化

012a-d 的整体指标（t_geo==t_mean=1-3%）掩盖了关键信号。按 token 类型分层后，四类 feature 呈现完全不同的行为模式。

### Token 类型分布

GPT-2 vocab (50257 tokens) 分为四类：

| 类型 | 数量 | 占比 | 示例 |
|------|------|------|------|
| content_word | 31766 | 63.2% | " quantum", " played", " China" |
| subword_fragment | 15132 | 30.1% | "tion", "ing", "ment" |
| punctuation | 3019 | 6.0% | ",", ".", "!", "\n" |
| function_word | 340 | 0.7% | " the", " of", " and", " is" |

每层 ~6144 features 的 t_geo 落入这四类的比例稳定：content 61%, subword 31%, punct 5%, function 2.5%。

### 关键发现 1: Function word features 是真正的 "token features"

**012b 按类型分层后的 mean activation direction：**

| Layer | 类型 | cos(d,mu) | t_geo==t_mean | pos0 匹配 | pos+1 匹配 |
|-------|------|-----------|---------------|-----------|------------|
| L2  | **function_word** | **0.292** | **22.4%** | **2.03%** | **2.84%** |
| L2  | content_word | 0.048 | 0.0% | 0.12% | 0.01% |
| L2  | punctuation | 0.216 | 0.0% | 3.70% | 0.11% |
| L2  | subword_fragment | 0.062 | 0.0% | 0.30% | 0.01% |
| L6  | **function_word** | **0.318** | **14.0%** | **1.58%** | **3.17%** |
| L6  | content_word | 0.062 | 0.0% | 0.23% | 0.07% |
| L11 | **function_word** | **0.293** | **34.9%** | **2.22%** | **4.96%** |
| L11 | content_word | -0.079 | 0.0% | 0.04% | 0.00% |

Function word features 表现出 011 预期但没找到的模式：
- **cos(d_i, mu_i) = 0.29-0.32**：decoder 方向确实近似激活质心方向
- **t_geo == t_mean 达到 14-35%**：几何最近邻确实 = 激活均值最近邻，远超整体的 1-3%
- **pos+1 匹配率最高（L11=4.96%）**：这些 feature 的 t_geo 倾向于是 next-token
- L11 最强：35% t_geo==t_mean，暗示最后层的 function word feature decoder 编码了"预测下一个功能词"的方向

**对比之下，content word features 的 cos(d,mu)≈0, t_geo==t_mean=0.0%——完全不同的机制。**

### 关键发现 2: Content word features 是 "causal/output features"

**012c 按类型分层后的 IO 一致性：**

| Layer | 类型 | geo=input | geo=causal |
|-------|------|-----------|------------|
| L0  | content_word | 2.6% | 0.0% |
| L0  | **function_word** | **13.6%** | N/A |
| L6  | content_word | 0.3% | **25.0%** |
| L6  | function_word | 4.7% | N/A |
| L7  | content_word | 0.0% | **40.0%** |
| L9  | content_word | 0.0% | **44.4%** |
| L10 | content_word | 0.0% | **46.2%** |
| L11 | content_word | 0.4% | **100.0%** |
| L11 | function_word | **18.6%** | N/A |

趋势非常清晰：
- **Function word features**: 高 geo=input (4-19%)，即 t_geo 是激活它们的 token
- **Content word features**: 高 geo=causal (L7-L11 = 40-100%)，即 t_geo 是它们因果影响的 token。geo=input 接近 0%。
- 两类 feature 的 alignment 机制完全不同

### 关键发现 3: Margin 在各类型间差异显著

| Layer | content | function | punct | subword |
|-------|---------|----------|-------|---------|
| L0  | 0.234 | **0.035** | 0.201 | **0.289** |
| L6  | 0.150 | **0.020** | 0.075 | 0.174 |
| L7  | 0.146 | **0.020** | 0.033 | 0.157 |
| L11 | 0.287 | **0.220** | 0.285 | **0.356** |

- **Function word margin 全层最低**（0.02-0.22）：因为功能词在 embedding 空间中密集（" the", " a", " an" 距离近），最近邻之间 margin 天然小
- **Subword fragment margin 全层最高**（0.16-0.36）：BPE 片段在 embedding 空间中更孤立
- L11 所有类型 margin 都升高，但 function word 的提升幅度最大（L7: 0.020 → L11: 0.220, 11x），说明最后层的 decoder 特别学到了指向功能词的方向

### 关键发现 4: Punctuation features 是 "input triggers"

Punctuation features 的 geo=input 始终较高（L0=16.9%, L2=5.5%）。这意味着标点符号出现时确实优先激活 t_geo=该标点符号的 feature。但 cos(d,mu) 适中（0.09-0.22），暗示标点虽然是激活条件之一，但不是唯一的。

---

## 012a-d 原始结果（整体指标）

### Alignment Margin — 所有层 real >> null

| Layer | Median Margin | Null (rotated) | Null (random) | Ratio |
|-------|---------------|----------------|---------------|-------|
| L0    | 0.244         | 0.005          | 0.005         | 49x   |
| L4    | 0.187         | 0.005          | 0.005         | 37x   |
| L7    | 0.138         | 0.005          | 0.005         | 28x   |
| L11   | 0.305         | 0.005          | 0.005         | 61x   |

U 形分布：早期层和晚期层高，L7 最低。

### Hub Token 分析

| 指标 | t_geo tokens | 全 vocab | 差异 |
|------|-------------|----------|------|
| 覆盖率 | ~5500 tokens (11%) | 50257 | 每个 token 平均被 1.1 个 feature 选中 |
| Embedding norm | 3.89-3.94 | 3.96 | t_geo 偏好略低 norm 的 token |
| kNN similarity | 0.537-0.540 | 0.558 | t_geo 偏好更孤立的 token |
| Top hub | " the" (10-19 features) | — | 高频功能词垄断轻微 |

### 层间 Jaccard Overlap

```
        L0    L3    L6    L9    L11
  L0    —    0.66  0.59  0.57  0.73
  L3          —    0.59  0.55  0.66
  L6                —    0.53  0.59
  L9                      —    0.60
  L11                           —
```

L0-L11 Jaccard (0.73) 高于 L0-L7 (0.56)。

### 全层 IO Decomposition — Feature 类别

| Layer | Alive% | context | unaligned | output | token |
|-------|--------|---------|-----------|--------|-------|
| L0    | 12.5%  | 42%     | 57%       | 0.5%   | 0     |
| L4    | 24.2%  | 36%     | 63%       | 0.9%   | 1     |
| L7    | 26.6%  | 31%     | 67%       | 1.8%   | 0     |
| L11   | 8.2%   | 58%     | 31%       | 10.5%  | 1     |

- Alive 峰值 L7 (26.6%)，L11 仅 8.2%
- Output feature 从 L0 (0.5%) 单调增长到 L11 (10.5%)
- Token feature: 全 12 层共 2 个
- input=causal: 全层 0.0%

### 四视角一致性

| Layer | geo=logit | geo=input | geo=causal | input=causal |
|-------|-----------|-----------|------------|--------------|
| L0    | 70.1%     | 3.9%      | 0.5%       | 0.0%         |
| L6    | 67.2%     | 0.7%      | 1.9%       | 0.0%         |
| L11   | 87.1%     | 2.0%      | 10.5%      | 0.0%         |

---

## 综合解读：t_geo 的三种含义

按 token 类型分层后，t_geo 的含义变得清晰——它对不同类型的 feature 意味着不同的事情：

### 1. Function Word Features: t_geo ≈ "我被什么词激活"

- cos(d,mu) = 0.29-0.32（decoder 方向 ≈ 激活质心方向）
- t_geo == t_mean 达到 14-35%
- geo=input 达到 4-19%
- 机制：这些 feature 的 decoder 学到了指向激活它们的 function word 的方向。因为 function word 频率高、语法角色明确，它们确实形成了可解释的 "输入 token detector"
- 但只占全部 feature 的 ~2.5%

### 2. Content Word Features: t_geo ≈ "我影响哪个词的 logit"

- cos(d,mu) ≈ 0 甚至为负（decoder 不等于激活质心）
- geo=input ≈ 0%
- geo=causal 在晚期层达到 40-100%
- 机制：这些 feature 的 decoder 指向它们因果影响的输出 token。L11 最强。
- 占全部 feature 的 ~61%

### 3. Subword Fragment Features: t_geo ≈ 几何标签（无功能意义）

- Margin 最高（最稳定的最近邻）
- geo=input = 0%, geo=causal 在晚期层也较高
- 机制：BPE 片段在 embedding 空间中最孤立，所以最近邻 margin 最大。但这只是几何属性，不直接对应功能。
- 占全部 feature 的 ~31%

### 为什么 input=causal 永远是 0%？

现在可以解释了：
- Function word features: t_geo ≈ t_input（激活条件），但 geo=causal 数据不足（这些 feature 通常不在 causal analysis 的 top-200 候选中，因为它们 geo_max_sim 低）
- Content word features: t_geo ≈ t_causal（输出方向），但 geo=input=0%
- 两类 feature 的 t_geo 指向不同的 token，所以 input=causal=0% 是必然的：**t_input 和 t_causal 本身就是不同的量**

---

## 假设判定（修订版）

| # | 假设 | 整体结果 | 按类型分层后 | 判定 |
|---|------|---------|-------------|------|
| H1 | t_geo = mean activation direction | 1-3% | **function word: 14-35%** | **条件成立**：仅对 function word features 成立 |
| H2 | t_geo = next-token | 0.2-0.5% | **function word L11: 4.96%** | **条件成立**：仅对 function word features 在 L11 |
| H3 | t_geo 是 artifact | margin >> null | 所有类型均 >> null | **否决** |
| H4 | embedding 几何偏置 | 轻微 | function word margin 低、subword margin 高 | **确认**：不同类型 margin 差 10x |
| H5 | 层依赖性 | U 形 margin | content 的 geo=causal 单调增长 | **确认** |

---

## 决策矩阵

| 结果模式 | 我们观察到的 | 结论 |
|----------|-------------|------|
| t_geo 对不同类型含义不同 | function=input label, content=output label | **Token type 决定 t_geo 语义** |
| L11 content feature geo=causal=100% | 最后层 decoder 编码输出方向 | **L11 的 content word t_geo 是 causal label** |
| Function word t_geo=t_mean=35% | 功能词 feature 是 input detector | **但只占 2.5% features** |
| 整体 token feature ≈ 0 | 没有三视角一致的 feature | **"token feature" 概念在 SAE 中不成立** |

## 下一步建议

1. **分类型做 feature 解释**：function word features 和 content word features 需要不同的解释框架
2. **L11 content word features 做 case study**：这些 feature 的 t_geo 是真正的 causal label，值得深入分析它们"做了什么"
3. **放弃统一的 token alignment 目标**：不同类型 feature 的 alignment 机制不同，不可能用一个 loss 同时优化
4. **探索 type-aware anchor loss**：对 function word features 用 input-aligned loss，对 content word features 用 output-aligned loss
