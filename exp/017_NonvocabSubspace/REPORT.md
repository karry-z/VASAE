# GPT-2 残差流非稀疏成分的正体鉴定

## 摘要

本报告回答一个核心问题：**GPT-2 残差流中，无法被少量词向量稀疏重建的成分，具体是什么？**

通过 7 个实验（A-F + 分解重建实验），我们建立了以下完整图景：

1. **非稀疏成分本质上是稠密的**：它不是"词汇表外"的方向（E 的列空间已是 R^768），而是在词汇表上呈现近均匀分布的系数模式（Gini ≈ 0.48, 熵比 ≈ 0.96）
2. **它由可解读的语言学特征构成**：主要是**序列位置**、**token 频率**、**词首标记**（is_word_start）三个特征，合计解释 OMP 残差能量的 80-99%
3. **它的产生源头是早期 MLP 和 attention 层**：Layer 0 的 attn.0 和 mlp.0 各贡献约 50%；随着层数增加，后续子层不断累积
4. **它对下一词预测有关键作用**：Layer 11 的非稀疏 PC1 被消融后 KL 散度 = 29.4，top-1 预测改变率 = 86%——这是模型最关键的单一方向
5. **即使完全移除已识别特征，剩余信号的相对 OMP 误差仍约 83%**：这表明存在更深层的几何不兼容性

---

## 1. 背景与问题定义

### 1.1 OMP 稀疏重建

给定 GPT-2 第 l 层残差流 h_l ∈ R^768 和词嵌入矩阵 E ∈ R^{50257×768}，OMP@k 选取 k 个词向量，通过最小二乘拟合重建 h_l。实验使用 k=8。

**关键概念澄清**：E 有 50257 行、768 列，rank = 768。因此 span(E) = R^768——不存在"词汇表之外"的方向。OMP 残差 r = h - h_recon 并非正交于 E，而是**在 E 上呈稠密表示**，即需要同时激活大量词向量才能表达。

### 1.2 前序发现

前序实验（`scripts/analyze_missing_components.py`）已确认：
- 所有子层输出的 OMP@8 相对误差 ≈ 87-96%
- PCA@8 远优于 OMP@128（低维结构存在，但不与词向量对齐）
- LayerNorm 后浅层 OMP 误差降 99%

本报告聚焦：**这些不与 E 对齐的低维方向，具体编码了什么信息？**

### 1.3 实验设置

- 模型：GPT-2 (117M)，12 层，768 维，12 头
- 数据：OpenWebText 采样 256 篇文本，max_length=64，flatten 后随机采样 512 个 token activation
- 代码：`scripts/analyze_nonvocab_subspace.py`（实验 A-F）, `scripts/exp_decompose_and_reconstruct.py`（分解重建实验）
- 输出目录：`exp/nonvocab_subspace/`

---

## 2. 实验 A：非稀疏成分的子层来源分解

**文件**: `expA_source_decomposition.json`, `expA_source_decomposition.png`

### 2.1 方法

GPT-2 的残差流是逐层累加的：h_l = wte + wpe + Σ_{i=0}^{l}(attn_i + mlp_i)。对每层计算 OMP@8 残差 r = h - h_recon 后，将每个子层的实际输出 c_i 投影到 r 上：

```
signed_fraction(c_i) = mean(c_i · r) / ||r||^2
```

这直接回答：非稀疏内容中，有多少比例来自哪个子层？

### 2.2 结果

**Layer 0**（`expA_source_decomposition.json`, layer_0）:

| 子层 | signed_fraction | 解读 |
|------|----------------|------|
| attn.0 | **0.509** | 第一层 attention 贡献残差的 51% |
| mlp.0 | **0.471** | 第一层 MLP 贡献残差的 47% |
| wte | 0.011 | 词嵌入本身贡献极小 |
| wpe | 0.009 | 位置嵌入直接贡献极小 |

**Layer 11**（`expA_source_decomposition.json`, layer_11）:

| 子层 | signed_fraction | 解读 |
|------|----------------|------|
| attn.11 | **0.279** | 最大单一贡献者 |
| mlp.10 | **0.153** | |
| mlp.11 | **0.140** | |
| mlp.2 | **0.123** | 早期 MLP 影响持续到最后一层 |
| mlp.9 | 0.048 | |
| attn.0 | 0.023 | 即使在 L11，L0 attention 仍可见 |

### 2.3 讨论

非稀疏成分的来源呈**分散累积**模式：浅层由 attn.0 + mlp.0 主导（各~50%），深层则由更多子层分担。值得注意的是：

- **mlp.2** 在所有深层（L2-L11）中贡献显著（L2: 8%, L6: 5.5%, L11: 12.3%），而其 `comp_norm` 极大（L2: 137k vs 残差 143k），说明 mlp.2 产生了巨大的向量但大部分被抵消，残留的非稀疏成分仍然可观
- 所有层的 `sum_of_fractions ≈ 1.0`（验证分解完备性），`recon_dot_residual ≈ 0`（验证 OMP 重建与残差正交）

---

## 3. 实验 B：稠密词汇系数的结构

**文件**: `expB_dense_vocab_structure.json`, `expB_dense_vocab_structure.png`

### 3.1 方法

既然 OMP 残差不是"E 之外"的方向，它在 E 上的表示是什么结构？计算：

```
c = E @ (E^T E)^{-1} @ r
```

得到残差在全部 50257 个词向量上的系数向量 c，然后分析 c 的分布特性。

### 3.2 结果

**所有层一致的结论**（`expB_dense_vocab_structure.json`）：

| 层 | Gini 系数 | 熵比（1=均匀） |
|----|-----------|---------------|
| L0 | 0.479 | 0.962 |
| L3 | 0.478 | 0.962 |
| L7 | 0.476 | 0.962 |
| L11 | 0.470 | 0.960 |

Gini 系数 ≈ 0.48（1=完全稀疏，0=完全均匀），熵比 ≈ 0.96——残差的词汇系数近乎均匀分布在整个 50257 维词表上。

Top 系数 token 无可解读模式——出现的都是罕见 token（如 `?????-?????-`, `ModLoader`, `mbuds`, `assetsadobe`），且频率不稳定。这些不是某个"缺失词"的问题，而是信息分散在所有词向量上。

### 3.3 讨论

这确认了核心几何事实：**非稀疏成分 = 在词汇表上稠密的方向**。它不能被少数几个词向量近似，而需要同时调动数千个词向量的微小系数。这是 OMP@k 固有的局限——不是 k 不够大，而是信息的编码方式与词向量字典不兼容。

---

## 4. 实验 D：语言学特征探测——非稀疏成分编码了什么

**文件**: `expD_linguistic_probes.json`, `expD_linguistic_probes.png`

### 4.1 方法

对每层 OMP 残差做 PCA 取 top-8 主方向，将每个 token 在这些方向上的投影与 6 个语言学特征做 Pearson 相关：

- **position**: token 在序列中的位置 (0-63)
- **log_frequency**: token unigram 频率的对数
- **is_function_word**: 是否功能词（the, is, of 等）
- **is_punctuation**: 是否标点符号
- **is_word_start**: token 字符串是否以空格开头（BPE 词首标记）
- **token_str_length**: token 字符串长度

### 4.2 结果

#### 4.2.1 残差范数与特征的聚合相关（residual norm vs feature）

（`expD_linguistic_probes.json`, 各层 aggregate 部分）

| 特征 | L0 | L3 | L5 | L7 | L9 | L11 |
|------|-----|-----|-----|-----|-----|------|
| position | **-0.33** | **-0.26** | **-0.22** | **-0.25** | **-0.22** | **+0.24** |
| log_frequency | **-0.24** | -0.02 | -0.11 | -0.07 | -0.09 | **+0.29** |
| is_word_start | -0.08 | **-0.26** | **-0.19** | **-0.21** | **-0.20** | -0.01 |
| is_punctuation | -0.12 | 0.00 | 0.00 | -0.05 | -0.05 | 0.00 |
| is_function_word | -0.06 | 0.08 | 0.02 | 0.04 | -0.01 | **+0.13** |
| token_str_length | +0.13 | -0.13 | -0.04 | -0.10 | -0.07 | **-0.14** |

#### 4.2.2 逐 PC 相关（Layer 0, 最强信号）

（`expD_linguistic_probes.json`, layer_0, per_pc）

| PC | 最强特征 | r 值 | p 值 |
|----|---------|------|------|
| PC0 | position | **-0.50** | 6.1e-34 |
| PC1 | log_frequency | **-0.74** | 0.0 |
| PC2 | position | **-0.69** | 0.0 |
| PC3 | is_function_word | **-0.59** | 0.0 |
| PC3 | is_word_start | **-0.55** | 3.2e-42 |
| PC4 | token_str_length | -0.24 | 2.8e-8 |

#### 4.2.3 逐 PC 相关（Layer 1, 频率信号更强）

（`expD_linguistic_probes.json`, layer_1, per_pc）

| PC | 最强特征 | r 值 |
|----|---------|------|
| PC1 | log_frequency | **-0.86** |
| PC1 | token_str_length | **+0.61** |
| PC2 | log_frequency | **+0.54** |
| PC3 | position | **+0.37** |

#### 4.2.4 逐 PC 相关（Layer 11, 不同模式）

（`expD_linguistic_probes.json`, layer_11, per_pc）

| PC | 最强特征 | r 值 |
|----|---------|------|
| PC0 | log_frequency | -0.22 |
| PC1 | is_word_start | -0.22 |
| PC2 | is_function_word | **+0.35** |
| PC2 | is_word_start | +0.22 |
| PC3 | is_function_word | +0.23 |
| PC3 | is_word_start | +0.22 |
| PC4 | is_word_start | **-0.48** |
| PC5 | is_word_start | **+0.45** |

### 4.3 讨论

非稀疏成分的前几个主方向编码了明确的语言学特征：

- **浅层（L0-L1）**：position 和 log_frequency 是最强信号。PC1 的频率相关达 |r|=0.86，这表明 MLP.0 将 token 频率信息写入残差流，且这种信息无法用少数词向量稀疏表示
- **中间层（L3-L9）**：is_word_start 成为稳定的强信号（r ≈ -0.20~-0.26），position 仍显著
- **深层（L11）**：特征信号减弱（最大 |r| ≈ 0.48），is_function_word 和 is_word_start 成为主要特征；position 的符号翻转（浅层负相关 → L11 正相关），log_frequency 也翻转。这提示深层可能对这些特征做了非线性变换

---

## 5. 实验 E：写入-读取电路对应

**文件**: `expE_write_read_circuits.json`, `expE_circuits_L*.png`

### 5.1 方法

对每个残差 PCA 主方向 v_i：
- **写入者 (Writers)**：计算所有 OV 矩阵和 MLP 权重矩阵对 v_i 的投影 → 谁写入了这个方向？
- **读取者 (Readers)**：计算下游所有 QK 矩阵对 v_i 的读出强度 → 谁在利用这个方向？

### 5.2 结果

**Layer 0 非稀疏 PC0 的电路**（`expE_write_read_circuits.json`, layer_0, PC0）：

| 角色 | 组件 | 强度 |
|------|------|------|
| 写入者 #1 | **L0_MLP** | projection = 0.468 |
| 写入者 #2 | L0_H6_OV | 0.036 |
| 写入者 #3 | L0_H8_OV | 0.034 |
| 读取者 #1 | **L4_H11_QK** | readout = 3894 |
| 读取者 #2 | L2_H2_QK | 1199 |
| 读取者 #3 | L2_H1_QK | 679 |

### 5.3 讨论

**L0_MLP → L4_H11 是关键的非稀疏信息传输通道**。L0 的 MLP 写入位置/频率信息的稠密方向，L4 的 Head 11 通过 QK 矩阵专门读取这个方向。这构成了一个跨层电路：

```
token identity → L0_MLP (写入频率/位置信号) → 残差流 → L4_H11 (读取并用于attention routing)
```

这解释了为什么非稀疏成分对模型功能很重要——它不是噪声，而是模型有意设计的信息传递机制。

---

## 6. 实验 C：下游 QK 读出强度比较

**文件**: `expC_downstream_readers.json`, `expC_readout_L*.png`

### 6.1 方法

对每层激活 h，分离为：h = h_sparse (OMP 重建) + r (残差)。计算下游每个 attention head 的 QK 矩阵对两部分的读出强度：

```
residual_total = ||W_QK @ r||^2 (对残差的读出)
recon_total = ||W_QK @ h_sparse||^2 (对稀疏部分的读出)
ratio = residual_total / recon_total (> 1 说明更关注残差)
```

### 6.2 结果

**Layer 0 → 下游头部读出**（`expC_downstream_readers.json`, layer_0）：

| Head | residual_total | recon_total | ratio |
|------|---------------|-------------|-------|
| L1_H2 | 214.7 | 69.0 | **3.11** |
| L2_H5 | 2222.3 | 730.1 | **3.04** |
| L1_H10 | 228.8 | 74.7 | **3.06** |
| L1_H3 | 125.1 | 42.7 | **2.93** |
| L1_H4 | 324.4 | 110.7 | **2.93** |
| L2_H2 | 6932.2 | 2451.6 | **2.83** |

几乎所有下游头的读出比 > 2，意味着**它们从残差中获取的信息是从稀疏部分获取的 2-3 倍**。

### 6.3 讨论

这不是偶然——下游 attention head 被训练成主动读取非稀疏方向。ratio > 2 在所有层中普遍存在，说明非稀疏信号是模型内部通信的重要通道，而非可以安全丢弃的"噪声"。

---

## 7. 实验 F：消融验证——非稀疏成分有多重要？

**文件**: `expF_ablation_impact.json`, `expF_ablation_impact.png`

### 7.1 方法

从 h_l 中去除 OMP 残差的 top-k PCA 方向的投影分量，计算 logit lens 输出的变化：
- **prediction_change_rate**：top-1 预测改变的比例
- **KL_divergence**：消融前后输出分布的 KL 散度
- **entropy_change**：输出熵的变化（正 = 更不确定，负 = 更确定）

对照组：消融激活本身的 top-k PCA 方向（方差最大方向）。

### 7.2 结果

**关键发现：Layer 11 的非稀疏 PC1 具有致命重要性**

（`expF_ablation_impact.json`, layer_11）

| k 值 | 对象 | prediction_change_rate | KL_divergence | entropy_change |
|------|------|----------------------|---------------|----------------|
| k=1 | nonsparse_pcs | **0.859** | **29.45** | **-3.10** |
| k=1 | activation_pcs | 0.258 | 0.19 | +0.31 |
| k=4 | nonsparse_pcs | **0.836** | **22.74** | **-3.02** |
| k=4 | activation_pcs | 0.389 | 0.59 | -0.23 |
| k=8 | nonsparse_pcs | **0.852** | **24.65** | **-3.02** |
| k=8 | activation_pcs | 0.473 | 0.86 | +0.15 |

**其他层：非稀疏 PC 的影响小于激活 PC**

（`expF_ablation_impact.json`, layer_7, k=4）

| 对象 | prediction_change_rate | KL_divergence |
|------|----------------------|---------------|
| nonsparse_pcs | 0.461 | 4.96 |
| activation_pcs | 0.793 | **15.05** |

### 7.3 讨论

Layer 11 的结果极为特殊：

- **仅移除 1 个非稀疏 PC**，就导致 86% 的预测改变和 KL=29.4（而同层激活 PC1 仅 KL=0.19）
- **entropy_change = -3.1**：移除后模型变得更"确定"但给出错误答案，说明该方向编码了用于区分候选词的关键信息
- 中间层（L2-L9）则相反：激活 PCA 方向的消融影响更大（KL 5-15），非稀疏 PCA 的影响较温和（KL 2-5）。这符合预期——中间层的非稀疏成分是"管道信息"（位置、频率），尚未直接用于预测

**Layer 11 的非稀疏 PC1 是 GPT-2 整个模型中最关键的单一方向之一。**

---

## 8. 分解重建实验：显式特征移除后的稀疏重建

**文件**: `exp_decompose_reconstruct.json`, `exp_decompose_reconstruct.png`

### 8.1 方法

既然非稀疏成分编码了位置、频率、词形，能否显式分解它们，使剩余信号变得稀疏？

构建特征矩阵 X [N, d_feat]，通过 OLS 回归学习 W = (X^TX + λI)^{-1} X^T H，然后：
```
h_clean = h - X @ W (移除可解读特征的线性投影)
```

在 h_clean 上做 OMP@8，测试重建效果。

渐进特征集：
- **F0**: 无特征（基线）
- **F1**: position（标量）
- **F2**: position + log_frequency
- **F3**: position + log_frequency + is_word_start
- **F4**: + is_punctuation + token_str_length
- **F5**: + is_function_word（6 个标量特征）
- **F6**: position one-hot (64维) + 5 个标量特征（共 69 维）

同时对照 OMP@16/32/64（增加稀疏预算但不移除特征）。

### 8.2 结果

#### 8.2.1 绝对 OMP 误差大幅下降

（`exp_decompose_reconstruct.json`, relative_error 字段）

| 层 | F0 (基线) | F3 (+词首) | F6 (69维) | 误差下降 |
|----|----------|-----------|----------|---------|
| L0 | 0.875 | 0.396 | **0.173** | 80.3% |
| L1 | 0.794 | 0.568 | **0.093** | 88.3% |
| L3 | 0.837 | 0.767 | **0.007** | **99.2%** |
| L5 | 0.840 | 0.785 | **0.018** | 97.9% |
| L7 | 0.840 | 0.782 | **0.028** | 96.7% |
| L9 | 0.835 | 0.752 | **0.049** | 94.1% |
| L11 | 0.952 | 0.324 | **0.110** | 88.5% |

#### 8.2.2 特征方差解释比

（`exp_decompose_reconstruct.json`, variance_explained_by_features 字段）

| 层 | F3 方差解释 | F6 方差解释 |
|----|-----------|-----------|
| L0 | 52.4% | **79.2%** |
| L1 | 24.8% | **88.9%** |
| L3 | 8.2% | **99.2%** |
| L5 | 6.4% | **97.9%** |
| L7 | 6.9% | **96.6%** |
| L9 | 10.0% | **93.9%** |
| L11 | 65.4% | **88.1%** |

#### 8.2.3 清洗后信号的相对误差——关键发现

（`exp_decompose_reconstruct.json`, relative_error_of_clean 字段）

| 层 | F1 clean rel_err | F3 clean rel_err | F6 clean rel_err |
|----|-----------------|-----------------|-----------------|
| L0 | 0.872 | 0.832 | **0.829** |
| L1 | 0.792 | 0.755 | **0.833** |
| L3 | 0.835 | 0.835 | **0.846** |
| L5 | 0.839 | 0.839 | **0.848** |
| L7 | 0.839 | 0.840 | **0.835** |
| L9 | 0.834 | 0.836 | **0.807** |
| L11 | 0.952 | 0.938 | **0.920** |

#### 8.2.4 与暴力增加 OMP 预算的比较

（`exp_decompose_reconstruct.json`, F0_baseline_k* 字段）

| 层 | OMP@8 | OMP@16 | OMP@32 | OMP@64 | **F6+OMP@8** |
|----|-------|--------|--------|--------|-------------|
| L0 | 0.875 | 0.856 | 0.826 | 0.780 | **0.173** |
| L3 | 0.837 | 0.825 | 0.804 | 0.775 | **0.007** |
| L5 | 0.840 | 0.829 | 0.807 | 0.778 | **0.018** |
| L11 | 0.952 | 0.944 | 0.934 | 0.921 | **0.110** |

### 8.3 讨论

这是本研究最核心的结果，揭示了双层结构：

**第一层：可解读特征**（position one-hot + frequency + wordform = 69 维）解释了 OMP 残差 80-99% 的能量。F6 + OMP@8 将绝对误差降至 0.7-17%，远超 OMP@64（仍在 78-92%）。这确认：**非稀疏成分的主体是 position、frequency、word-start 这三类特征。**

**第二层：不可消除的几何失配**。`relative_error_of_clean ≈ 0.83` 在所有层、所有特征集上高度一致。这意味着：
- 移除特征后，残留信号的绝对量级大幅缩小（缩小到 1-21%）
- 但这个小信号本身**仍然有 83% 无法稀疏重建**
- 这不是某个未发现的特征——而是激活方向与词向量字典之间的**固有几何不兼容性**

类比：词向量字典 E 像一组坐标轴。模型的计算产生的方向不平行于任何轴，也不平行于少数几个轴的组合。即使移除了可解读的"偏移量"，剩余的语义计算仍然在词向量坐标系中呈现稠密表示。

---

## 9. 综合结论

### 9.1 非稀疏成分的完整身份

| 成分 | 占比（能量） | 来源 | 功能 |
|------|------------|------|------|
| **序列位置编码** | 大，尤其浅层 | wpe + L0 attn → OV 重写 | 位置感知的 attention routing |
| **Token 频率信号** | 大，尤其 L0-L1 | L0 MLP | 先验概率校准 |
| **词首标记 (is_word_start)** | 大，尤其 L3-L9 | attention heads | BPE 分词边界信息 |
| **功能词/标点标记** | 中等 | 各层累积 | 句法结构标记 |
| **不可解读的稠密残留** | ~17% 能量但 ~83% 相对误差 | 全部子层 | 语义计算的固有表示方式 |

### 9.2 对 VASAE 架构的启示

当前发现直接建议以下架构改进：

**方案 1：双路径解码器**
```python
h_reconstructed = E @ sparse_codes + X @ W_features + bias
```
其中 X 是 [batch, 69] 的特征矩阵（position one-hot + token 属性），W_features 是可学习的 [69, 768] 权重。sparse_codes 专注于语义内容，W_features 处理系统性偏移。

**方案 2：低秩自由分量**
在稀疏解码器之外，添加一个小的 learned low-rank component（rank 8-16），让模型自由学习非稀疏方向，而不预设它们是什么。

**方案 3（本报告推荐）：结合两者**
```python
h_reconstructed = E @ sparse_codes + V @ dense_codes + bias
```
其中 V ∈ R^{768×r}（r≈16）是可学习的非稀疏基底。用本报告识别的特征初始化 V，但允许梯度进一步优化。

### 9.3 开放问题

1. **83% 的几何失配来自哪里？** 移除 69 维特征后，剩余信号虽小但仍抗拒稀疏化。这可能是 attention 计算产生的混合方向（多个词义的加权平均），也可能是模型的 implicit regularization 产生的低范数但分散的表示
2. **Layer 11 的非稀疏 PC1 到底编码了什么？** 它的消融影响极大（KL=29.4），但与已知简单特征的相关性较弱（最大 |r|≈0.35）。它可能编码了更复杂的上下文信息
3. **非稀疏成分跨输入的稳定性如何？** 如果是模型固有结构（而非输入依赖），则固定基底 V 即可；否则需要依赖输入的动态编码

---

## 附录：实验配置与复现

```bash
# 实验 A-F
uv run python scripts/analyze_nonvocab_subspace.py \
  --exp all --layers 0,1,2,3,4,5,6,7,8,9,10,11 \
  --n_samples 512 --n_texts 256 \
  --output_dir exp/nonvocab_subspace

# 分解重建实验
uv run python scripts/exp_decompose_and_reconstruct.py \
  --layers 0,1,3,5,7,9,11 --n_samples 512 --n_texts 256 \
  --output_dir exp/nonvocab_subspace
```

所有结果数据（JSON）和可视化（PNG）保存在 `exp/nonvocab_subspace/` 目录下。
