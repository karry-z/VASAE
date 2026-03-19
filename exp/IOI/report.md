---
title: IOI 干预实验
date: 
---

# 目的
前面的实验已经证明 VASAE 可以分解为 token 切近的方向，但是否可以通过干预这些方向影响下游任务的效果（比如增加某些方向的信息，是得模型最终回答正确的 token；或者删除某些方向，以此抹除模型的部分知识，使其不输出某些有害信息）。概括来说，就是要证明这些分解出来的方向是具有因果效应的，而不仅仅是读出中间表示含有这些信息（相关性，不能操作）

# 方法
我们采用 IOI（Indirect Object Identification）任务，并构造一对输入：clean text 与 corrupted text，其中前者对应模型应输出正确名字的情形，后者通过扰动关键结构使模型更倾向输出错误名字。记正确答案与错误答案的 logits 分别为 $u_{\text{correct}}$ 与 $u_{\text{wrong}}$，定义 logit difference 为 $\Delta u = u_{\text{correct}} - u_{\text{wrong}}$，该量衡量模型对正确答案的相对偏好。在理想情况下，clean 输入应满足 $\Delta u_{\text{clean}} > 0$，而 corrupted 输入会显著降低该值，形成 $\Delta u_{\text{corr}} < \Delta u_{\text{clean}}$，后者可视为任务性能退化的参考水平。

在此基础上，我们对 clean 输入的前向计算进行干预：选取单个 VASAE 特征方向，在其对应表示上施加修改，并记录干预后的 logit difference，记为 $\Delta u_{\text{clean, intervened}}$。若某一特征的干预导致 $\Delta u_{\text{clean, intervened}} < \Delta u_{\text{clean}}$，则说明该方向的变化削弱了模型对正确答案的支持，表明该特征可能参与了 IOI 解析过程。然而，仅凭这一现象仍不足以说明其为任务特异性的因果成分，因为类似的下降也可能来源于模型整体表示被破坏或噪声注入。为区分这两种情况，我们进一步比较干预后的结果与 corrupted 输入的表现：若 $\Delta u_{\text{clean, intervened}} \approx \Delta u_{\text{corr}}$，则说明该干预并非仅降低性能，而是使模型行为向“错误解析”状态系统性靠近，从而更有力地表明该特征承载了支持正确 IOI 解析的关键内部信息。

为了在不同特征之间进行可比分析，我们引入归一化指标刻画干预效果，即
$$\text{Recovery}^{(f)} = \frac{\Delta u_{\text{clean}} - \Delta u_{\text{clean, intervened}}^{(f)}}{\Delta u_{\text{clean}} - \Delta u_{\text{corr}}},$$
该指标衡量某一特征干预后将模型从 clean 状态“推向” corrupted 状态的程度。当该值接近 1 时，说明干预几乎完全复现了 corrupted 输入所带来的行为变化；当其接近 0 时，则表明该特征对任务决策影响较弱。此外，为避免将“全局破坏性特征”误判为因果特征，我们还需结合非目标行为上的稳定性指标进行控制，例如监测干预是否导致整体 logit 分布异常扩散或在非 IOI 相关 token 上产生同等幅度的扰动，从而区分特异性因果效应与非特异性性能退化。

实验过程中，我们对所有 VASAE 特征逐一施加干预，并统计其对 $\Delta u$ 的影响分布。预期结果是，大多数特征的干预仅产生微弱或不稳定的变化，而少数特征能够显著降低 $\Delta u_{\text{clean}}$，并使其接近 $\Delta u_{\text{corr}}$，表明这些方向在模型执行 IOI 解析时具有实质性的因果作用。进一步地，在这些高影响特征中，若存在一部分在非 IOI 指标上表现出较小副作用，则可以认为它们更可能对应于任务特异性的内部机制，而非通用计算结构。我们还预期这些因果特征在层级分布上呈现非均匀性，例如集中于中后层或特定子空间，这将为理解模型内部信息流提供额外线索。

需要强调的是，该实验只能证明 VASAE 分解中存在具有因果效应的特征方向，但并不能解决表示的识别性问题。换言之，即便某些特征在事后被验证为因果性的，也不意味着当前分解在数学上是唯一的，或这些特征可以在无干预验证的情况下被可靠识别。因此，该结果更应被理解为对”因果结构存在性”的实证支持，而非对”可解释分解完备性”的充分证明。

# 实验流程

## 模型

使用 `010_soft_align` 实验中训练的 soft-aligned VASAE 模型（k=32, anchor_coeff=1e-3），覆盖 GPT-2 全部 12 层（L0–L11）。模型路径：
```
/scratch/b5bq/pu22650.b5bq/VASAE_out/010_soft_align/010_soft_gpt2_L{layer}_k32_a1e-3
```

## 脚本

### 1. 核心实验脚本：`scripts/eval_ioi_feature_sweep.py`

对每个 IOI prompt，逐一消融 clean 输入在目标层的每个活跃 VASAE 特征，计算 Recovery 和 Specificity。

关键设计：
- 使用 SAE 完整重建（`du_recon`）而非原始前向传播（`du_clean`）作为 Recovery 分子的基线，以隔离单个特征消融的效应，避免 SAE 重建误差污染因果度量
- 过滤无效 prompt：跳过 `du_clean <= 0`（模型本身答错）和 `|du_clean - du_corr| < min_gap`（clean/corrupted 无区分度）的 prompt
- KL 散度同样以 SAE 重建分布为参考，而非原始模型分布

单层本地测试（CPU，4 prompts）：
```bash
uv run python scripts/eval_ioi_feature_sweep.py \
    --layer-idx 8 --n-prompts 4 --device cpu \
    --sae-root /scratch/b5bq/pu22650.b5bq/VASAE_out/010_soft_align \
    --output-dir /tmp/ioi_sweep_test
```

### 2. Slurm 作业：`exp/IOI/run_feature_sweep.sh`

12 层并行 array job（`--array=0-11`），每层 1 GPU，100 prompts：
```bash
sbatch exp/IOI/run_feature_sweep.sh
```

输出存储于 `/scratch/b5bq/pu22650.b5bq/VASAE_out/ioi_feature_sweep/layer_{0..11}.json`。

### 3. 后处理与绘图：`scripts/plot_ioi_feature_sweep.py`

在全部 12 层作业完成后，在登录节点运行（仅 CPU）：
```bash
uv run python scripts/plot_ioi_feature_sweep.py \
    --input-dir /scratch/b5bq/pu22650.b5bq/VASAE_out/ioi_feature_sweep \
    --output-dir exp/IOI/figures
```

生成：
- `top_features_table.md`：按 mean Recovery 排序的 top-20 特征表
- `max_recovery_vs_layer.pdf`：各层最大 Recovery 曲线
- `recovery_heatmap_layer{best}.pdf`：最佳层的 Recovery 热力图（features × prompts）

# 结果
## 表：代表性特征的干预效果

| Feature ID | Layer | Feature Strength | $\Delta u_{\text{clean}}$ | $\Delta u_{\text{corr}}$ | $\Delta u_{\text{clean, intervened}}$ | Effect | Recovery | Specificity Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6050 | 1 | 3.3901 | 1.4259 | 0.0597 | 0.7111 | 1.9579 | 1.3894 | 51.8498 |
| 4049 | 0 | 10.3791 | 3.5142 | 2.6983 | 1.0944 | 1.0359 | 1.2697 | 32.6691 |
| 3617 | 1 | 6.4383 | 1.1188 | 0.0233 | -0.4040 | 1.0239 | 1.1616 | 78.7427 |
| 1932 | 1 | 4.9155 | 2.2228 | 1.2839 | 2.2555 | 0.9125 | 0.9975 | 240.2463 |
| 3590 | 1 | 6.4451 | 1.7854 | 0.6092 | -0.6098 | 0.8085 | 0.9335 | 24.0343 |
| 4649 | 4 | 8.1222 | 1.6752 | 0.1642 | -1.0137 | 1.1600 | 0.7802 | 50.9851 |
| 2877 | 2 | 7.0833 | 1.6691 | 0.4853 | -1.0387 | 0.8993 | 0.7750 | 19.2771 |
| 5595 | 2 | 6.3969 | 1.2781 | 0.6115 | -4.6668 | 0.5144 | 0.7717 | 87.4279 |
| 6143 | 1 | 5.8612 | 3.0245 | 1.9158 | -2.0406 | 0.7971 | 0.7190 | 122.6794 |
| 5628 | 0 | 12.9279 | 1.7360 | 0.2098 | -0.9344 | 1.1272 | 0.6880 | 46.8644 |
| 3733 | 7 | 20.8690 | 1.6460 | 0.1539 | -0.8677 | 0.9689 | 0.6758 | 34.7489 |
| 5315 | 1 | 5.3388 | 2.1706 | 0.9513 | 0.8476 | 0.5381 | 0.5813 | 31.4781 |
| 2842 | 3 | 6.2160 | 2.8894 | 1.8819 | -0.8534 | 0.5852 | 0.5808 | 146.4569 |
| 13 | 5 | 16.3010 | 1.7034 | 0.1658 | -1.1053 | 0.8812 | 0.5662 | 16.6370 |
| 2046 | 10 | 15.9733 | 3.5142 | 2.6983 | 0.4046 | 0.4601 | 0.5639 | 88.7479 |
| 3395 | 2 | 4.4334 | 2.5852 | 0.9373 | 1.6426 | 0.8064 | 0.5454 | 4.6405 |
| 3724 | 1 | 6.5965 | 1.6216 | 0.2720 | -1.0608 | 0.5255 | 0.5437 | 34.0763 |
| 278 | 3 | 7.9854 | 1.5426 | 0.0045 | -0.7780 | 0.8657 | 0.5423 | 35.4913 |
| 5448 | 1 | 6.4848 | 2.1774 | 0.3479 | -2.4738 | 0.8324 | 0.5416 | 77.1301 |
| 5628 | 6 | 13.9967 | 1.7360 | 0.2098 | -0.4132 | 0.7798 | 0.5298 | 29.4690 |

其中：
- Feature ID：VASAE 特征编号
- Layer：特征所在层
- Feature Strength：该特征在 clean 样本上的激活强度
- Effect：$\Delta u_{\text{clean}} - \Delta u_{\text{clean, intervened}}$
- Recovery：归一化靠近 corrupted 的程度
- Specificity Score：在 IOI 效果与非 IOI 破坏之间的比值或其他特异性指标。	IOI 效应：Recovery，副作用：KL divergence。最终：$\text{Specificity}^{(f)} = \frac{\text{Recovery}^{(f)}}{D_{\text{KL}}(P_{\text{clean}} \| P_{\text{intervened}}) + \epsilon}$
    - 高 Recovery + 低 Damage → 高 Specificity
→ 理想因果特征（精准改变 IOI 决策）
    - 高 Recovery + 高 Damage → 中等 Specificity
→ 有效但“粗暴”（可能是通用计算特征）
    - 低 Recovery + 高 Damage → 低 Specificity
→ 纯破坏（噪声特征）
    - 低 Recovery + 低 Damage → 接近 0
→ 无关特征

这个表可以只展示 top-k 因果特征。

## 分析

实验使用 100 个 IOI prompt，其中 73 个通过有效性过滤（$\Delta u_{\text{clean}} > 0$ 且 $|\Delta u_{\text{clean}} - \Delta u_{\text{corr}}| \geq 0.5$），每层 topk=32 个活跃特征，共计每层 2336 个 (feature, prompt) 对。Recovery 以 SAE 完整重建的 $\Delta u_{\text{recon}}$ 为基线，隔离单个特征消融的纯粹效应。

### 1. 稀疏因果结构符合预期

Recovery 分布呈现典型的稀疏因果图景：各层 median Recovery 接近 0（L0: 0.028, L5: 0.010, L11: −0.001），**大多数特征的消融对 IOI 决策几乎无影响**。约 50% 的 (feature, prompt) 对 Recovery 落在 $[0,1]$ 区间，Recovery $> 1$（过冲）的比例很低（L5 仅 17/2336，L11 仅 12/2336）。这与预期一致：只有少数特征承载了 IOI 任务的关键信息。

### 2. 代表性因果特征（表格讨论）

Top-20 特征的 mean Recovery 集中在 0.5–1.4 范围，其中多个特征接近理想值 1。按因果效应类型可分为三类：

**精准因果特征**（Recovery $\approx 1$, Specificity 高）：
- **L1 Feature 1932**（Recovery=0.997, Specificity=240.2）：几乎完美地复现了 corrupted 扰动的效果，且 KL 散度极低，说明它精准地改变了 IOI 决策而几乎不扰动其他输出分布。该特征仅在 3/73 个 prompt 上被激活，说明它是高度 prompt 特异性的 IOI 相关方向。
- **L1 Feature 3590**（Recovery=0.933, Specificity=24.0）：同样接近 1，但在 6 个 prompt 上激活且 std=1.43，表明其因果效应在不同 prompt 间存在较大波动。
- **L3 Feature 2842**（Recovery=0.581, Specificity=146.5）：中层代表，Specificity 极高，说明其消融对整体分布的扰动极小。

**过冲特征**（Recovery $> 1$）：
- **L1 Feature 6050**（Recovery=1.389）和 **Feature 3617**（Recovery=1.162）：消融效应超过了 corrupted 扰动。它们仅在 2–3 个 prompt 上激活，小样本可能导致 Recovery 估计的方差较大。
- **L0 Feature 4049**（Recovery=1.270）：前层特征，activation strength=10.4，可能承载了对后续层有放大效应的低层信息。

**跨层共享特征**：
- **Feature 5628** 在 L0（Recovery=0.688）和 L6（Recovery=0.530）均进入 top-20，说明某些因果方向可能跨层持续参与 IOI 计算。
- **Feature 13** 在 L5（Recovery=0.566）和 L6（Recovery=0.524）也有类似表现。这些跨层特征是否对应同一语义方向，有待进一步通过 decoder 向量的余弦相似度验证。

值得注意的是，表中许多高 Recovery 特征具有较高的 Specificity（多数 $> 30$），说明它们的消融主要影响了 IOI 相关的 logit difference，而非引发全局性的输出分布破坏。这与方法部分提出的"特异性因果效应 vs 非特异性性能退化"的区分框架一致。

### 3. 层级分布

因果特征在层间分布不均匀：

- **前层（L0–L1）因果特征最密集**：L1 有 31 个唯一特征 mean Recovery $\in [0.3, 1]$，远高于其他层（L5: 3 个, L8: 2 个, L11: 0 个）。这可能反映了 IOI 任务依赖的名字识别、共指消解等低层信息处理。L1 同时也是过冲最多的层（Recovery $> 1$ 的有 213/2336 对），说明前层特征的因果效应更强但也更容易超调。
- **中层（L3–L6）少量但精准**：每层 3–7 个强因果特征，Specificity 普遍较高（如 L3 Feature 2842 的 Specificity=146.5），说明中层特征更具任务特异性，消融副作用更小。
- **后层（L9–L11）因果效应衰减**：L11 无 mean Recovery $> 0.3$ 的特征，且活跃特征数锐减（L11: 42 个 vs L6: 129 个 vs L1: 112 个）。后层的表示更加集中，但对 IOI 决策的边际贡献更低，这与后层主要负责通用语言建模而非特定任务解析的假说一致。

### 4. 负 Recovery 特征

约 42–50% 的 (feature, prompt) 对具有负 Recovery，即消融该特征后 $\Delta u$ 反而增大（模型更偏向正确答案）。多数负 Recovery 幅度很小（接近 0），但 Feature 218 在 L7（mean Recovery=−0.549, 73/73 prompts 全激活）和 L8（mean Recovery=−0.379）表现出持续的负因果效应。这意味着 Feature 218 在中后层**抑制了模型对 IOI 正确答案的偏好**——消融它反而有助于 IOI 任务。结合其在 L0–L9 全层激活且 activation strength 持续增长的特点，Feature 218 可能对应于某种通用的注意力或位置编码机制，其存在对 IOI 解析构成了干扰而非帮助。

### 5. Max Recovery 随层变化（图 1 讨论）

![Max Recovery vs Layer](figures/max_recovery_vs_layer.pdf)

该图展示了每层在所有 (feature, prompt) 对上的最大 Recovery。曲线呈现 **L1 显著突出、其余层相对平坦** 的模式：

- **L1 的 max Recovery 最高（8.06）**，由 Feature 2541 在 prompt 39 上贡献。这一极端值反映了 L1 某些特征在特定 prompt 上的强烈因果效应，但也可能与该 prompt 的 clean-corrupted gap 或 SAE 重建特性有关。
- **其余层 max Recovery 在 2.0–4.0 之间波动**，无明显的单调趋势。L0（3.88）、L7（3.72）、L8（3.96）略高于中间层，暗示前层和中后层的部分特征在特定 prompt 上可能产生较强的过冲效应。
- **L11 的 max Recovery 最低（2.07）**，与该层因果效应整体衰减的趋势一致。

需要注意的是，max Recovery 对单个异常值非常敏感，不宜作为层间因果强度的稳健度量。更有参考价值的是 mean Recovery 在 $[0.3, 1]$ 区间内的特征数量（见第 3 节层级分布的讨论）。

### 6. Recovery 热力图（图 2 讨论）

![Recovery Heatmap](figures/recovery_heatmap_layer1.pdf)

该热力图展示了 L1（max Recovery 最高的层）中 top-50 特征在 73 个有效 prompt 上的 Recovery 分布。主要观察：

- **热力图极度稀疏**：填充率仅 15.6%（568/3650），即 top-50 特征中大多数在大多数 prompt 上并未被激活。这反映了 topk=32 稀疏度下特征激活的 prompt 特异性——不同 prompt 激活的特征集合差异很大。
- **因果效应高度集中**：少数 (feature, prompt) 对呈现强烈的暖色（高 Recovery），而大面积为空白（NaN，未激活）或冷色（低/负 Recovery）。这进一步证实了稀疏因果结构：IOI 信息被分散编码在不同 prompt 所激活的不同特征子集中。
- **特征激活频率分布不均**：L1 的 112 个唯一特征中，59 个仅在 $< 10$ 个 prompt 上激活，17 个在 $> 60$ 个 prompt 上激活。高频特征（如 Feature 246, 在 60/73 个 prompt 上激活）的 mean Recovery 较低（0.135），而低频特征（如 Feature 1932, 仅 3 个 prompt）反而具有最高的 mean Recovery（0.997）。这暗示 IOI 因果特征倾向于是 prompt 特异性的稀疏方向，而非普遍激活的通用方向。

### 7. 结论

实验证实了 VASAE 分解中存在对 IOI 任务具有因果效应的特征方向：

1. **因果效应稀疏且可量化**：大多数特征 Recovery 接近 0，少数特征（每层 1–31 个）具有 Recovery $\in [0.3, 1]$ 的显著因果效应，符合"少数方向承载关键信息"的稀疏可解释性假设。
2. **因果效应具有任务特异性**：高 Recovery 特征同时具有高 Specificity（如 L1 Feature 1932 的 Specificity=240），说明消融这些特征精准地改变了 IOI 决策，而非泛化性地破坏模型输出。
3. **层级分布提示信息流结构**：IOI 因果特征集中于前层和中层（L0–L7），后层因果效应衰减，暗示 IOI 的核心解析发生在模型的前半部分。
4. **因果特征具有 prompt 特异性**：高 Recovery 特征往往仅在少数 prompt 上激活，说明 IOI 信息被分散编码在不同的特征子集中，而非由一组固定特征统一承载。

局限性：Recovery $> 1$ 的少量过冲现象仍然存在，可能源于 SAE 重建误差在后续层的非线性放大，或特定 prompt 上的统计波动。完整的因果归因还需特征组合消融、与随机方向的对照等更精细的控制实验。
