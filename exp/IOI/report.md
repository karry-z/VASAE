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

需要强调的是，该实验只能证明 VASAE 分解中存在具有因果效应的特征方向，但并不能解决表示的识别性问题。换言之，即便某些特征在事后被验证为因果性的，也不意味着当前分解在数学上是唯一的，或这些特征可以在无干预验证的情况下被可靠识别。因此，该结果更应被理解为对“因果结构存在性”的实证支持，而非对“可解释分解完备性”的充分证明。

# 结果
## 表：代表性特征的干预效果

<!-- Auto-generated: paste from exp/IOI/figures/top_features_table.md after running plot_ioi_feature_sweep.py -->

| Feature ID | Layer | Feature Strength | $\Delta u_{\text{clean}}$ | $\Delta u_{\text{corr}}$ | $\Delta u_{\text{clean, intervened}}$ | Effect | Recovery | Specificity Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (pending sweep results) | - | - | - | - | - | - | - | - |

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

附录：

## 图：$\max_{f,p} \text{Recovery}(l,f,p)$ 随 $l$ 的曲线

![Max Recovery vs Layer](figures/max_recovery_vs_layer.pdf)

## 图：上一个图最高的 $l$，$ \text{Recovery}(f,p)$ 随 $f,p$ 的热力图

![Recovery Heatmap](figures/recovery_heatmap_layer8.pdf)

<!-- Note: figure filename assumes best layer is 8; update if sweep results differ -->