# Exp
## Benchmarking VASAE

Objective: 验证 soft anchor 不损害重建质量，同时说明 hard tied 需要 soft anchor（hard tied 作为 ablation）。

Table: model {GPT-2, LLaMA3-8B}, method {SAE, VASAE (hard tied, ablation), VASAE (soft anchored)}, MSE, VE, CE, CE recovery, logitlens acc, mean cosine sim, strong alignment %, alive feature %

Figure: GPT-2 CE recovery 直方图. 横轴：layer index {L0–L11}, 纵轴：CE recovery，图例：raw / SAE / VASAE

Figure: LLaMA3-8B CE recovery 直方图. 横轴：layer index {L0–L31}, 纵轴：CE recovery，图例：raw / SAE / VASAE



## Feature Alignment Distribution

Objective: 证明 VASAE 分解出的 token-aligned features 具有稳定的功能分化，而非随机噪声。

Table: model {GPT-2, LLaMA3-8B}, layer {GPT-2:L0/L3/L6/L9/L11, LLaMA3:L0/L8/L16/L24/L31}, strong alignment (≥0.8), moderate alignment [0.5,0.8), weak alignment [0.3,0.5), unaligned (<0.3)

Figure: GPT-2 分解比例堆叠直方图. 横轴：layer index {L0–L11}, 纵轴：比例，图例：input-aligned / output-aligned / other

Figure: LLaMA3-8B 分解比例堆叠直方图. 横轴：layer index {L0–L31}, 纵轴：比例，图例：input-aligned / output-aligned / other

## Causal Intervention and  Specificity

Objective: 验证 output-aligned features 对任务目标具有因果贡献，且该贡献具有特异性（影响目标而非整体扰动）。

Table: model {GPT-2, LLaMA3-8B}, task {IOI, Fact Recall}, intervention {none, random feature, random output feature, VASAE target output feature}, task target score, task target drop, top-1 task accuracy, off-target KL, validation perplexity change

这里的 output-aligned features 是不可识别的，因此此实验不具备可操作性。让我们回到这个实验最初的目的。前面的实验已经证明 VASAE 可以分解为 token 切近的方向，但是否可以通过干预这些方向影响下游任务的效果（比如增加某些方向的信息，是得模型最终回答正确的 token；或者删除某些方向，以此抹除模型的部分知识，使其不输出某些有害信息）。概括来说，就是要证明这些分解出来的方向是具有因果效应的，而不仅仅是读出中间表示含有这些信息（相关性，不能操作）

# Appendix

## Soft Anchor Variants

Objective: 分析不同软对齐强度与形式对性能与对齐质量的影响，验证主文设置的合理性。

Table: model {GPT-2, LLaMA3-8B}, soft anchor type {cosine, L2, hybrid}, anchor strength, MSE, CE recovery, mean cosine sim, strong alignment %, alive feature %

Figure: GPT-2 soft anchor 强度曲线. 横轴：anchor strength，纵轴：CE recovery / alignment metrics

Figure: LLaMA3-8B soft anchor 强度曲线. 横轴：anchor strength，纵轴：CE recovery / alignment metrics


## Sparsity (L0) Analysis

Objective: 分析不同稀疏度对重建质量与对齐特性的影响。

Table: model {GPT-2, LLaMA3-8B}, L0, MSE, CE recovery, strong alignment %, alive feature %

Figure: GPT-2 sparsity–performance 曲线. 横轴：L0，纵轴：CE recovery / alignment metrics

Figure: LLaMA3-8B sparsity–performance 曲线. 横轴：L0，纵轴：CE recovery / alignment metrics

（必要：这是 SAE/VASAE 类方法的核心控制变量）


## Layer-wise Behavior

Objective: 补充展示关键指标随层变化的完整趋势（主文只展示部分）。

Figure: GPT-2 layer-wise 指标变化. 横轴：layer {L0–L11}，纵轴：CE recovery / alignment metrics

Figure: LLaMA3-8B layer-wise 指标变化. 横轴：layer {L0–L31}，纵轴：CE recovery / alignment metrics



## Comparison with Logit Lens

Objective: 对比 VASAE feature 与 logit lens 的表示差异，说明其并非简单重参数化。

Figure: GPT-2 VASAE vs logit lens 对比. 横轴：layer，纵轴：accuracy / cosine sim，图例：VASAE / logit lens

Figure: LLaMA3-8B VASAE vs logit lens 对比. 横轴：layer，纵轴：accuracy / cosine sim，图例：VASAE / logit lens



## Feature Visualization

Objective: 展示典型 feature 的行为模式，辅助理解其功能分化。

Figure: GPT-2 feature 可视化示例. 横轴：token / position，纵轴：activation / contribution

Figure: LLaMA3-8B feature 可视化示例. 横轴：token / position，纵轴：activation / contribution



## Failure Cases

Objective: 展示方法失效或不稳定的情况，界定适用范围。

Table: model {GPT-2, LLaMA3-8B}, case type {misalignment, unstable feature, ambiguous feature}, frequency, impact on CE recovery, impact on alignment



## Dead Features

Objective: 分析未激活或低利用率 feature 的比例及其影响。

Table: model {GPT-2, LLaMA3-8B}, method {SAE, VASAE}, dead feature %, alive feature %, CE recovery

Figure: dead feature 分布. 横轴：layer，纵轴：比例

