---
title: benchmarking
date: 
---

# 目的
验证 VASAE 的重构质量。SAE 使用过完备字典中的少量特征拟合一个神经网络原始中间表征。这些特征能高质量重构原始中间表征，是后续分析、使用这些特征的前提。因此我们关注重构质量，包括原始特征和重构特征在几何上是否接近，以及功能上是否保真。功能保真意味着使用重构中间表示继续前向传播，神经网络会输出同样的预测分布，即重构没有破坏表征的计算功能。


# 方法

为验证 VASAE 在不同规模语言模型上的适用性，我们选取了 GPT-2-small（hidden dim 768，vocab size 50257，12 层）和 Llama-3.1-8B（hidden dim 4096，vocab size 128256，32 层）分别代表经典的小型语言模型和现代大型语言模型。由于不同层之间的表征特性存在显著差异，为探究 VASAE 的适用边界，例如是否仅在浅层有效，我们对每个模型的所有层逐一训练和评估。

VASAE 的核心思想是将 SAE 的 decoder features 与语言模型的 token vocabulary 对齐，使每个 latent 天然对应一个 token 方向，从而获得可解释性。实现对齐有两种方式：一是 **hard tying**，将 decoder 权重直接绑定为 token embedding 矩阵，稀疏维度等于词表大小；二是 **soft anchoring**，decoder 权重保持独立可学习，但通过正则项 $\mathcal{L}_{\text{anchor}} = -\frac{1}{m}\sum_{i} \max_j \cos(d_i, h^{(0)}_j)$ 鼓励每个 decoder feature 靠近至少一个 token embedding 方向。本实验的核心目标是验证 VASAE-Soft（soft anchoring）在重构质量上可以接近甚至达到标准 SAE 的水平，同时保留词表对齐的可解释性优势。

三种方案的具体配置如下：

| | Plain SAE | VASAE-Hard | VASAE-Soft |
| -- | -- | -- | -- |
| decoder | 独立可学习 | 绑定为 token embedding 矩阵（frozen） | 独立可学习 |
| dim_sparse | vocab_size（GPT-2: 50257, Llama: 128256） | vocab_size（GPT-2: 50257, Llama: 128256） | vocab_size（GPT-2: 50257, Llama: 128256） |
| anchor loss | 无 | 无（decoder 已绑定，无需正则） | $\lambda_{\text{anchor}} = 10^{-4}$，mode = hard max |

VASAE-Soft 的超参数基于前序预实验（Exp 007 FineLambdaSweep、Exp 010 SoftAlignSweep）的结论选定：anchor 系数 $\lambda_{\text{anchor}} = 10^{-4}$，anchor 模式为 hard max（Exp 008 验证其优于 logsumexp/softmax），稀疏维度等于词表大小。所有方案共享相同的训练配置：线性 encoder，TopK 稀疏（k=32），nonneg latents，Adam 优化器（学习率 1e-3），训练最多 20 个 epoch，采用早停策略（patience=3，即基于验证集 loss允许模型在 连续 3 个评估轮次里都没有取得更好的验证集 loss，训练才会停止），最终以 best model 在测试集上评估。训练数据来自 WikiText-103，序列最大长度 128。GPT2 上的训练按 50000/10000/5000 条样本划分为训练集、验证集和测试集（约 6.4M/1.3M/0.6M tokens）；llama 为了快速训练，我们使用 20000/2000/5000。为避免 padding 位置的无意义 activations 干扰 SAE 训练和指标计算，所有 special token（padding）位置在送入 SAE 前被 attention mask 过滤。Activations 通过 nnsight 在线提取，所有指标（包括 CE Loss 和 CE Recovery）均在前传过程中直接计算。GPT-2 以 float32 精度训练，batch size 为 32；Llama-3.1-8B 以 bfloat16 精度训练，batch size 为 8（受显存限制）。
<!-- WikiText-103只是方便获取的较小数据集，没有什么强的动机一定要使用它 -->
<!-- 检查 llama 的其他 special token是否也被 mask 掉了 -->

**Evaluation Metrics**：我们从几何和功能两个维度评估重构质量。

几何指标方面，MSE（$\|x - \hat{x}\|^2$）衡量重构 activations 与原始 activations 的欧氏距离；Variance Explained（$\text{VE} = 1 - \text{MSE} / \text{Var}(x)$）对 MSE 进行归一化，消除不同层 activations 尺度差异的影响，更适合跨层比较。

功能指标方面，CE Loss 将重构后的 activations 替换原始 activations 继续前传，计算下一词预测的交叉熵损失，直接衡量重构对模型功能的影响；CE Recovery（$1 - (\text{CE}_{sae} - \text{CE}_{id}) / (\text{CE}_{zero} - \text{CE}_{id})$）以 identity（不做替换）和 zero ablation 为参照对 CE Loss 归一化，值为 1 表示完美恢复模型功能；LogitLens Accuracy 通过 unembedding 矩阵 $W_U$ 比较重构前后在当前层的 token 预测是否一致（$\mathbb{1}[\arg\max W_U \hat{x} = \arg\max W_U x]$），该指标无需完整前传，计算开销较低。

# 流程

```bash
# 1. 提交 GPT-2 实验（12 层 × 3 变体 = 36 tasks） 变体：plain, hard, soft
sbatch exp/F001_Benchmarking/run_gpt2.sh

# 2. 提交 Llama-3.1-8B 实验（32 层 × 3 变体 = 96 tasks）
sbatch exp/F001_Benchmarking/run_llama.sh

# 3. 汇总结果（生成 per-layer CSV 和 summary CSV）
uv run python scripts/aggregate/collect_benchmarking_results.py \
    --results-dir /scratch/b5bq/pu22650.b5bq/VASAE_out/001_F_Benchmarking \
    --output-dir exp/F001_Benchmarking
```

每个 task 的原始结果存储为 `{save-dir}/{exp-name}/results.json`。汇总脚本扫描所有 `results.json`，解析实验名称（格式 `001F_{model}_L{layer}_{variant}`），输出逐层结果 `results_per_layer.csv` 和按 model × variant 分组的均值 ± 标准差汇总 `results_summary.csv`。

# 结果

以下结果是 mean ± std 的格式，mean 和 std 是指在 layer 上做的聚合

| model | method | MSE | VE | CE loss | CE recovery | logitlens acc |
| -- | -- | -- | -- | -- | -- | -- |
| GPT-2 | Plain SAE | 2.04 ± 2.78 | 0.965 ± 0.053 | 4.18 ± 0.11 | 0.975 ± 0.028 | 0.879 ± 0.031 |
| GPT-2 | VASAE-Hard | 110.39 ± 62.53 | -0.434 ± 1.046 | 14.64 ± 10.65 | -0.606 ± 2.642 | 0.832 ± 0.057 |
| GPT-2 | VASAE-Soft | 2.04 ± 2.78 | 0.965 ± 0.054 | 4.17 ± 0.11 | 0.975 ± 0.028 | 0.878 ± 0.031 |
| Llama-3.1-8B | Plain SAE | 0.057 ± 0.096 | 0.931 ± 0.091 | 3.55 ± 0.68 | 0.906 ± 0.075 | 0.351 ± 0.093 |
| Llama-3.1-8B | VASAE-Hard | 0.672 ± 0.236 | -0.021 ± 0.058 | 11.46 ± 0.92 | 0.032 ± 0.101 | 0.009 ± 0.009 |
| Llama-3.1-8B | VASAE-Soft | 0.058 ± 0.097 | 0.931 ± 0.091 | 3.55 ± 0.67 | 0.906 ± 0.074 | 0.351 ± 0.093 |

## 分析
### VASAE-Hard 在两个模型上均失败

Hard tying（decoder = frozen embedding matrix）在两个模型上均严重失败：

- **GPT-2**: VE -0.434（负值，重建比均值还差）, CE Recovery -0.606
- **Llama**: VE -0.021, CE Recovery 0.032 (接近 0，意味着 ce(sae)和 ce(zero)接近，也就是重构后和置零预测下一个 token 没啥区别)

token embedding matrix 的几何结构主要为词表预测服务，不是为中间表征重构设计的。将 decoder 强制绑定为该矩阵后，模型只能在一个固定且与激活分布不完全匹配的基上做稀疏重构；在 TopK 和 nonneg 约束下，这会显著削弱表达能力，因此重构质量会明显下降。这个结果更支持“hard tying 的 inductive bias 不适合重构任务”，而不是“token embedding 本身不可能重构 hidden states”。

### 核心结论：VASAE-Soft 不损害重建质量

在两个模型上，VASAE-Soft 与 Plain SAE 的所有指标几乎完全一致：

- **GPT-2**: VE 0.965 vs 0.965, CE Recovery 0.975 vs 0.975, LogitLens 0.878 vs 0.879
- **Llama**: VE 0.931 vs 0.931, CE Recovery 0.906 vs 0.906, LogitLens 0.351 vs 0.351

差异在千分位级别（< 0.1%），说明 soft anchor loss（λ=1e-4）对重建能力几乎没有代价。这符合预实验 Exp 002/007 的发现：弱 anchor 是"free lunch"。验证了这 soft anchoring 的必要性——需要允许 decoder 自由调整来兼顾重建，同时通过正则项引导对齐。



### Llama 的 LogitLens Accuracy 显著低于 GPT-2

GPT-2 的 LogitLens Accuracy ~0.88，Llama 仅 ~0.35。这不是 SAE 质量问题，而是 logit lens 在 Llama 上本身就不太 work——Llama 使用 RMSNorm 且深层表示与 unembedding space 的对齐弱于 GPT-2。CE Recovery（0.906）是更可靠的功能性指标，表明 Llama 上的重建仍然很好。

### 早停行为

GPT-2 的 plain 和 soft 变体均跑满 20 个 epoch（未触发早停），说明这些模型在 20 epoch 时仍在缓慢改善。Llama 的部分层在 8-16 epoch 触发早停，可能因为 Llama 的 hidden state 更简单（RMSNorm 后方差小），SAE 更快收敛。Hard 变体早停最早（patience=3 很快触发，因为 loss 一直不降）。
results.json里面的 stopped_epoch 字段记录了早停的内容,如 [001F_gpt2_L0_hard](file:///scratch/b5bq/pu22650.b5bq/VASAE_out/001_F_Benchmarking/001F_gpt2_L0_hard/results.json) 结果

