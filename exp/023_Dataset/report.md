---
title: datasets
date: 20260427
---

# Purpose
使用 200M token 的 FineWeb/DCLM/The Pile balanced mixture 重新训练 GPT-2-small L5 的 VASAE-Soft，并在各数据集 held-out split 上分别评估重构和对齐。

目前的实验 VASAE-Soft 都是在 wiki-103 上训练测试的，但 wiki-103 数据比较脏，且token 总数很少。这将降低 feature 的需求量，无法表达更多低频词、复杂概念和特定模式。比如在代码语料上的需要表达的特征不一定在自然语言文本中出现。


# Methods

## 模型
为了快速验证，我们选择 GPT2 的中间实验，具体来说是第五层 (GPT2-Small L5)。我们只训练 VASAE-Soft k=32 lambda=1e-4.

**Datasets**
FineWeb + DCLM + The Pile 混合数据

### FineWeb

FineWeb 是 Hugging Face 发布的大规模英文网页预训练语料，来源于 96 个 Common Crawl dump，时间覆盖 2013 年夏季到 2024 年 4 月。官方数据集卡描述其经过清洗、过滤和去重，规模约为 15T GPT-2 tokens，后续版本统计约超过 18T tokens。FineWeb 的核心特点是规模极大、领域覆盖广，且主要保留通用网页文本分布，因此适合作为高质量 open web baseline。对本实验来说，FineWeb 可以检验 VASAE-Soft 在比 WikiText-103 更大、更自然的网页语料上是否能激活更多长尾 token/实体/格式相关特征。

Source: [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)

### DCLM

DCLM 指 DataComp for Language Models，是一个用于研究“数据质量如何影响语言模型训练”的数据基准。DCLM-Baseline 是该项目筛选出的高质量网页预训练语料，来自 Common Crawl，并通过系统化的数据过滤、去重和基准反馈进行构建。和 FineWeb 相比，DCLM 更强调 controlled data curation：模型结构和训练 recipe 尽量固定，让数据本身成为主要实验变量。因此 DCLM 适合用来测试 VASAE-Soft 在经过更强筛选的网页语料上是否表现不同，尤其是 feature alive ratio、重构质量和对齐覆盖率是否会受数据质量/过滤策略影响。

Source: [DataComp-LM](https://www.datacomp.ai/dclm/), [mlfoundations/dclm-baseline-1.0-parquet](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet)

### The Pile

The Pile 是 EleutherAI 构建的 825 GiB 英文语言模型预训练语料，由 22 个子数据集组成，覆盖学术论文、代码、网页、书籍、论坛、对话和专业文本等多种来源。它和 FineWeb/DCLM 的主要区别是异质性更强：The Pile 不是单纯的网页清洗语料，而是多来源混合语料，因此包含更多结构化文本、专业域文本和代码相关模式。对本实验来说，The Pile 可以用来检验“WikiText-103 缺少的低频概念、代码模式、领域特征”是否会在 VASAE-Soft 中诱导出额外活跃 feature。

Source: [The Pile paper](https://arxiv.org/abs/2101.00027), [Datasheet for the Pile](https://arxiv.org/abs/2201.07311)

### 数据集使用策略

本实验的第一阶段不分别在 FineWeb、DCLM 和 The Pile 上各训练一个 VASAE-Soft，而是将三者按 token 数做 balanced mixture 后共同训练一个 GPT-2-small L5 VASAE-Soft。具体来说，训练集计划使用总计 200M tokens，其中 FineWeb、DCLM、The Pile 各贡献约 1/3 token。这样可以避免某个大数据集按自然规模压倒其他数据集，同时让模型在同一个训练过程中接触通用网页、高质量筛选网页和多来源异质文本。

训练后，评估阶段不再混合，而是在三个数据集各自的 held-out split 上分别测试。这样可以区分两个问题：一是混合大语料训练是否能缓解 WikiText-103 训练下 feature 大量不激活的问题；二是不同数据分布是否会激活不同的 feature 子集。除了 MSE、VE、CE recovery 等重构指标外，本实验尤其关注 dead_rate、avg L0、每个数据集上的 alive feature set，以及不同数据集 active feature set 的 overlap。

这个设计比“每个数据集单独训练一个 SAE”更适合作为可行性验证，因为它只需要训练一个模型，成本较低，而且直接回答当前核心假设：WikiText-103 的小规模、窄分布是否限制了 VASAE-Soft 的 feature demand。如果混合训练后 dead_rate 明显下降，并且不同 held-out 数据集激活出不同 feature 子集，再进一步扩展为 train-on-FineWeb / train-on-DCLM / train-on-Pile / train-on-mixture 的完整 train-test matrix。



**Eval Metrics**
- MSE / VE: 几何重构质量
- CE loss / CE recovery: 功能重构质量
- dead_rate:
- avg l0:

# Workflow

收集数据集 大约 1 min
```bash
sbatch fetch_corpus_slurm.sh
```


# Results
