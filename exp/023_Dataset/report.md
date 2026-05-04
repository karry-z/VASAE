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

收集数据集：
```bash
sbatch exp/023_Dataset/fetch_corpus_slurm.sh
```

验证 JSONL、manifest、token 数和 split 边界
```bash
uv run --no-sync python scripts/collect/validate_corpus.py \
  --out-dir "$VASAE_OUT/Dataset/data"
```

快速 smoke test，确认探索脚本能跑通：
```bash
uv run --no-sync python scripts/training/train_sae_online.py \
  --data-source jsonl \
  --model-name gpt2 \
  --layer-idx 5 \
  --corpus-dir "$VASAE_OUT/Dataset/data" \
  --train-tokens 10000 \
  --valid-tokens 2000 \
  --train-batchsize 4 \
  --valid-batchsize 4 \
  --max-length 128 \
  --dim-sparse 50257 \
  --sparsity-type topk \
  --k 32 \
  --nonneg-latents \
  --anchor-coeff 1e-4 \
  --anchor-mode hard \
  --num-epochs 1 \
  --save-dir "$VASAE_OUT/Dataset/runs" \
  --exp-name "gpt2_L5_mixture_soft_smoke" \
  --no-wandb

for corpus in fineweb dclm pile; do
  uv run --no-sync python scripts/eval/eval_sae_online.py \
    --data-source jsonl \
    --sae-path "$VASAE_OUT/Dataset/runs/gpt2_L5_mixture_soft_smoke" \
    --model-name gpt2 \
    --layer-idx 5 \
    --corpus "$corpus" \
    --eval-tokens 2000 \
    --corpus-dir "$VASAE_OUT/Dataset/data" \
    --test-batchsize 4 \
    --max-length 128
done

uv run --no-sync python scripts/aggregate/summarize_dataset_results.py \
  --run-dir "$VASAE_OUT/Dataset/runs/gpt2_L5_mixture_soft_smoke"
```

完整探索实验：
```bash
sbatch exp/023_Dataset/run_train_eval_slurm.sh
```
默认仍是 GPT-2-small L5；如需复用同一入口测试其他支持的模型/层，可通过 `MODEL_NAME`、`LAYER_IDX`、`DTYPE` 和 `RUN_NAME` 覆盖 Slurm 环境变量。`DIM_SPARSE` 默认会从模型 vocab size 自动解析，也可以手动覆盖。

默认输出：
- checkpoint 和训练 sanity eval: `$VASAE_OUT/Dataset/runs/gpt2_L5_mixture_soft/results.json`
- held-out eval: `results_eval_fineweb.json`, `results_eval_dclm.json`, `results_eval_pile.json`
- 汇总: `summary.json`, `summary.md`


# Results

## 运行信息

本次完整探索作业为 `exp/023_Dataset/logs/023_train_eval_4414657.log`，输出目录：

```text
/projects/b5bq/VASAE/Dataset/runs/gpt2_L5_mixture_soft
```

实际语料收集完成情况：

| corpus | train tokens | train docs | heldout tokens | heldout docs |
|---|---:|---:|---:|---:|
| FineWeb | 66,666,901 | 95,905 | 1,000,447 | 1,453 |
| DCLM | 66,667,394 | 53,104 | 1,000,329 | 770 |
| The Pile | 66,666,757 | 45,181 | 1,035,254 | 32 |
| **total** | **200,001,052** | **194,190** | **3,036,030** | **2,255** |

训练配置：GPT-2-small L5，VASAE-Soft，`k=32`，`lambda=1e-4`，`max_length=128`，`batch_size=32`，`num_epochs=1`，训练 token budget 为 200M，训练期 sanity validation token budget 为 300k。

训练期指标：

| split | loss | MSE | VE | logitlens |
|---|---:|---:|---:|---:|
| train mixture | 0.9609 | 0.9610 | 0.9902 | 0.8388 |
| valid mixture sanity | 0.8649 | 0.8649 | 0.9912 | 0.8433 |

## Held-out Evaluation

每个 corpus 使用 1M token held-out budget 评估。结果文件为 `results_eval_fineweb.json`、`results_eval_dclm.json`、`results_eval_pile.json`，汇总文件为 `summary.json` 和 `summary.md`。

| corpus | MSE | VE | logitlens | CE recovered | dead_rate | L0 | n_alive |
|---|---:|---:|---:|---:|---:|---:|---:|
| FineWeb | 0.8324 | 0.9917 | 0.8566 | 0.9814 | 0.9142 | 31.9562 | 4,311 |
| DCLM | 0.8372 | 0.9914 | 0.8472 | 0.9791 | 0.9138 | 31.9578 | 4,331 |
| The Pile | 0.9065 | 0.9903 | 0.8341 | 0.9793 | 0.9145 | 31.9599 | 4,299 |

## Active Feature Overlap

| feature sets | intersection | union | jaccard |
|---|---:|---:|---:|
| FineWeb / DCLM | 4,274 | 4,368 | 0.9785 |
| FineWeb / The Pile | 4,260 | 4,350 | 0.9793 |
| DCLM / The Pile | 4,269 | 4,361 | 0.9789 |
| all three | 4,257 |  |  |

## Observations

混合训练后的重构和功能指标在三个 held-out split 上都比较稳定：VE 均约为 0.990-0.992，CE recovered 约为 0.979-0.981。The Pile 的 MSE 略高、logitlens 略低，符合其文本来源更异质的预期。

feature 活跃集合在三个 held-out split 上高度重叠，pairwise Jaccard 约为 0.979，三者共有 4,257 个 alive features。当前 1M-token held-out 评估没有显示出很强的数据集特异 active feature 子集；如果要继续验证低频/领域特征差异，下一步可以扩大 held-out token budget，或针对 The Pile 子域、代码/论文/论坛等来源做分组评估。
