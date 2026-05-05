---
title: alive feature alignment
date: 20260505
---

# Purpose

评估 mixed-corpus GPT-2 VASAE-Soft 在测试语料上真正被使用到的 feature 是否保持 vocab alignment。

之前的 geometric alignment 指标默认以整个 dictionary 为分母：

$$
\frac{\#\{i: s_i \ge 0.8\}}{\#\text{features}}
$$

其中 $s_i = \max_v \cos(d_i, e_v)$，$d_i$ 是 SAE decoder feature，$e_v$ 是 GPT-2 token embedding。这个指标会把测试语料中完全没有被激活的 dead features 也算入分母。对于大词表维度的 VASAE（GPT-2: 50257 features），测试语料只能覆盖其中一部分 feature，因此我们更关心 **alive feature alignment rate**：

$$
\frac{\#\{i: i \text{ is alive and } s_i \ge 0.8\}}{\#\{i: i \text{ is alive}\}}
$$

Alive feature 指在测试语料的 valid token positions 上至少激活一次的 feature；dead feature 指测试期间从未激活的 feature。本实验回答的问题是：在 FineWeb、DCLM、The Pile 各自 held-out split 中，被模型实际使用到的 feature 有多大比例是 strongly vocab-aligned 的？

# Methods

## Checkpoints

使用 `exp/F001_Benchmarking/run_mix_gpt2_train.sh` 训练完成的 GPT-2 mixed-corpus VASAE-Soft checkpoints：

```text
/projects/b5bq/VASAE/F001_Benchmarking_mix/001Fmix_gpt2_L{layer}_soft
```

覆盖层：

```text
L0, L3, L6, L9, L11
```

训练配置为 GPT-2-small，`dim_sparse=50257`，TopK `k=32`，nonneg latents，soft anchor `lambda=1e-4`，anchor mode 为 hard max。训练数据为 FineWeb + DCLM + The Pile balanced mixture。

## Test Corpora

评估阶段不混合语料，而是在每个 corpus 的 held-out split 上分别计算 alive feature set：

| corpus | split | token budget |
|---|---|---:|
| FineWeb | `fineweb/raw/heldout.jsonl` | 1,000,000 |
| DCLM | `dclm/raw/heldout.jsonl` | 1,000,000 |
| The Pile | `pile/raw/heldout.jsonl` | 1,000,000 |

默认语料目录：

```text
/projects/b5bq/VASAE/Dataset/data
```

## Metric

对每个 `layer x corpus`：

1. 加载 SAE checkpoint 和 GPT-2 token embedding matrix $W_E$。
2. 计算每个 decoder feature 的 geometric alignment score：

$$
s_i = \max_v \cos(d_i, e_v)
$$

3. 对 held-out corpus 流式提取对应层 activations，送入 SAE encoder。
4. 使用 attention mask 过滤 padding position；若 feature activation `z_i > 0` 至少出现一次，则 feature 为 alive。
5. 使用阈值 `alignment_threshold = 0.8` 统计 strongly aligned features。
6. 输出：

| field | meaning |
|---|---|
| `n_features` | dictionary size, GPT-2 为 50257 |
| `n_alive` | held-out corpus 激活过的 feature 数 |
| `n_aligned` | 全 dictionary 中 `alignment_score >= 0.8` 的 feature 数 |
| `n_alive_aligned` | alive features 中 `alignment_score >= 0.8` 的 feature 数 |
| `alive_alignment_rate` | `n_alive_aligned / n_alive` |
| `dead_rate` | `1 - n_alive / n_features` |

# Workflow

## Full Slurm Run

提交 15 个 task：

```bash
sbatch exp/F001_Benchmarking/run_mix_gpt2_alive_alignment.sh
```

Slurm array mapping：

```text
5 layers x 3 corpora = 15 tasks
layers  = 0, 3, 6, 9, 11
corpora = fineweb, dclm, pile
```

每个 task 输出一个 JSON：

```text
exp/F001_Benchmarking/alive_alignment/gpt2_mix/L{layer}/{corpus}.json
```

完整 array 完成后汇总 CSV 和图：

```bash
uv run --no-sync python scripts/analyze/alignment/eval_alive_alignment.py \
  --output-dir exp/F001_Benchmarking/alive_alignment/gpt2_mix \
  --aggregate-only
```

汇总输出：

```text
exp/F001_Benchmarking/alive_alignment/gpt2_mix/alive_alignment_per_layer.csv
exp/F001_Benchmarking/alive_alignment/gpt2_mix/alive_alignment_rate_by_layer.png
exp/F001_Benchmarking/alive_alignment/gpt2_mix/alive_alignment_rate_by_layer.pdf
```

## Smoke Test

在 GPU 节点上可用小 token budget 跑单层单语料 smoke test：

```bash
uv run --no-sync python scripts/analyze/alignment/eval_alive_alignment.py \
  --results-dir /projects/b5bq/VASAE/F001_Benchmarking_mix \
  --model-name gpt2 \
  --layer-idx 0 \
  --corpus fineweb \
  --corpus-dir /projects/b5bq/VASAE/Dataset/data \
  --eval-tokens 2048 \
  --alignment-threshold 0.8 \
  --output-dir exp/F001_Benchmarking/alive_alignment/gpt2_mix_smoke \
  --batch-size 32 \
  --max-length 128 \
  --force
```

预期检查：

- `tokens_processed > 0`
- `0 <= alive_alignment_rate <= 1`
- `n_alive_aligned <= n_alive`
- `n_alive <= n_features`
- 生成 `alive_alignment_per_layer.csv` 和 `alive_alignment_rate_by_layer.{png,pdf}`

# Results

当前状态：analysis script、Slurm runner、unit tests 已实现；完整 GPU evaluation 尚未在本地环境运行。本地环境 `nvidia-smi` 无法连接 driver，因此需要在集群 GPU 节点上提交 Slurm array。

结果完成后，将从以下文件汇总：

```text
exp/F001_Benchmarking/alive_alignment/gpt2_mix/alive_alignment_per_layer.csv
```

建议报告表格格式：

| layer | FineWeb alive align % | DCLM alive align % | The Pile alive align % | FineWeb n_alive | DCLM n_alive | The Pile n_alive |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | pending | pending | pending | pending | pending | pending |
| 3 | pending | pending | pending | pending | pending | pending |
| 6 | pending | pending | pending | pending | pending | pending |
| 9 | pending | pending | pending | pending | pending | pending |
| 11 | pending | pending | pending | pending | pending | pending |

# Notes

- 这个指标和 F002 的全 dictionary geometric alignment 不同：这里分母只包含测试语料实际激活过的 alive features。
- 如果某层的 `dead_rate` 很高，全 dictionary alignment rate 可能低估实际使用特征的可解释性。
- 如果三个 corpus 的 alive alignment rate 接近，说明 mixed training 学到的 activated feature 子集在不同 held-out distribution 上具有相似的 vocab alignment quality。
- 如果 The Pile 明显不同，可能说明异质/专业文本会激活一批 alignment quality 不同的 feature，后续可以按 Pile 子域继续拆分。
