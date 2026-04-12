---
title: <英文短语标题>
date: <YYYY-MM-DD>
---

# 目的

<2–4 段。说明：(a) 紧接哪个前序实验 FXXX 的结论或未解问题；(b) 本实验要验证 / 回答什
么；(c) 为什么值得做。禁止放方法细节。>

# 方法

<!-- 若涉及数学符号，先放 Notation 定义；不需要可整段删除 -->

### Notation 定义

| 符号 | 含义 |
| --- | --- |
| $x$ | <含义> |
| $y$ | <含义> |

## 实验一：<子实验名>

<一段动机 + 设计描述。>

固定 <参数>，sweep：

$$<变量> \in \{ \text{<候选值列表>} \}$$

代表层 / 代表配置：<列表>。共 <X> × <Y> = **<N> tasks**。

<!-- 若有第二个子实验，复制下面这一节并改名 -->

## 实验二：<子实验名>

<同上。>

## 共享配置

- 模型：<gpt2 / meta-llama/Llama-3.1-8B / ...>
- 精度：<float32 / bfloat16>
- batch size：<train / eval>
- dataset：<WikiText-103 / ...>，max_length=<...>
- 切分：train/eval/test = <.../.../...>
- optimizer：Adam (lr=<...>)，max <N> epochs，early stopping (patience=<N>)
- 其它：<dim_sparse、decoder tied/untied、sparsity 模块、anchor 配置 ...>

## 评估指标

- **VE**（Variance Explained）：归一化重构质量
- **CE Recovery**：功能性重构质量
- **Dead Feature Rate**：测试集上激活次数为 0 的 feature 占比
- **L0**：每个输入的平均非零激活 feature 数
- <如有自定义指标，按"**缩写**：一句定义"格式补充>

# 流程

```bash
# 0. <前置依赖步骤，例如先验证一个超参>
sbatch exp/FXXX_Name/<run_pre.sh>

# 1. <主实验，可并行>
sbatch exp/FXXX_Name/<run_gpt2.sh>
sbatch exp/FXXX_Name/<run_llama.sh>

# 2. 汇总结果
uv run python scripts/<collect_xxx.py> \
    --results-dir /scratch/b5bq/pu22650.b5bq/VASAE_out/<FXXX_Name> \
    --output-dir exp/FXXX_Name
```

# 结果

## 实验一：<子实验名>

> <如有复用：† 标注的行复用 FXXX_<Other> 结果，该批次未记录 <字段名>。>
>
> 缺失：<列出失败 / 未跑的 run 及原因>。

### GPT-2

<可选一段引言。>

| layer | <sweep变量> | VE | CE Recovery | Dead Rate |
| -- | -- | -- | -- | -- |
| 0 | <值> | <数值> | <数值> | <数值> |
| ... | ... | ... | ... | ... |

![alt text](figures/<exp1_gpt2_stable_name>.png)

（数据来源：`exp/FXXX_Name/<subdir>/results.json`，对应 `<field_path>` 字段）

### Llama-3.1-8B

| layer | <sweep变量> | VE | CE Recovery | Dead Rate |
| -- | -- | -- | -- | -- |
| 0 | <值> | <数值> | <数值> | <数值> |
| ... | ... | ... | ... | ... |

![alt text](figures/<exp1_llama_stable_name>.png)

（数据来源：`exp/FXXX_Name/<subdir>/results.json`，对应 `<field_path>` 字段）

<!-- 若有第二个子实验，复制 ## 实验一 整段并改名 -->

## 实验二：<子实验名>

### GPT-2

| layer | <sweep变量> | <指标1> | <指标2> | <指标3> |
| -- | -- | -- | -- | -- |
|  |  |  |  |  |

![alt text](figures/<exp2_gpt2_stable_name>.png)

（数据来源：`exp/FXXX_Name/<subdir>/results.json`，对应 `<field_path>` 字段）

### Llama-3.1-8B

| layer | <sweep变量> | <指标1> | <指标2> | <指标3> |
| -- | -- | -- | -- | -- |
|  |  |  |  |  |

![alt text](figures/<exp2_llama_stable_name>.png)

（数据来源：`exp/FXXX_Name/<subdir>/results.json`，对应 `<field_path>` 字段）

# 分析

### 实验一：<子实验名>

<段落只做解读：与前序实验是否一致 / 冲突，异常值的可能原因，限制条件。禁止重复结果数值。>

**结论**：<一句结论>。

### 实验二：<子实验名>

<同上。>

**结论**：<一句结论>。
