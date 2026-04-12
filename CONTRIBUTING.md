# 贡献指南

感谢你对 VASAE（Vocab-Aligned Sparse Auto-Encoder）项目的关注！VASAE 是一个研究项目，旨在训练解码器与 token 词表嵌入对齐的稀疏自编码器，以实现对神经网络激活值的可解释潜在表示。

我们欢迎以下类型的贡献：新 SAE 变体、分析脚本、评估任务、文档改进、测试补充、Bug 修复。

## 开发环境搭建

**前置条件：** Python 3.12 + [uv](https://docs.astral.sh/uv/) 包管理器

```bash
git clone <repo-url>
cd VASAE
uv sync                # 安装依赖
uv run pytest tests/   # 运行测试
```

**代码风格：** 使用 **Black** 默认配置 + **isort**（`--profile black`）格式化。鼓励类型注解但不强制，遵循已有代码模式。

```bash
uv run black src/ scripts/ tests/
uv run isort --profile black src/ scripts/ tests/
```

**测试：** 使用 **pytest**，测试文件命名 `tests/test_*.py`，新增模型或 metric 需编写对应测试。

## 项目结构

```
VASAE/
├── src/vasae/          # 核心库
│   ├── models/         # SAE 模型（sae.py, encoders, sparsity, factory）
│   ├── analysis/       # 分析工具（alignment, hooks, stats, I/O）
│   ├── data/           # 数据集与数据 schema
│   ├── engine/         # 训练与评估循环
│   ├── metrics/        # 评估指标（logit lens, variance explained 等）
│   └── utils/          # 工具函数
├── scripts/            # 入口脚本（训练、数据收集、分析、评估、可视化）
├── tests/              # pytest 测试
├── exp/                # 实验目录（SLURM 作业脚本与日志）
└── notebooks/          # Jupyter notebooks
```

## HPC / SLURM 实验流程

### 实验目录命名规范

实验目录位于 `exp/`，所有实验统一使用三位数编号 + 名称。如需插入变体实验，在编号后追加字母：

- 探索性实验：`编号_名称`，如 `001_SaeFeatureVocabIdentible/`
- 正式实验：`F编号_名称`，如 `F012_TgeoMeaning/`（`F` 标记正式实验）

```
exp/
├── 001_SaeFeatureVocabIdentible/
├── 002_WeakTokenAnchoringRegularizer/
├── 007A_FineLambdaSweep/       # 007 的变体实验
├── 007B_FineLambdaSweepV2/
├── F012_TgeoMeaning/          # 正式实验
└── ...
```

### 实验目录内容

每个实验目录应包含：

- **`report.md`** — 实验报告，实验开始前先写好规划（目的、方法、结果（预期形式，比如设计描述清楚表头或图线的横纵轴）），实验完成后填入结果与结论
- **`run.sh`** — SLURM 作业脚本
- **`logs/`** — 运行日志

参考 `exp/001_SaeFeatureVocabIdentible/` 作为模板。

### 作业提交与输出

```bash
cd exp/013_YourExperiment/
sbatch run.sh
```

实验输出（模型 checkpoint、结果文件等）统一存储在 `/scratch/b5bq/pu22650.b5bq/VASAE_out`，避免将大文件存放在项目仓库内。

## 提交规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：`feat:`、`fix:`、`refactor:`、`docs:`、`test:`

```
feat: 添加 BatchTopK 稀疏模块
fix: 修复 logit lens 在 batch_size=1 时的维度错误
refactor: 将 decoder 初始化逻辑移入 factory
```

PR 标题应简洁明了，描述中说明改动目的和测试结果。优先复用已有代码和模式。

## 提交新实验

1. 在 `exp/` 下创建目录，按命名规范编号（参见上方实验目录命名规范）
2. 编写 `report.md`：明确实验目的、假设、方法、预期结果形式（表头、图线横纵轴等）
3. 编写 `run.sh` SLURM 作业脚本，输出路径指向 `/scratch/b5bq/pu22650.b5bq/VASAE_out`
4. 如需新模型或 metric，先提交对应的代码变更（见下方），再提交实验
5. 实验完成后在 `report.md` 中补充结果与结论

## 添加新 SAE 变体

1. **`src/vasae/models/`** — 定义新模型类，参考已有的 `sae.py`
2. **`src/vasae/models/factory.py`** — 注册工厂函数
3. **`scripts/`** — 添加训练脚本，遵循已有 argparse 模式
4. **`tests/`** — 编写单元测试

## 添加分析 / 评估脚本

脚本放置于 `scripts/`，遵循已有的 argparse 模式。新的评估指标应实现 `IMetric` 接口（`src/vasae/metrics/`），参考 `logitlens.py` 和 `variance_explained.py`。

## Bug 报告与功能请求

**Bug 报告**请提供：问题描述、复现步骤、完整 traceback、环境信息（Python 版本、CUDA 版本、GPU 型号等）。

**功能请求**请说明：想要实现的功能、使用场景与动机，如果可能提供初步的实现思路。
