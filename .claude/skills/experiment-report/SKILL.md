---
name: experiment-report
description: This skill should be used when the user asks to "写实验报告", "写 FXXX 的 report", "新建实验报告", "整理 results.json 成 report", "把这次跑的结果写成报告", "draft an experiment report", or otherwise wants to author/update a `exp/FXXX_Name/report.md` in the VASAE project. Provides the fixed 6-section skeleton, table conventions, data-source citation rules, and a copy-ready template.
---

# VASAE 实验报告撰写规范

VASAE 项目中每个实验目录 `exp/FXXX_Name/` 都包含一份 `report.md`，并遵循统一的结构，
方便复现、互审和交流。本技能负责在新建或更新这类报告时强制执行该格式。

**核心原则**：一个 `report.md` 只负责一个实验主题；多主题应拆分为 `FXXX_A_Name`、
`FXXX_B_Name` 等多份报告。

## 何时触发

- 用户说"写 / 新建 FXXX 的实验报告"、"整理 results.json 成 report"。
- 用户说"把这次跑的结果写成 report"、"补一下 exp/FXXX_Name 的 report.md"。
- 用户给出一个 `exp/FXXX_Name/` 目录并要求基于其中的 `results.json` / `figures/` 出报告。

## 工作流

1. **确认元信息**：向用户确认实验目录名 `FXXX_Name`（F + 序号 + 可选字母 + 下划线 + 短
   驼峰，例如 `F001A_AblationSoft`、`F002_AlignmentAnalysis`），以及报告 `title`（英
   文短语）和 `date`（绝对日期 `YYYY-MM-DD`，若用户给"今天/昨天"需转换）。
2. **复制骨架**：读 `references/report-template.md`，写入 `exp/FXXX_Name/report.md`。
   不要新建任何无关文件。
3. **填充内容**：
   - 若 `exp/FXXX_Name/<subdir>/results.json` 存在，读取并按结构填表。
   - 若数据缺失，留 `—` 占位并在小节顶部用 `> 缺失：...` blockquote 注明原因（超时 /
     CUDA error / 任务未创建 / 复用旧实验）。
   - 图片若不存在，插入占位 `![alt text](figures/<stable_name>.png)`，**不要**伪造数据
     或编造跑数。
4. **撰写分析**：每个子实验一段分析，禁止重复结果数值，只做解读；段末以 `**结论**：`
   一句作结。若证据不足，写 `TODO: 待补充` 而不是猜测。
5. **自审**：对照下文"检查清单"逐项核对。

## 固定 6 段骨架

每份 `report.md` **必须**按顺序包含以下 6 段（标题用词固定，便于 grep）：

| # | 标题 | 用途 | 关键内容 |
|---|------|------|----------|
| 0 | YAML frontmatter | 元信息 | `title`, `date` |
| 1 | `# 目的` | 动机 | 前序实验链 / 待回答的问题 / 为什么重要。**禁止**写方法细节 |
| 2 | `# 方法` | 实验设计 | 子实验子节、Notation（可选）、`## 共享配置`、`## 评估指标` |
| 3 | `# 流程` | 复现命令 | 可直接 `bash` 粘贴的 `sbatch` / `uv run python` 块 |
| 4 | `# 结果` | 客观数据 | 每子实验一节，按模型拆 `###`，pipe-table + 图 + 数据源 |
| 5 | `# 分析` | 解读 | 每子实验一段，**结论** 句作结 |

> 同义标题允许：第 3 段也可写 `# 实验步骤`，第 5 段也可写 `# 分析与讨论`。新报告默认用
> `# 流程` 和 `# 分析` 以保持一致性。

### 第 2 段 `# 方法` 的子结构

- 若实验涉及数学符号：先放 `### Notation 定义` 表（两列：符号 / 含义）。
- 每个子实验单独一个 `##` 或 `###`，并写明：sweep 的变量列表（用 `$\{...\}$` 集合记法）、
  固定参数、task 总数（`X × Y = N tasks`）。
- 末尾两节固定为 `## 共享配置` 和 `## 评估指标`。共享配置列出：模型 / 精度 / batch
  size / dataset / 切分 / optimizer / lr / max epoch / early stopping。评估指标用一句
  话定义每个缩写（VE = Variance Explained，CE Recovery = 功能性重构质量，等）。

### 第 4 段 `# 结果` 的写法

每个子实验一个 `##`，内部按模型分 `###`（GPT-2 / Llama-3.1-8B / ...）。每个 `###`：

1. 可选一段引言。
2. 标准 Markdown pipe-table，列顺序固定为 `layer | <sweep变量> | <指标1> | <指标2> | ...`。
3. 表格紧接 `![alt text](figures/<stable_name>.png)`。图片放在 `exp/FXXX_Name/figures/`。
4. **每个表 / 数字段落必须写出数据源**，格式：

   > （数据来源：`exp/FXXX_Name/<subdir>/results.json`，对应 `<field_path>` 字段）

   field_path 用 JSON 点路径，例如 `geometric.aligned_pct`、`input_correlation.mean_rho`。
5. 复用旧实验的行用 `†` 上标 + 小节顶部 blockquote 解释（例如
   `> † 标注的行复用 F001_Benchmarking 结果`）。
6. 缺失 / 失败的 run 用 `> 缺失：...` blockquote 列出。

### 第 5 段 `# 分析` 的写法

- 每子实验一个 `###`。
- 段落只做解读：与前序实验一致 / 冲突，异常值原因，限制条件。
- **禁止重复结果数值**。
- 段末用 `**结论**：...` 一句作结。

## 写作约定

- **路径**：项目代码 / 报告内引用使用项目相对路径（`exp/...`, `scripts/...`）；大数据
  路径用绝对 `/scratch/b5bq/pu22650.b5bq/...`；checkpoint 用绝对路径。
- **缺失值**：表格用 `—`；复用值用 `†`；失败原因写在 blockquote 里。
- **数据源**：每个结果数字 / 表格段落必须有 `（数据来源：...）` 引用，**不能省**。
- **图片**：所有图片放在 `exp/FXXX_Name/figures/`；alt text 自由；不要在报告里嵌 base64。
- **指标缩写**：首次出现时哪怕放在 `## 评估指标` 节也要一句话展开。
- **结论句**：分析段必须以 `**结论**：` 句作结，单独一行或段尾。
- **复现命令**：必须可直接 `bash` 粘贴执行；用真实脚本路径和参数；按依赖顺序排列。
- **进度条**：脚本输出严禁包含 tqdm / 进度条（项目规则，CLAUDE.md）。
- **Logger**：脚本若被引用，提醒使用 `shared_utils.log` 而非 `print`（项目规则）。
- **数学**：行内 `$...$`，独立公式 `$$...$$`；定义 `\newcommand` 集中放在 Notation 节
  顶部 `$$...$$` 块里。
- **中英文**：正文中文为主，专有名词 / 指标名 / 命令保留英文；标点中英混排时英文标点
  后留半空格。

## 检查清单

完成报告后逐项核对：

- [ ] YAML frontmatter 含 `title` 和 `date`（绝对日期）
- [ ] 6 段标题齐全且顺序正确
- [ ] 一份报告只覆盖一个实验主题
- [ ] `# 方法` 末尾包含 `## 共享配置` 和 `## 评估指标`
- [ ] `# 流程` 中所有命令可直接 bash 执行
- [ ] 每个结果表 / 数字段都有 `（数据来源：...）` 引用
- [ ] 缺失值用 `—`，复用行用 `†` 并在 blockquote 注明
- [ ] 失败 / 缺失 run 列在 blockquote `> 缺失：...`
- [ ] 图片路径都在 `exp/FXXX_Name/figures/`
- [ ] 每个分析子节末尾有 `**结论**：` 句
- [ ] 分析段不重复结果数值，只做解读
- [ ] 没有伪造未跑的实验数据；缺信息用 `TODO`

## 资源

- **`references/report-template.md`** —— 完整 6 段空白骨架，新建报告时直接复制并填空。
