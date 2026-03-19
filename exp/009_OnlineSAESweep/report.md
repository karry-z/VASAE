# Experiment 009: Online SAE Sweep (train_sae_online)

## 实验目的

对 `scripts/train_sae_online.py` 做参数 sweep，观察在线抽激活训练下：
- 不同层（L0-L11）
- 不同稀疏度（TopK: k=8/16/32）
- 不同 anchor 强度（anchor_coeff=0 / 1e-4）

对重建质量与可解释性指标（Variance Explained / LogitLens / CE Loss Recovered）的影响。

## Sweep 配置

| 维度 | 取值 |
|---|---|
| model-name | gpt2 |
| layer-idx | 0~11 |
| k | 8, 16, 32 |
| anchor-coeff | 0, 1e-4 |
| tied-decoder | True |
| nonneg-latents | True |
| num-epochs | 5 |
| train/eval samples | 4000 / 1000 |

总任务数：$12 \times 3 \times 2 = 72$（SLURM array 0-71）。

## 运行方式

```bash
sbatch exp/009_p_OnlineSAESweep/run.sh
```

## 输出位置

- 训练输出根目录：`/scratch/b5bq/pu22650.b5bq/VASAE_out/009_online_sweep`
- 单任务目录命名：`009_online_gpt2_L{layer}_k{k}_a{anchor}`
- 日志目录：`exp/009_p_OnlineSAESweep/logs/`

## 后处理建议

训练脚本会在每个 epoch 打印并记录：
- `loss`
- `variance_explained`
- `logitlens_acc`
- `loss_recovered` (eval)

推荐后续再汇总 wandb 或日志，做一张 72 组配置的对比表。
