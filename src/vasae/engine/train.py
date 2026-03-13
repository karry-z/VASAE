from logging import Logger
from types import SimpleNamespace
from typing import Dict

import torch

from vasae.configs.train import TrainConfig
from vasae.metrics.interface import Aggregator, MetricComposer
from vasae.models.sae_hf import SAEConfig, SAEModel, SAEOutput


def train_one_epoch(
    model: SAEModel,
    loader,
    train_cfg: TrainConfig,
    device,
    optimizer: torch.optim.Optimizer,
    metrics: MetricComposer,
    logger: Logger,
    epoch: int,
):
    model.train()

    aggregator = Aggregator()
    for batch_i, data in enumerate(loader):
        activations = data["activations"]
        activations = activations.to(device)

        optimizer.zero_grad()

        output: SAEOutput = model(activations)
        decoded = output.hidden_states_recon
        eval_outcomes = metrics.compute({"data": activations, "decoded": decoded})
        # accumulate loss
        aggregator.add(
            {
                "loss": output.loss,
                "l1_loss": output.l1_loss,
                "loss_reconst": output.recon_loss,
                "logitlens_acc": eval_outcomes["logitlens_acc"],
                "loss_lowrank": output.loss_lowrank,
                "loss_anchor": output.loss_anchor,
            },
            activations.size(0),
        )

        output.loss.backward()
        optimizer.step()
        if logger is not None:
            logger.info(
                f"[Train] Epoch {epoch+1}/{train_cfg.num_epochs} "
                f"batch {batch_i+1}/{len(loader)} "
                f"loss {output.loss.item():.4f} "
                f"acc: {eval_outcomes['logitlens_acc']*100:.2f}%"
            )

        if train_cfg.max_batchsize > 0 and batch_i > train_cfg.max_batchsize:
            break

    return aggregator.compute()
