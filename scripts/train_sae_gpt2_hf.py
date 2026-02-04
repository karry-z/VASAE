import argparse
import runpy
from logging import Logger
from pathlib import Path

import torch
import torch.optim as optim

import wandb
from vasae.configs.data import DataConfig
from vasae.configs.train import TrainConfig
from vasae.data.dataset import get_dataloader
from vasae.engine import evaluate, train
from vasae.metrics.interface import MetricComposer
from vasae.metrics.logitlens import LogitLens, LogitLensAccuracy, LogitLensMetric
from vasae.models.factory import load_embeding_layer, load_unembeding_layer
from vasae.models.sae_hf import SAEConfig, SAEModel, SAEOutput
from vasae.utils.log import get_logger
from vasae.utils.seed import set_seed


def train_model(
    model: SAEModel,
    *,
    optimizer: optim.Optimizer,
    train_cfg: TrainConfig,
    train_loader,
    valid_loader,
    logger,
    metrics: MetricComposer,
    device,
):

    for epoch in range(train_cfg.num_epochs):
        # train
        train_out = train.train_one_epoch(
            model=model,
            loader=train_loader,
            train_cfg=train_cfg,
            device=device,
            optimizer=optimizer,
            metrics=metrics,
            logger=logger,
            epoch=epoch,
        )

        eval_out = evaluate.evaluate(
            model=model,
            data_loader=valid_loader,
            metrics=metrics,
            device=device,
            logger=logger,
        )

        wandb.log(
            {
                **{f"train/{k}": v for k, v in train_out.items()},
                **{f"valid/{k}": v for k, v in eval_out.items()},
            },
        )


def parse_args():
    parser = argparse.ArgumentParser()

    # logging / save
    parser.add_argument(
        "--log",
        type=str,
        required=True,
    )

    # config
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = runpy.run_path(args.config)
    data_cfg: DataConfig = cfg["data_cfg"]
    sae_cfg: SAEConfig = cfg["sae_cfg"]
    train_cfg: TrainConfig = cfg["train_cfg"]
    system_cfg = cfg["system_cfg"]

    device = torch.device(system_cfg["device"])
    set_seed(system_cfg["seed"])

    # prepare_paths
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = Path(sae_cfg.sae_save_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    logger = get_logger(log_path)
    logger.info(vars(args))

    train_loader, valid_loader, test_loader = get_dataloader(
        data_cfg, system_cfg["seed"]
    )

    emb = load_embeding_layer(cfg["blackbox_model_cfg"])
    unemb = load_unembeding_layer(cfg["blackbox_model_cfg"])

    vocab_size, model_dim = emb.weight.shape

    sae_cfg.dim_input = model_dim
    sae_cfg.dim_sparse = vocab_size
    model = SAEModel(sae_cfg).to(device)
    if sae_cfg.tied_decoder:
        model.attach_embedding(emb, freeze=sae_cfg.freeze_decoder)

    # train
    logitlens = LogitLens(unemb)
    logitlens_acc = LogitLensAccuracy()

    metrics = MetricComposer([LogitLensMetric(logitlens, logitlens_acc)])

    # wandb
    if system_cfg["wandb"]:
        wandb.init(
            project="VASAE",
            name=cfg["exp_name"],
            group=system_cfg.get("wandb_group", None),
            config={"cfg_path": args.config},
        )
    else:
        wandb.init(mode="disabled")

    optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr)
    logger.info(f"use Adam with lr={train_cfg.lr}")

    train_model(
        model,
        optimizer=optimizer,
        train_cfg=train_cfg,
        train_loader=train_loader,
        valid_loader=valid_loader,
        logger=logger,
        metrics=metrics,
        device=system_cfg["device"],
    )

    # save model
    torch.save(model.state_dict(), model_path)
    logger.info(f"save model in {model_path}")

    # test
    outcome = evaluate.evaluate(
        model=model,
        data_loader=test_loader,
        metrics=metrics,
        device=system_cfg["device"],
        logger=logger,
    )

    logger.info(
        f"[Test] "
        f"loss {outcome["loss_reconst"]:.4f} ± {outcome["loss_reconst_std"]:.4f} "
        f"acc: {outcome["acc"] * 100:.2f}% "
    )

    wandb.log(
        {
            "test/loss_recons": outcome["loss_reconst"],
            "test/loss_recons_std": outcome["loss_reconst_std"],
            "test/acc": outcome["acc"],
        }
    )

    wandb.finish()


if __name__ == "__main__":
    main()
