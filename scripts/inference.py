import argparse
import runpy
from gc import disable
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

import wandb
from vasae.configs.data import DataConfig
from vasae.configs.train import TrainConfig
from vasae.data.dataset import get_dataloader
from vasae.engine.evaluate import evaluate
from vasae.metrics.interface import MetricComposer
from vasae.metrics.logitlens import LogitLens, LogitLensAccuracy, LogitLensMetric
from vasae.models.factory import get_blackbox_model
from vasae.models.sae_hf import SAEConfig, SAEModel, SAEOutput
from vasae.utils.log import get_logger
from vasae.utils.seed import set_seed


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

    _, _, test_loader = get_dataloader(data_cfg, system_cfg["seed"])

    blackbox_model, _ = get_blackbox_model(
        cfg["blackbox_model_cfg"]["model_name"],
        device,
    )

    vocab_size, model_dim = blackbox_model.transformer.wte.weight.shape

    sae_cfg.dim_input = model_dim
    sae_cfg.dim_sparse = vocab_size
    model = SAEModel(sae_cfg)
    if sae_cfg.tied_decoder:
        model.attach_embedding(
            blackbox_model.transformer.wte, freeze=sae_cfg.freeze_decoder
        )

    # load model
    model.load_state_dict(torch.load(sae_cfg.sae_save_path))
    model = model.to(device)

    # train
    logitlens = LogitLens(blackbox_model.lm_head)
    logitlens_acc = LogitLensAccuracy()

    metrics = MetricComposer([LogitLensMetric(logitlens, logitlens_acc)])

    # wandb
    if system_cfg["wandb"]:
        wandb.init(
            project="VASAE", name=cfg["exp_name"], config={"cfg_path": args.config}
        )
    else:
        wandb.init(mode="disabled")

    outcome = evaluate(
        model=model,
        data_loader=test_loader,
        metrics=metrics,
        device=system_cfg["device"],
        logger=logger,
    )

    logger.info(
        f"[Test] "
        f"loss {outcome["loss_reconst"]:.4f} ± {outcome["loss_reconst_std"]:.4f}"
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
