import argparse
import runpy
from gc import disable
from pathlib import Path

import torch
import torch.optim as optim

import wandb
from vasae.configs.data import DataConfig
from vasae.configs.train import TrainConfig
from vasae.data.dataset import get_dataloader
from vasae.metrics.interface import MetricComposer
from vasae.metrics.logitlens import LogitLens, LogitLensAccuracy, LogitLensMetric
from vasae.models.factory import get_blackbox_model
from vasae.models.sae_hf import SAEConfig, SAEModel, SAEOutput
from vasae.utils.log import get_logger
from vasae.utils.seed import set_seed


def train_model(
    model: SAEModel,
    *,
    train_cfg: TrainConfig,
    train_loader,
    valid_loader,
    logger,
    metrics: MetricComposer,
    device,
):
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr)
    logger.info(f"use Adam with lr={train_cfg.lr}")

    for epoch in range(train_cfg.num_epochs):
        model.train()

        for batch_i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            output: SAEOutput = model(data)

            output.loss.backward()
            optimizer.step()
            decoded = output.hidden_states_recon

            train_eval_outcomes = metrics.compute({"data": data, "decoded": decoded})

            logger.info(
                f"[Train] Epoch {epoch+1}/{train_cfg.num_epochs} "
                f"batch {batch_i+1}/{len(train_loader)} "
                f"loss {output.loss.item():.4f} "
                f"acc: {train_eval_outcomes["logitlens_acc"] * 100:.2f}% "
            )

            wandb.log(
                {
                    "train/loss": output.loss,
                    "train/loss_recons": output.recon_loss,
                    "train/loss_l1": output.l1_loss,
                    "train/acc": train_eval_outcomes["logitlens_acc"],
                }
            )

            if train_cfg.max_batchsize > 0 and batch_i >= train_cfg.max_batchsize:
                logger.debug(f"break at batch {batch_i}")
                break

        model.eval()

        with torch.no_grad():
            for batch_i, data in enumerate(valid_loader):
                data = data.to(device)
                output = model(data)
                decoded = output.hidden_states_recon

                valid_eval_outcomes = metrics.compute(
                    {"data": data, "decoded": decoded}
                )
                logger.info(
                    f"[Valid] Epoch {epoch+1}/{train_cfg.num_epochs} "
                    f"batch {batch_i+1}/{len(valid_loader)} "
                    f"loss {output.loss.item():.4f} "
                    f"acc: {valid_eval_outcomes["logitlens_acc"] * 100:.2f}% "
                )

                wandb.log(
                    {
                        "valid/loss": output.loss,
                        "valid/loss_recons": output.recon_loss,
                        "valid/loss_l1": output.l1_loss,
                        "valid/acc": train_eval_outcomes["logitlens_acc"],
                    }
                )

                if train_cfg.max_batchsize > 0 and batch_i >= train_cfg.max_batchsize:
                    logger.debug(f"break at batch {batch_i}")
                    break


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

    logger = get_logger(args.log)
    logger.info(vars(args))

    # prepare_paths
    model_path = Path(sae_cfg.sae_save_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    train_loader, valid_loader, _ = get_dataloader(
        data_cfg.meta_path,
        data_cfg.layer_name,
        train_bs=data_cfg.train_batchsize,
        test_bs=data_cfg.valid_batchsize,
        use_centralize=data_cfg.use_centralize,
    )

    blackbox_model, _ = get_blackbox_model(
        cfg["blackbox_model_cfg"]["model_name"],
        device,
    )

    vocab_size, model_dim = blackbox_model.transformer.wte.weight.shape

    sae_cfg.dim_input = model_dim
    sae_cfg.dim_sparse = vocab_size
    model = SAEModel(sae_cfg).to(device)
    model.attach_embedding(blackbox_model.transformer.wte, freeze=True)

    logger.info(f"model {type(model).__name__} loaded")

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
        wandb.init(mode="disable")

    train_model(
        model,
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

    wandb.finish()


if __name__ == "__main__":
    main()
