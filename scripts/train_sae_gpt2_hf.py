import argparse
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
from vasae.models.factory import (
    BlackBoxModelConfig,
    load_embeding_layer,
    load_unembeding_layer,
)
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

    # data config
    parser.add_argument("--train-batchsize", type=int, default=128)
    parser.add_argument("--valid-batchsize", type=int, default=128)
    parser.add_argument("--test-batchsize", type=int, default=128)
    parser.add_argument("--use-centralize", type=bool, default=True)
    parser.add_argument(
        "--layer-name",
        type=str,
        default="transformer.h.11",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=r"/scratch/b5bq/pu22650.b5bq/activations_gpt2_Geralt-Targaryen_openwebtext2",
    )

    # sae config
    parser.add_argument("--dim-input", type=int, default=768)
    parser.add_argument("--dim-sparse", type=int, default=50257)
    parser.add_argument("--encoder-type", type=str, default="linear")
    parser.add_argument("--sparsity-type", type=str, default="topk")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--per-item-in-eval", type=bool, default=False)
    parser.add_argument("--nonneg-latents", type=bool, default=True)
    parser.add_argument("--l1-coeff", type=float, default=0.0)
    parser.add_argument("--tied-decoder", type=bool, default=True)
    parser.add_argument("--mse-reduction", type=str, default="mean")
    parser.add_argument(
        "--sae-save-path",
        type=str,
        default=r"/scratch/b5bq/pu22650.b5bq/VASAE_out/sae.pth",
    )
    parser.add_argument("--freeze-decoder", type=bool, default=True)
    parser.add_argument("--use-lowrank", type=bool, default=True)
    parser.add_argument("--lowrank-coeff", type=float, default=0.1)

    # train config
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--max-batchsize", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)

    # system config
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", type=bool, default=True)
    parser.add_argument("--wandb-group", type=str, default="test")

    # blackbox model config
    parser.add_argument("--blackbox-model-name", type=str, default="gpt2")
    parser.add_argument(
        "--blackbox-model-dir",
        type=str,
        default=r"/scratch/b5bq/pu22650.b5bq/VASAE_out/BlackBoxModels/gpt2",
    )

    # exp name
    parser.add_argument("--exp-name", type=str, required=True)

    return parser.parse_args()


def parse_data_cfg(args) -> DataConfig:
    return DataConfig(
        train_batchsize=args.train_batchsize,
        valid_batchsize=args.valid_batchsize,
        test_batchsize=args.test_batchsize,
        use_centralize=args.use_centralize,
        layer_name=args.layer_name,
        data_dir=args.data_dir,
    )


def parse_sae_cfg(args) -> SAEConfig:
    return SAEConfig(
        dim_input=args.dim_input,
        dim_sparse=args.dim_sparse,
        encoder_type=args.encoder_type,
        sparsity_type=args.sparsity_type,
        k=args.k,
        per_item_in_eval=args.per_item_in_eval,
        nonneg_latents=args.nonneg_latents,
        l1_coeff=args.l1_coeff,
        tied_decoder=args.tied_decoder,
        mse_reduction=args.mse_reduction,
        sae_save_path=args.sae_save_path,
        freeze_decoder=args.freeze_decoder,
        use_lowrank=args.use_lowrank,
        lowrank_coeff=args.lowrank_coeff,
    )


def parse_train_cfg(args) -> TrainConfig:
    return TrainConfig(
        num_epochs=args.num_epochs,
        max_batchsize=args.max_batchsize,
        lr=args.lr,
    )


def parse_system_cfg(args) -> dict:
    return {
        "device": args.device,
        "seed": args.seed,
        "wandb": args.wandb,
        "wandb_group": args.wandb_group,
    }


def parse_blackbox_model_cfg(args) -> BlackBoxModelConfig:
    return BlackBoxModelConfig(
        name=args.blackbox_model_name,
        dir=Path(args.blackbox_model_dir) if args.blackbox_model_dir else None,
    )


def main():
    args = parse_args()

    data_cfg = parse_data_cfg(args)
    sae_cfg = parse_sae_cfg(args)
    train_cfg = parse_train_cfg(args)
    system_cfg = parse_system_cfg(args)
    blackbox_model_cfg = parse_blackbox_model_cfg(args)
    exp_name = args.exp_name

    device = torch.device(system_cfg["device"])
    set_seed(system_cfg["seed"])

    # prepare_paths
    model_path = Path(sae_cfg.sae_save_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    logger = get_logger()

    train_loader, valid_loader, test_loader = get_dataloader(
        data_cfg, system_cfg["seed"]
    )

    # Load model layers from pretrained blackbox model
    emb = load_embeding_layer(blackbox_model_cfg)
    unemb = load_unembeding_layer(blackbox_model_cfg)

    vocab_size, model_dim = emb.weight.shape

    sae_cfg.dim_input = model_dim
    sae_cfg.dim_sparse = vocab_size
    logger.info(vars(args))
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
            name=exp_name,
            group=system_cfg["wandb_group"],
            config=vars(args),
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
