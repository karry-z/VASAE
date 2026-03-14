"""Offline SAE training on pre-extracted activations (memmap).

Works with any model whose activations have been collected beforehand
(e.g. via collect_gpt2_activations.py or collect_llava_activations.py).
"""

import argparse
from pathlib import Path

import torch
import torch.optim as optim

import wandb
from vasae.data.schema import DataConfig
from vasae.engine.config import TrainConfig
from vasae.data.dataset import get_dataloader
from vasae.engine import evaluate, train
from vasae.metrics.base import MetricComposer
from vasae.metrics.logitlens import LogitLens, LogitLensAccuracy, LogitLensMetric
from vasae.models.factory import (
    BlackBoxModelConfig,
    load_embedding_layer,
    load_unembedding_layer,
)
from vasae.models.sae import SAEConfig, SAEModel
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
            max_batchsize=train_cfg.max_batchsize,
        )

        wandb.log(
            {
                **{f"train/{k}": v for k, v in train_out.items()},
                **{f"valid/{k}": v for k, v in eval_out.items()},
            },
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline SAE training on pre-extracted activations"
    )

    # data config
    parser.add_argument("--train-batchsize", type=int, default=128)
    parser.add_argument("--valid-batchsize", type=int, default=128)
    parser.add_argument("--test-batchsize", type=int, default=128)
    parser.add_argument("--use-centralize", action="store_true")
    parser.add_argument("--layer-name", type=str, required=True,
                        help="Layer name in meta.json (e.g. transformer.h.11)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with meta.json and activation .dat files")

    # sae config
    parser.add_argument("--dim-input", type=int, default=768)
    parser.add_argument("--dim-sparse", type=int, default=50257)
    parser.add_argument("--encoder-type", type=str, default="linear")
    parser.add_argument("--sparsity-type", type=str, default="topk")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--per-item-in-eval", action="store_true")
    parser.add_argument("--nonneg-latents", action="store_true")
    parser.add_argument("--l1-coeff", type=float, default=0.0)
    parser.add_argument("--no-tied-decoder", action="store_true")
    parser.add_argument("--mse-reduction", type=str, default="mean")
    parser.add_argument(
        "--sae-save-path", type=str,
        default="/scratch/b5bq/pu22650.b5bq/VASAE_out/sae.pth",
    )
    parser.add_argument("--no-freeze-decoder", action="store_true")
    parser.add_argument("--use-lowrank", action="store_true")
    parser.add_argument("--lowrank-coeff", type=float, default=0.1)
    parser.add_argument("--use-abs-topk", action="store_true")
    parser.add_argument("--anchor-coeff", type=float, default=0.0)
    parser.add_argument(
        "--anchor-mode", type=str, default="hard",
        choices=["hard", "logsumexp", "softmax"],
    )
    parser.add_argument("--anchor-topk", type=int, default=10)
    parser.add_argument(
        "--random-anchor", type=str, default="none",
        choices=["none", "shuffle", "gaussian"],
    )

    # train config
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--max-batchsize", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)

    # system config
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-group", type=str, default="offline")

    # blackbox model config
    parser.add_argument("--blackbox-model-name", type=str, default="gpt2")
    parser.add_argument("--blackbox-model-dir", type=str, required=True,
                        help="Directory with emb.pth and unemb.pth")

    # exp name
    parser.add_argument("--exp-name", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    data_cfg = DataConfig(
        train_batchsize=args.train_batchsize,
        valid_batchsize=args.valid_batchsize,
        test_batchsize=args.test_batchsize,
        use_centralize=args.use_centralize,
        layer_name=args.layer_name,
        data_dir=args.data_dir,
    )
    sae_cfg = SAEConfig(
        dim_input=args.dim_input,
        dim_sparse=args.dim_sparse,
        encoder_type=args.encoder_type,
        sparsity_type=args.sparsity_type,
        k=args.k,
        per_item_in_eval=args.per_item_in_eval,
        nonneg_latents=args.nonneg_latents,
        l1_coeff=args.l1_coeff,
        tied_decoder=not args.no_tied_decoder,
        mse_reduction=args.mse_reduction,
        sae_save_path=args.sae_save_path,
        freeze_decoder=not args.no_freeze_decoder,
        use_lowrank=args.use_lowrank,
        lowrank_coeff=args.lowrank_coeff,
        use_abs_topk=args.use_abs_topk,
        anchor_coeff=args.anchor_coeff,
        anchor_mode=args.anchor_mode,
        anchor_topk=args.anchor_topk,
    )
    train_cfg = TrainConfig(
        num_epochs=args.num_epochs,
        max_batchsize=args.max_batchsize,
        lr=args.lr,
    )
    bbm_cfg = BlackBoxModelConfig(
        name=args.blackbox_model_name,
        dir=Path(args.blackbox_model_dir),
    )

    device = torch.device(args.device)
    set_seed(args.seed)

    model_path = Path(sae_cfg.sae_save_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    logger = get_logger()

    train_loader, valid_loader, test_loader = get_dataloader(data_cfg, args.seed)

    # Load model layers from pretrained blackbox model
    emb = load_embedding_layer(bbm_cfg).float()
    unemb = load_unembedding_layer(bbm_cfg).float()

    vocab_size, model_dim = emb.weight.shape
    sae_cfg.dim_input = model_dim
    sae_cfg.dim_sparse = vocab_size
    logger.info(vars(args))
    model = SAEModel(sae_cfg).to(device)

    if sae_cfg.tied_decoder:
        model.attach_embedding(emb, freeze=sae_cfg.freeze_decoder)

    if sae_cfg.anchor_coeff > 0 and not sae_cfg.tied_decoder:
        if args.random_anchor == "shuffle":
            perm = torch.randperm(emb.weight.size(0))
            anchor_emb = torch.nn.Embedding.from_pretrained(emb.weight[perm], freeze=True)
        elif args.random_anchor == "gaussian":
            rand_w = torch.randn_like(emb.weight)
            rand_w = rand_w / rand_w.norm(dim=1, keepdim=True) * emb.weight.norm(dim=1, keepdim=True)
            anchor_emb = torch.nn.Embedding.from_pretrained(rand_w, freeze=True)
        else:
            anchor_emb = emb
        model.attach_anchor_embedding(anchor_emb)
        if args.random_anchor != "none":
            random_emb_path = Path(sae_cfg.sae_save_path).parent / "random_emb.pt"
            torch.save(anchor_emb.weight.data, random_emb_path)
            logger.info(f"Saved random anchor embedding to {random_emb_path}")

    # metrics
    logitlens = LogitLens(unemb)
    logitlens_acc = LogitLensAccuracy()
    metrics = MetricComposer([LogitLensMetric(logitlens, logitlens_acc)])

    # wandb
    if not args.no_wandb:
        wandb.init(
            project="VASAE", name=args.exp_name,
            group=args.wandb_group, config=vars(args),
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
        device=args.device,
    )

    # save model
    torch.save(model.state_dict(), model_path)
    logger.info(f"save model in {model_path}")

    # test
    outcome = evaluate.evaluate(
        model=model,
        data_loader=test_loader,
        metrics=metrics,
        device=args.device,
        logger=logger,
        max_batchsize=train_cfg.max_batchsize,
    )

    logger.info(
        f"[Test] "
        f"loss={outcome['loss']:.4f} "
        f"recon={outcome['loss_reconst']:.4f} "
        f"acc={outcome['logitlens_acc'] * 100:.2f}% "
        f"lowrank={outcome.get('loss_lowrank', 0):.4f}"
    )

    wandb.log({f"test/{k}": v for k, v in outcome.items()})
    wandb.finish()


if __name__ == "__main__":
    main()
