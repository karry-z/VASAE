"""
Three-stage training pipeline for Sparse + PCA Decomposition SAE.

Stage 1: Train initial sparse encoder (standard VASAE with tied decoder)
Stage 2: Compute PCA of residuals (h - z_s @ E.T)
Stage 3: End-to-end retrain DecomposeSAEModel with frozen E and W
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

import wandb
from vasae.configs.data import DataConfig
from vasae.configs.train import TrainConfig
from vasae.data.dataset import get_dataloader
from vasae.engine import evaluate, train
from vasae.metrics.interface import Aggregator, MetricComposer
from vasae.metrics.logitlens import LogitLens, LogitLensAccuracy, LogitLensMetric
from vasae.models.decompose_sae import DecomposeSAEModel, DecomposeSAEOutput
from vasae.models.factory import (
    BlackBoxModelConfig,
    load_embeding_layer,
    load_unembeding_layer,
)
from vasae.models.sae_hf import SAEConfig, SAEModel
from vasae.utils.log import get_logger
from vasae.utils.seed import set_seed


# ---- Stage 1: Train initial sparse SAE ----

def run_stage1(
    train_loader,
    valid_loader,
    emb,
    unemb,
    train_cfg,
    sae_cfg,
    device,
    logger,
    output_path: Path,
):
    logger.info("=== Stage 1: Train initial sparse SAE ===")

    model = SAEModel(sae_cfg).to(device)
    if sae_cfg.tied_decoder:
        model.attach_embedding(emb, freeze=sae_cfg.freeze_decoder)

    logitlens = LogitLens(unemb)
    logitlens_acc = LogitLensAccuracy()
    metrics = MetricComposer([LogitLensMetric(logitlens, logitlens_acc)])

    optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr)

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
        wandb.log({
            **{f"stage1_train/{k}": v for k, v in train_out.items()},
            **{f"stage1_valid/{k}": v for k, v in eval_out.items()},
        })

    torch.save(model.state_dict(), output_path)
    logger.info(f"Stage 1 model saved to {output_path}")
    return model


# ---- Stage 2: PCA of residuals ----

@torch.no_grad()
def run_stage2(
    model: SAEModel,
    train_loader,
    device,
    logger,
    output_path: Path,
    max_batchsize: int = 0,
):
    logger.info("=== Stage 2: Compute PCA of residuals ===")
    model.eval()

    dim = model.config.dim_input
    sum_r = torch.zeros(dim, device=device, dtype=torch.float64)
    sum_rrt = torch.zeros(dim, dim, device=device, dtype=torch.float64)
    n = 0

    for batch_i, data in enumerate(train_loader):
        h = data["activations"].to(device)
        # flatten to (N, dim)
        h_flat = h.view(-1, dim)

        _, z_s = model.encode(h_flat)
        h_sparse = model.decoder(z_s)  # z_s @ E.T (use_lowrank=False)

        r = (h_flat - h_sparse).to(torch.float64)
        sum_r += r.sum(0)
        sum_rrt += r.T @ r
        n += r.shape[0]

        if max_batchsize > 0 and batch_i > max_batchsize:
            break

    mean_r = sum_r / n
    C = sum_rrt / n - mean_r.outer(mean_r)

    eigenvalues, eigenvectors = torch.linalg.eigh(C.float())
    # eigh returns ascending order, flip to descending
    eigenvalues = eigenvalues.flip(0)
    W_full = eigenvectors.flip(1)  # [dim, dim], descending by eigenvalue

    torch.save({
        "W_full": W_full.cpu(),
        "mean_r": mean_r.float().cpu(),
        "eigenvalues": eigenvalues.cpu(),
    }, output_path)

    logger.info(f"PCA components saved to {output_path}")
    logger.info(f"Top 10 eigenvalues: {eigenvalues[:10].tolist()}")
    return W_full, mean_r.float(), eigenvalues


# ---- Stage 3: End-to-end retrain ----

def train_stage3_one_epoch(
    model: DecomposeSAEModel,
    loader,
    optimizer,
    train_cfg: TrainConfig,
    metrics: MetricComposer,
    device,
    logger,
    epoch: int,
):
    model.train()
    aggregator = Aggregator()

    for batch_i, data in enumerate(loader):
        h = data["activations"].to(device)
        optimizer.zero_grad()

        out: DecomposeSAEOutput = model(h)

        eval_outcomes = metrics.compute({"data": h, "decoded": out.h_recon})

        aggregator.add({
            "loss": out.loss,
            "logitlens_acc": eval_outcomes["logitlens_acc"],
        }, h.size(0))

        out.loss.backward()
        optimizer.step()

        if logger is not None:
            logger.info(
                f"[Stage3 Train] Epoch {epoch+1}/{train_cfg.num_epochs} "
                f"batch {batch_i+1}/{len(loader)} "
                f"loss {out.loss.item():.4f} "
                f"acc: {eval_outcomes['logitlens_acc']*100:.2f}%"
            )

        if train_cfg.max_batchsize > 0 and batch_i > train_cfg.max_batchsize:
            break

    return aggregator.compute()


@torch.no_grad()
def eval_stage3(
    model: DecomposeSAEModel,
    loader,
    metrics: MetricComposer,
    device,
    logger,
    max_batchsize: int = 0,
):
    model.eval()
    aggregator = Aggregator()

    for batch_i, data in enumerate(loader):
        h = data["activations"].to(device)

        out: DecomposeSAEOutput = model(h)
        eval_outcomes = metrics.compute({"data": h, "decoded": out.h_recon})

        aggregator.add({
            "loss": out.loss.detach().cpu().item(),
            "logitlens_acc": eval_outcomes["logitlens_acc"],
        }, h.size(0))

        logger.info(f"{batch_i}/{len(loader)}")

        if max_batchsize > 0 and batch_i > max_batchsize:
            break

    return aggregator.compute()


@torch.no_grad()
def compute_test_metrics(
    model: DecomposeSAEModel,
    test_loader,
    device,
    logger,
    max_batchsize: int = 0,
):
    """Compute VE and VE_sparse on test set."""
    model.eval()

    sum_mse_full = 0.0
    sum_mse_sparse = 0.0
    sum_var = 0.0
    n = 0

    # first pass: compute mean of h
    sum_h = torch.zeros(model.dim_input, device=device, dtype=torch.float64)
    total_n = 0
    for batch_i, data in enumerate(test_loader):
        h = data["activations"].to(device)
        h_flat = h.view(-1, model.dim_input).to(torch.float64)
        sum_h += h_flat.sum(0)
        total_n += h_flat.shape[0]
        if max_batchsize > 0 and batch_i > max_batchsize:
            break
    mean_h = (sum_h / total_n).float()

    # second pass: compute MSE and variance
    for batch_i, data in enumerate(test_loader):
        h = data["activations"].to(device)
        h_flat = h.view(-1, model.dim_input)

        out: DecomposeSAEOutput = model(h_flat)

        # full reconstruction MSE
        sum_mse_full += F.mse_loss(out.h_recon, h_flat, reduction="sum").item()
        # sparse-only reconstruction: h_sparse + bias
        h_sparse_recon = out.h_sparse + model.bias
        sum_mse_sparse += F.mse_loss(h_sparse_recon, h_flat, reduction="sum").item()
        # variance
        sum_var += F.mse_loss(h_flat, mean_h.unsqueeze(0).expand_as(h_flat), reduction="sum").item()
        n += h_flat.shape[0] * model.dim_input

        if max_batchsize > 0 and batch_i > max_batchsize:
            break

    mse_full = sum_mse_full / n
    mse_sparse = sum_mse_sparse / n
    var_h = sum_var / n

    ve = 1.0 - mse_full / var_h
    ve_sparse = 1.0 - mse_sparse / var_h

    logger.info(f"VE={ve:.4f}, VE_sparse={ve_sparse:.4f}, MSE_full={mse_full:.6f}, MSE_sparse={mse_sparse:.6f}")
    return {
        "ve": ve,
        "ve_sparse": ve_sparse,
        "mse_full": mse_full,
        "mse_sparse": mse_sparse,
        "var_h": var_h,
    }


def run_stage3(
    emb,
    unemb,
    W_full,
    d_pca,
    k,
    train_loader,
    valid_loader,
    test_loader,
    train_cfg,
    device,
    logger,
    output_dir: Path,
):
    logger.info(f"=== Stage 3: End-to-end retrain with d_pca={d_pca} ===")

    dim_input = emb.weight.shape[1]
    dim_sparse = emb.weight.shape[0]

    model = DecomposeSAEModel(dim_input, dim_sparse, d_pca, k).to(device)
    model.attach_embedding(emb, freeze=True)
    W = W_full[:, :d_pca].to(device)
    model.attach_pca(W)

    logitlens = LogitLens(unemb)
    logitlens_acc = LogitLensAccuracy()
    metrics = MetricComposer([LogitLensMetric(logitlens, logitlens_acc)])

    # only optimize trainable params (sparse_encoder, dense_encoder, bias)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=train_cfg.lr)
    logger.info(f"Trainable params: {sum(p.numel() for p in trainable_params)}")

    for epoch in range(train_cfg.num_epochs):
        train_out = train_stage3_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            train_cfg=train_cfg,
            metrics=metrics,
            device=device,
            logger=logger,
            epoch=epoch,
        )
        eval_out = eval_stage3(
            model=model,
            loader=valid_loader,
            metrics=metrics,
            device=device,
            logger=logger,
            max_batchsize=train_cfg.max_batchsize,
        )
        wandb.log({
            **{f"stage3_train/{k_}": v for k_, v in train_out.items()},
            **{f"stage3_valid/{k_}": v for k_, v in eval_out.items()},
        })

    # test metrics
    test_out = eval_stage3(
        model=model,
        loader=test_loader,
        metrics=metrics,
        device=device,
        logger=logger,
        max_batchsize=train_cfg.max_batchsize,
    )
    ve_metrics = compute_test_metrics(
        model, test_loader, device, logger, max_batchsize=train_cfg.max_batchsize
    )

    results = {**test_out, **ve_metrics, "d_pca": d_pca, "k": k}
    wandb.log({f"test/{k_}": v for k_, v in results.items()})

    # save
    torch.save(model.state_dict(), output_dir / "model.pth")
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Stage 3 model saved to {output_dir / 'model.pth'}")
    logger.info(f"Results: {results}")
    return model, results


# ---- CLI ----

def parse_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--train-batchsize", type=int, default=128)
    parser.add_argument("--valid-batchsize", type=int, default=128)
    parser.add_argument("--test-batchsize", type=int, default=128)
    parser.add_argument("--use-centralize", action="store_true")
    parser.add_argument("--layer-name", type=str, default="transformer.h.11")
    parser.add_argument(
        "--data-dir", type=str,
        default=r"/scratch/b5bq/pu22650.b5bq/activations_gpt2_Geralt-Targaryen_openwebtext2",
    )

    # decompose SAE
    parser.add_argument("--d-pca", type=int, required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--sparsity-type", type=str, default="topk")
    parser.add_argument("--nonneg-latents", action="store_true")
    parser.add_argument("--use-abs-topk", action="store_true")

    # training
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--stage1-epochs", type=int, default=None,
                        help="Epochs for stage 1 (default: same as --num-epochs)")
    parser.add_argument("--max-batchsize", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)

    # output
    parser.add_argument("--output-dir", type=str, required=True)

    # system
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-group", type=str, default="decompose")
    parser.add_argument("--exp-name", type=str, required=True)

    # blackbox model
    parser.add_argument("--blackbox-model-name", type=str, default="gpt2")
    parser.add_argument(
        "--blackbox-model-dir", type=str,
        default=r"/scratch/b5bq/pu22650.b5bq/VASAE_out/BlackBoxModels/gpt2",
    )

    # stage caching: shared directory for stage1+2 outputs (per layer)
    parser.add_argument("--shared-layer-dir", type=str, default=None,
                        help="Shared dir for stage1/2 outputs. If set, caches stage1_model.pth and pca_components.pt here.")

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # shared layer dir for caching stages 1+2
    shared_dir = Path(args.shared_layer_dir) if args.shared_layer_dir else output_dir
    shared_dir.mkdir(parents=True, exist_ok=True)

    stage1_path = shared_dir / "stage1_model.pth"
    pca_path = shared_dir / "pca_components.pt"

    device = torch.device(args.device)
    set_seed(args.seed)
    logger = get_logger()

    # wandb
    if not args.no_wandb:
        wandb.init(project="VASAE", name=args.exp_name, group=args.wandb_group, config=vars(args))
    else:
        wandb.init(mode="disabled")

    # data
    data_cfg = DataConfig(
        train_batchsize=args.train_batchsize,
        valid_batchsize=args.valid_batchsize,
        test_batchsize=args.test_batchsize,
        use_centralize=args.use_centralize,
        layer_name=args.layer_name,
        data_dir=args.data_dir,
    )
    train_loader, valid_loader, test_loader = get_dataloader(data_cfg, args.seed)

    # blackbox model components
    bb_cfg = BlackBoxModelConfig(
        name=args.blackbox_model_name,
        dir=Path(args.blackbox_model_dir),
    )
    emb = load_embeding_layer(bb_cfg)
    unemb = load_unembeding_layer(bb_cfg)

    vocab_size, model_dim = emb.weight.shape
    stage1_epochs = args.stage1_epochs if args.stage1_epochs is not None else args.num_epochs

    # ---- Stage 1 ----
    if stage1_path.exists():
        logger.info(f"Stage 1: Loading cached model from {stage1_path}")
        sae_cfg = SAEConfig(
            dim_input=model_dim, dim_sparse=vocab_size,
            encoder_type="linear", sparsity_type=args.sparsity_type,
            k=args.k, nonneg_latents=args.nonneg_latents,
            tied_decoder=True, freeze_decoder=True,
            use_lowrank=False, use_abs_topk=args.use_abs_topk,
        )
        stage1_model = SAEModel(sae_cfg).to(device)
        if sae_cfg.tied_decoder:
            stage1_model.attach_embedding(emb, freeze=True)
        stage1_model.load_state_dict(torch.load(stage1_path, map_location=device, weights_only=True))
    else:
        sae_cfg = SAEConfig(
            dim_input=model_dim, dim_sparse=vocab_size,
            encoder_type="linear", sparsity_type=args.sparsity_type,
            k=args.k, nonneg_latents=args.nonneg_latents,
            tied_decoder=True, freeze_decoder=True,
            use_lowrank=False, use_abs_topk=args.use_abs_topk,
        )
        stage1_train_cfg = TrainConfig(
            num_epochs=stage1_epochs, max_batchsize=args.max_batchsize, lr=args.lr,
        )
        stage1_model = run_stage1(
            train_loader, valid_loader, emb, unemb,
            stage1_train_cfg, sae_cfg, device, logger, stage1_path,
        )

    # ---- Stage 2 ----
    if pca_path.exists():
        logger.info(f"Stage 2: Loading cached PCA from {pca_path}")
        pca_data = torch.load(pca_path, map_location=device, weights_only=True)
        W_full = pca_data["W_full"]
    else:
        W_full, _, _ = run_stage2(
            stage1_model, train_loader, device, logger, pca_path,
            max_batchsize=args.max_batchsize,
        )

    # ---- Stage 3 ----
    stage3_train_cfg = TrainConfig(
        num_epochs=args.num_epochs, max_batchsize=args.max_batchsize, lr=args.lr,
    )
    run_stage3(
        emb, unemb, W_full, args.d_pca, args.k,
        train_loader, valid_loader, test_loader,
        stage3_train_cfg, device, logger, output_dir,
    )

    wandb.finish()
    logger.info("Done.")


if __name__ == "__main__":
    main()
