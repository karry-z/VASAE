"""
Training pipeline for Dual-Path SAE (proposal Section 3.4).

Phase 1: Compute P_E (embedding projector) and PCA of embedding-orthogonal residuals.
Phase 2: Train DualPathSAE with frozen W_E and P_k, L1 on both codes.
Phase 3: Evaluate VE, VE_sparse, VE_dense, logitlens on test set.
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
from vasae.metrics.interface import Aggregator, MetricComposer
from vasae.metrics.logitlens import LogitLens, LogitLensAccuracy, LogitLensMetric
from vasae.models.dualpath_sae import DualPathSAE, DualPathSAEOutput
from vasae.models.factory import (
    BlackBoxModelConfig,
    load_embeding_layer,
    load_unembeding_layer,
)
from vasae.utils.log import get_logger
from vasae.utils.seed import set_seed


# ---- Phase 1: Compute embedding-orthogonal PCA ----

@torch.no_grad()
def run_phase1(
    emb: torch.nn.Embedding,
    train_loader,
    device,
    logger,
    output_path: Path,
    max_batchsize: int = 0,
):
    """Compute P_E = W_E^+ @ W_E and PCA of embedding-orthogonal residuals."""
    logger.info("=== Phase 1: Compute embedding-orthogonal PCA ===")

    W_E = emb.weight.to(device).float()  # [vocab, d]
    vocab_size, dim = W_E.shape

    # Compute embedding projector P_E = W_E^T @ (W_E @ W_E^T)^{-1} @ W_E = W_E^+ @ W_E
    # Using pinv: P_E = W_E^+ @ W_E where W_E^+ = pinv(W_E)
    W_E_pinv = torch.linalg.pinv(W_E)  # [d, vocab]
    P_E = W_E_pinv @ W_E  # [d, d] — projects onto column span of W_E^T

    # Sanity checks
    rank_W_E = torch.linalg.matrix_rank(W_E).item()
    logger.info(f"Rank of W_E: {rank_W_E} (out of {dim})")
    logger.info(f"||P_E - I||_F = {torch.norm(P_E - torch.eye(dim, device=device)):.6f}")

    # Iterate training data to compute covariance of embedding-orthogonal residuals
    I_minus_PE = torch.eye(dim, device=device) - P_E  # [d, d]

    sum_r = torch.zeros(dim, device=device, dtype=torch.float64)
    sum_rrt = torch.zeros(dim, dim, device=device, dtype=torch.float64)
    n = 0

    for batch_i, data in enumerate(train_loader):
        h = data["activations"].to(device).float()
        h_flat = h.view(-1, dim)

        # r = (I - P_E) @ h — embedding-orthogonal component
        r = (h_flat @ I_minus_PE.T).to(torch.float64)
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

    # Verify orthogonality: P_k^T @ W_E^T should be ≈ 0
    top_k = min(8, dim)
    P_k_test = W_full[:, :top_k]  # [d, top_k]
    ortho_check = torch.norm(P_k_test.T @ W_E.T)
    logger.info(f"||P_k^T @ W_E^T|| (top {top_k} dirs) = {ortho_check:.6e} (should be ≈ 0)")

    # Explained variance
    total_var = eigenvalues.sum().item()
    for k in [1, 2, 4, 8, 16, 32]:
        if k <= len(eigenvalues):
            ev = eigenvalues[:k].sum().item() / total_var * 100
            logger.info(f"Explained variance (top {k}): {ev:.2f}%")

    logger.info(f"Top 20 eigenvalues: {eigenvalues[:20].tolist()}")

    torch.save({
        "P_E": P_E.cpu(),
        "W_full": W_full.cpu(),
        "mean_r": mean_r.float().cpu(),
        "eigenvalues": eigenvalues.cpu(),
        "rank_W_E": rank_W_E,
    }, output_path)

    logger.info(f"Phase 1 saved to {output_path}")
    return W_full, mean_r.float(), eigenvalues


# ---- Phase 2: Train DualPathSAE ----

def train_one_epoch(
    model: DualPathSAE,
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

        out: DualPathSAEOutput = model(h)
        eval_outcomes = metrics.compute({"data": h, "decoded": out.h_recon})

        aggregator.add({
            "loss": out.loss.detach().cpu().item(),
            "recon_loss": out.recon_loss.detach().cpu().item(),
            "l1_z": out.l1_z.detach().cpu().item(),
            "l1_y": out.l1_y.detach().cpu().item(),
            "logitlens_acc": eval_outcomes["logitlens_acc"],
            "z_l0": (out.z > 0).float().sum(-1).mean().item(),
        }, h.size(0))

        out.loss.backward()
        optimizer.step()

        if logger is not None:
            logger.info(
                f"[Train] Epoch {epoch+1}/{train_cfg.num_epochs} "
                f"batch {batch_i+1}/{len(loader)} "
                f"loss {out.loss.item():.4f} "
                f"recon {out.recon_loss.item():.4f} "
                f"l1_z {out.l1_z.item():.4f} "
                f"l1_y {out.l1_y.item():.4f} "
                f"acc {eval_outcomes['logitlens_acc']*100:.2f}%"
            )

        if train_cfg.max_batchsize > 0 and batch_i > train_cfg.max_batchsize:
            break

    return aggregator.compute()


@torch.no_grad()
def evaluate_model(
    model: DualPathSAE,
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

        out: DualPathSAEOutput = model(h)
        eval_outcomes = metrics.compute({"data": h, "decoded": out.h_recon})

        aggregator.add({
            "loss": out.loss.detach().cpu().item(),
            "recon_loss": out.recon_loss.detach().cpu().item(),
            "l1_z": out.l1_z.detach().cpu().item(),
            "l1_y": out.l1_y.detach().cpu().item(),
            "logitlens_acc": eval_outcomes["logitlens_acc"],
            "z_l0": (out.z > 0).float().sum(-1).mean().item(),
        }, h.size(0))

        if max_batchsize > 0 and batch_i > max_batchsize:
            break

    return aggregator.compute()


# ---- Phase 3: Test evaluation (VE, VE_sparse, VE_dense) ----

@torch.no_grad()
def compute_test_metrics(
    model: DualPathSAE,
    test_loader,
    device,
    logger,
    max_batchsize: int = 0,
):
    """Compute VE (full), VE_sparse (token path only), VE_dense (dense path only)."""
    model.eval()
    dim = model.dim_input

    # First pass: compute mean of h
    sum_h = torch.zeros(dim, device=device, dtype=torch.float64)
    total_n = 0
    for batch_i, data in enumerate(test_loader):
        h = data["activations"].to(device)
        h_flat = h.view(-1, dim).to(torch.float64)
        sum_h += h_flat.sum(0)
        total_n += h_flat.shape[0]
        if max_batchsize > 0 and batch_i > max_batchsize:
            break
    mean_h = (sum_h / total_n).float()

    # Second pass: compute MSE and variance
    sum_mse_full = 0.0
    sum_mse_sparse = 0.0
    sum_mse_dense = 0.0
    sum_var = 0.0
    n = 0

    for batch_i, data in enumerate(test_loader):
        h = data["activations"].to(device)
        h_flat = h.view(-1, dim)

        out: DualPathSAEOutput = model(h_flat)

        # Full reconstruction MSE
        sum_mse_full += F.mse_loss(out.h_recon, h_flat, reduction="sum").item()

        # Sparse-only: z @ W_E + mean_r
        h_sparse_recon = out.h_sparse + model.mean_r
        sum_mse_sparse += F.mse_loss(h_sparse_recon, h_flat, reduction="sum").item()

        # Dense-only: y @ P_k^T + mean_r
        h_dense_recon = out.y @ model.P_k.T + model.mean_r
        sum_mse_dense += F.mse_loss(h_dense_recon, h_flat, reduction="sum").item()

        # Variance
        sum_var += F.mse_loss(h_flat, mean_h.unsqueeze(0).expand_as(h_flat), reduction="sum").item()
        n += h_flat.shape[0] * dim

        if max_batchsize > 0 and batch_i > max_batchsize:
            break

    mse_full = sum_mse_full / n
    mse_sparse = sum_mse_sparse / n
    mse_dense = sum_mse_dense / n
    var_h = sum_var / n

    ve = 1.0 - mse_full / var_h
    ve_sparse = 1.0 - mse_sparse / var_h
    ve_dense = 1.0 - mse_dense / var_h

    logger.info(
        f"VE={ve:.4f}, VE_sparse={ve_sparse:.4f}, VE_dense={ve_dense:.4f}, "
        f"MSE_full={mse_full:.6f}, MSE_sparse={mse_sparse:.6f}, MSE_dense={mse_dense:.6f}"
    )
    return {
        "ve": ve,
        "ve_sparse": ve_sparse,
        "ve_dense": ve_dense,
        "mse_full": mse_full,
        "mse_sparse": mse_sparse,
        "mse_dense": mse_dense,
        "var_h": var_h,
    }


def run_phase2(
    emb,
    unemb,
    W_full,
    mean_r,
    d_pca,
    lambda_z,
    lambda_y,
    train_loader,
    valid_loader,
    test_loader,
    train_cfg,
    device,
    logger,
    output_dir: Path,
):
    logger.info(f"=== Phase 2: Train DualPathSAE with d_pca={d_pca} ===")

    dim_input = emb.weight.shape[1]
    vocab_size = emb.weight.shape[0]

    model = DualPathSAE(dim_input, vocab_size, d_pca, lambda_z, lambda_y).to(device)
    model.attach_embedding(emb)
    P_k = W_full[:, :d_pca].to(device)
    model.attach_pca(P_k, mean_r.to(device))

    logitlens = LogitLens(unemb)
    logitlens_acc = LogitLensAccuracy()
    metrics = MetricComposer([LogitLensMetric(logitlens, logitlens_acc)])

    # Only optimize encoders (decoders are frozen buffers)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=train_cfg.lr)
    logger.info(f"Trainable params: {sum(p.numel() for p in trainable_params)}")

    for epoch in range(train_cfg.num_epochs):
        train_out = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            train_cfg=train_cfg,
            metrics=metrics,
            device=device,
            logger=logger,
            epoch=epoch,
        )
        eval_out = evaluate_model(
            model=model,
            loader=valid_loader,
            metrics=metrics,
            device=device,
            logger=logger,
            max_batchsize=train_cfg.max_batchsize,
        )
        wandb.log({
            **{f"train/{k}": v for k, v in train_out.items()},
            **{f"valid/{k}": v for k, v in eval_out.items()},
            "epoch": epoch,
        })

    # Phase 3: Test evaluation
    logger.info("=== Phase 3: Test evaluation ===")
    test_out = evaluate_model(
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

    results = {**test_out, **ve_metrics, "d_pca": d_pca, "lambda_z": lambda_z, "lambda_y": lambda_y}
    wandb.log({f"test/{k}": v for k, v in results.items()})

    # Save
    torch.save(model.state_dict(), output_dir / "model.pth")
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Model saved to {output_dir / 'model.pth'}")
    logger.info(f"Results: {results}")
    return model, results


# ---- CLI ----

def parse_args():
    parser = argparse.ArgumentParser(description="Train Dual-Path SAE (proposal Section 3.4)")

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

    # DualPathSAE
    parser.add_argument("--d-pca", type=int, required=True, help="Number of PCA directions")
    parser.add_argument("--lambda-z", type=float, default=1e-3, help="L1 weight for sparse code")
    parser.add_argument("--lambda-y", type=float, default=1e-4, help="L1 weight for dense code")

    # training
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--max-batchsize", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)

    # output
    parser.add_argument("--output-dir", type=str, required=True)

    # system
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-group", type=str, default="dualpath")
    parser.add_argument("--exp-name", type=str, required=True)

    # blackbox model
    parser.add_argument("--blackbox-model-name", type=str, default="gpt2")
    parser.add_argument(
        "--blackbox-model-dir", type=str,
        default=r"/scratch/b5bq/pu22650.b5bq/VASAE_out/BlackBoxModels/gpt2",
    )

    # phase caching
    parser.add_argument("--shared-layer-dir", type=str, default=None,
                        help="Shared dir for phase 1 outputs (PCA). Cached per layer.")

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_dir = Path(args.shared_layer_dir) if args.shared_layer_dir else output_dir
    shared_dir.mkdir(parents=True, exist_ok=True)

    pca_path = shared_dir / "dualpath_pca.pt"

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

    # ---- Phase 1: Compute embedding-orthogonal PCA ----
    if pca_path.exists():
        logger.info(f"Phase 1: Loading cached PCA from {pca_path}")
        pca_data = torch.load(pca_path, map_location=device, weights_only=True)
        W_full = pca_data["W_full"]
        mean_r = pca_data["mean_r"]
    else:
        W_full, mean_r, _ = run_phase1(
            emb, train_loader, device, logger, pca_path,
            max_batchsize=args.max_batchsize,
        )

    # ---- Phase 2+3: Train and evaluate DualPathSAE ----
    train_cfg = TrainConfig(
        num_epochs=args.num_epochs, max_batchsize=args.max_batchsize, lr=args.lr,
    )
    run_phase2(
        emb, unemb, W_full, mean_r, args.d_pca,
        args.lambda_z, args.lambda_y,
        train_loader, valid_loader, test_loader,
        train_cfg, device, logger, output_dir,
    )

    wandb.finish()
    logger.info("Done.")


if __name__ == "__main__":
    main()
