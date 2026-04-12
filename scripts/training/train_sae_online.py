"""Online SAE training with nnsight — supports any HuggingFace causal LM.

Extracts activations on-the-fly, trains a vocab-aligned SAE,
and evaluates with Variance Explained, CE Loss Recovered, and LogitLens accuracy.

Examples:

    # GPT-2
    python scripts/training/train_sae_online.py --exp-name gpt2_L11 \
        --model-name gpt2 --layer-idx 11 --tied-decoder --nonneg-latents --no-wandb

    # LLaMA-3.2-1B (layer 15)
    python scripts/training/train_sae_online.py --exp-name llama1b_L15 \
        --model-name meta-llama/Llama-3.2-1B --layer-idx 15 \
        --tied-decoder --nonneg-latents --dtype float16 --no-wandb

    # Qwen2.5-0.5B
    python scripts/training/train_sae_online.py --exp-name qwen05b_L20 \
        --model-name Qwen/Qwen2.5-0.5B --layer-idx 20 \
        --tied-decoder --nonneg-latents --no-wandb
"""

import argparse
import json
import os

# Disable progress bars before any HF/tqdm imports
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

# Preload CUDA libraries before torch import (some nodes lack LD_LIBRARY_PATH)
import ctypes
import site

_sp = site.getsitepackages()[0]
for _lib in [
    "nvidia/cusparselt/lib/libcusparseLt.so.0",
    "nvidia/cusparse/lib/libcusparse.so.12",
]:
    _path = os.path.join(_sp, _lib)
    if os.path.exists(_path):
        ctypes.CDLL(_path)

import shutil
from pathlib import Path

import torch
import torch.optim as optim

from vasae.utils.log import get_logger
from vasae.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Online SAE training for any HF causal LM")

    # black-box model
    p.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        help="Any HuggingFace causal LM (gpt2, meta-llama/Llama-3.2-1B, ...)",
    )
    p.add_argument("--layer-idx", type=int, default=11)
    p.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype (default: model's native dtype)",
    )

    # data
    p.add_argument("--dataset", type=str, default="wikitext")
    p.add_argument(
        "--dataset-config",
        type=str,
        default="wikitext-103-raw-v1",
        help="Dataset config name (e.g., wikitext-103-raw-v1)",
    )
    p.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column name for text in the dataset",
    )
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--train-batchsize", type=int, default=32)
    p.add_argument("--eval-batchsize", type=int, default=32)
    p.add_argument("--train-samples", type=int, default=8000)
    p.add_argument("--eval-samples", type=int, default=2000)
    p.add_argument("--test-samples", type=int, default=1000)

    # sae architecture
    p.add_argument(
        "--dim-sparse",
        type=int,
        default=0,
        help="Sparse dim (0 = auto: vocab_size if tied, 8*dim_input otherwise)",
    )
    p.add_argument(
        "--encoder-type", type=str, default="linear", choices=["linear", "mlp"]
    )
    p.add_argument(
        "--sparsity-type",
        type=str,
        default="topk",
        choices=["none", "topk", "batch_topk"],
    )
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--per-item-in-eval", action="store_true")
    p.add_argument("--nonneg-latents", action="store_true")
    p.add_argument("--l1-coeff", type=float, default=0.0)
    p.add_argument(
        "--tied-decoder",
        action="store_true",
        help="Tie decoder to token embeddings (VASAE)",
    )
    p.add_argument("--freeze-decoder", action="store_true")
    p.add_argument("--use-abs-topk", action="store_true")
    p.add_argument("--anchor-coeff", type=float, default=0.0)
    p.add_argument(
        "--anchor-mode",
        type=str,
        default="hard",
        choices=["hard", "logsumexp", "softmax"],
    )
    p.add_argument("--anchor-topk", type=int, default=10)
    p.add_argument(
        "--anchor-every",
        type=int,
        default=1,
        help="Compute anchor loss every N training steps (1 = every batch). "
        "Higher values speed up training for large vocabularies.",
    )

    # training
    p.add_argument("--num-epochs", type=int, default=5)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping patience (0 = disabled). Stop if eval loss does not improve for N epochs.",
    )

    # system
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--save-dir", type=str, default="/scratch/b5bq/pu22650.b5bq/VASAE_out/online"
    )
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-group", type=str, default="online")
    p.add_argument("--exp-name", type=str, required=True)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    logger = get_logger()
    device = args.device
    logger.info(f"use device: {device}")

    save_dir: Path = Path(args.save_dir) / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Lazy imports (nnsight ~10s, transformers ~8s, wandb ~2s) --- so --help don't need to wait so long.
    import datasets
    import transformers
    from datasets import load_dataset, load_from_disk
    from nnsight import NNsight

    import wandb

    datasets.disable_progress_bars()
    transformers.logging.set_verbosity_error()

    from vasae.data.activation_source import OnlineActivationSource
    from vasae.engine.trainer import Trainer
    from vasae.metrics.base import MetricComposer
    from vasae.metrics.ce_loss import CELossRecovered
    from vasae.metrics.logitlens import LogitLens, LogitLensMetric
    from vasae.metrics.variance_explained import VarianceExplained
    from vasae.models.factory import get_embedding, get_layers, get_lm_head, load_model
    from vasae.models.sae import SAEConfig, SAEModel

    # --- Load LLM (model-agnostic) ---
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype)

    logger.info(f"Loading {args.model_name}...")
    llm, tokenizer = load_model(args.model_name, device=device, dtype=dtype)
    nn_model = NNsight(llm)

    n_layers = len(get_layers(llm))
    emb = get_embedding(llm)
    lm_head = get_lm_head(llm)
    dim_input = emb.weight.size(1)
    vocab_size = emb.weight.size(0)

    logger.info(
        f"Model: {type(llm).__name__}, dim={dim_input}, vocab={vocab_size}, "
        f"layers={n_layers}, using layer {args.layer_idx}"
    )

    if args.layer_idx >= n_layers:
        raise ValueError(f"layer_idx={args.layer_idx} >= n_layers={n_layers}")

    # --- Load dataset (with shared cache) ---
    ds_cache_name = f"{args.dataset}_{args.dataset_config or 'default'}".replace(
        "/", "_"
    )
    data_cache_dir = Path(args.save_dir) / ".data_cache" / ds_cache_name

    if (data_cache_dir / "dataset_info.json").exists():
        logger.info(f"Loading cached dataset from {data_cache_dir}")
        ds = load_from_disk(str(data_cache_dir))
    else:
        logger.info(f"Loading dataset {args.dataset} (first run, will cache)...")
        ds = load_dataset(args.dataset, args.dataset_config, split="train")
        # Rename text column if needed
        if args.text_column != "text" and args.text_column in ds.column_names:
            ds = ds.rename_column(args.text_column, "text")
        # Filter empty texts
        ds = ds.filter(lambda x: len(x["text"].strip()) > 50)
        # Save to shared cache (atomic via temp dir + rename)
        tmp_dir = data_cache_dir.with_name(f"{data_cache_dir.name}.tmp.{os.getpid()}")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(tmp_dir))
        try:
            tmp_dir.rename(data_cache_dir)
            logger.info(f"Cached dataset to {data_cache_dir}")
        except OSError:
            # Another job already saved — use theirs
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info(f"Cache already exists at {data_cache_dir}")

    n_total = len(ds)
    n_train = min(args.train_samples, n_total)
    n_eval = min(args.eval_samples, n_total - n_train)
    n_test = min(args.test_samples, n_total - n_train - n_eval)
    train_ds = ds.select(range(n_train))
    eval_ds = ds.select(range(n_train, n_train + n_eval))
    test_ds = ds.select(range(n_train + n_eval, n_train + n_eval + n_test))

    logger.info(f"Data split: train={n_train}, eval={n_eval}, test={n_test}")

    train_source = OnlineActivationSource(
        model=nn_model,
        tokenizer=tokenizer,
        layer_idx=args.layer_idx,
        text_dataset=train_ds,
        batch_size=args.train_batchsize,
        max_length=args.max_length,
    )
    eval_source = OnlineActivationSource(
        model=nn_model,
        tokenizer=tokenizer,
        layer_idx=args.layer_idx,
        text_dataset=eval_ds,
        batch_size=args.eval_batchsize,
        max_length=args.max_length,
    )
    test_source = OnlineActivationSource(
        model=nn_model,
        tokenizer=tokenizer,
        layer_idx=args.layer_idx,
        text_dataset=test_ds,
        batch_size=args.eval_batchsize,
        max_length=args.max_length,
    )

    # --- Build SAE ---
    if args.dim_sparse > 0:
        dim_sparse = args.dim_sparse
    elif args.tied_decoder:
        dim_sparse = vocab_size
    else:
        dim_sparse = 8 * dim_input

    sae_cfg = SAEConfig(
        dim_input=dim_input,
        dim_sparse=dim_sparse,
        encoder_type=args.encoder_type,
        sparsity_type=args.sparsity_type,
        k=args.k,
        per_item_in_eval=args.per_item_in_eval,
        nonneg_latents=args.nonneg_latents,
        l1_coeff=args.l1_coeff,
        tied_decoder=args.tied_decoder,
        freeze_decoder=args.freeze_decoder,
        use_abs_topk=args.use_abs_topk,
        anchor_coeff=args.anchor_coeff,
        anchor_mode=args.anchor_mode,
        anchor_topk=args.anchor_topk,
        anchor_every=args.anchor_every,
    )
    sae_model = SAEModel(sae_cfg).to(device)

    if args.tied_decoder:
        sae_model.attach_embedding(emb, freeze=args.freeze_decoder)

    if args.anchor_coeff > 0:
        sae_model.attach_anchor_embedding(emb)

    # Ensure SAE is float32 even if LLM embeddings were bfloat16
    sae_model = sae_model.float()

    logger.info(
        f"SAE: dim_sparse={dim_sparse}, tied={args.tied_decoder}, "
        f"sparsity={args.sparsity_type}, k={args.k}"
    )

    # --- Metrics ---
    logitlens_metric = LogitLensMetric(LogitLens(lm_head))
    ve_metric = VarianceExplained()
    ce_metric = CELossRecovered(nn_model, layer_idx=args.layer_idx)

    train_metrics = MetricComposer([logitlens_metric, ve_metric])
    eval_metrics = MetricComposer([logitlens_metric, ve_metric])
    test_metrics = MetricComposer([logitlens_metric, ve_metric, ce_metric])

    # --- Trainer ---
    optimizer = optim.Adam(
        [p for p in sae_model.parameters() if p.requires_grad], lr=args.lr
    )
    trainer = Trainer(
        sae_model=sae_model,
        optimizer=optimizer,
        metrics=train_metrics,
        eval_metrics=eval_metrics,
        device=device,
        logger=logger,
    )

    # --- wandb ---
    if not args.no_wandb:
        wandb.init(
            project="VASAE",
            name=args.exp_name,
            group=args.wandb_group,
            config=vars(args),
        )
    else:
        wandb.init(mode="disabled")

    # --- Training loop (with optional early stopping) ---
    best_eval_loss = float("inf")
    patience_counter = 0
    stopped_epoch = args.num_epochs

    for epoch in range(args.num_epochs):
        logger.info(f"=== Epoch {epoch + 1}/{args.num_epochs} ===")

        train_out = trainer.train_epoch(
            train_source,
            max_batches=args.max_batches,
            epoch=epoch + 1,
            num_epochs=args.num_epochs,
        )
        logger.info(
            f"[Train] loss={train_out['loss']:.4f} "
            f"VE={train_out.get('variance_explained', 0):.4f} "
            f"logitlens={train_out.get('logitlens_acc', 0) * 100:.2f}%"
        )

        eval_out = trainer.evaluate(eval_source, max_batches=args.max_batches)
        logger.info(
            f"[Eval] loss={eval_out['loss']:.4f} "
            f"VE={eval_out.get('variance_explained', 0):.4f} "
            f"logitlens={eval_out.get('logitlens_acc', 0) * 100:.2f}% "
            f"CE_recovered={eval_out.get('loss_recovered', 0):.4f}"
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                **{f"train/{k}": v for k, v in train_out.items()},
                **{f"eval/{k}": v for k, v in eval_out.items()},
            }
        )

        # Early stopping: track best eval loss and save best model
        if args.patience > 0:
            if eval_out["loss"] < best_eval_loss:
                best_eval_loss = eval_out["loss"]
                patience_counter = 0
                sae_model.save_pretrained(save_dir)
                logger.info(f"Best model saved (eval_loss={best_eval_loss:.4f})")
            else:
                patience_counter += 1
                logger.info(f"No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                stopped_epoch = epoch + 1
                logger.info(f"Early stopping at epoch {stopped_epoch}")
                break

    # --- Load best model for final test (if early stopping was used) ---
    if args.patience > 0:
        logger.info("Loading best model for final test...")
        # Free old model and optimizer to avoid OOM when loading best checkpoint
        del trainer.sae_model, sae_model
        del optimizer
        torch.cuda.empty_cache()
        sae_model = SAEModel.from_pretrained(save_dir).to(device)
        if args.tied_decoder:
            sae_model.attach_embedding(emb, freeze=args.freeze_decoder)
        if args.anchor_coeff > 0:
            sae_model.attach_anchor_embedding(emb)
        sae_model = sae_model.float()
        trainer.sae_model = sae_model
    else:
        sae_model.save_pretrained(save_dir)

    # --- Final test evaluation (with CE loss) ---
    logger.info("=== Final Test ===")
    trainer.eval_metrics = test_metrics
    test_out = trainer.evaluate(test_source)
    logger.info(
        f"[Test] loss={test_out['loss']:.4f} "
        f"VE={test_out.get('variance_explained', 0):.4f} "
        f"logitlens={test_out.get('logitlens_acc', 0) * 100:.2f}% "
        f"CE_recovered={test_out.get('loss_recovered', 0):.4f}"
    )
    wandb.log({f"test/{k}": v for k, v in test_out.items()})

    # --- Compute dead feature rate and L0 (over test set) ---
    logger.info("Computing dead feature rate and L0 on test set...")
    feature_counts = torch.zeros(sae_model.config.dim_sparse, device=device)
    l0_sum = 0.0
    n_samples = 0
    sae_model.eval()
    with torch.no_grad():
        for batch in test_source:
            activations = batch["activations"].to(device).float()
            mask = batch.get("attention_mask")
            if mask is not None:
                activations = activations[mask.bool()]
            _, z = sae_model.encode(activations)
            nonzero = (z != 0).float()
            feature_counts += nonzero.sum(dim=0)
            l0_sum += nonzero.sum(dim=1).sum().item()
            n_samples += activations.size(0)

    dead_rate = (feature_counts == 0).float().mean().item()
    l0 = l0_sum / n_samples if n_samples > 0 else 0.0
    logger.info(
        f"Dead feature rate: {dead_rate:.4f}, L0: {l0:.2f} (over {n_samples} samples)"
    )

    logger.info(f"Model saved to {save_dir}")

    results = {
        "config": vars(args),
        "stopped_epoch": stopped_epoch,
        "test": {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in test_out.items()
        },
        "dead_rate": dead_rate,
        "l0": l0,
        "last_eval": {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in eval_out.items()
        },
    }
    results_path = save_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
