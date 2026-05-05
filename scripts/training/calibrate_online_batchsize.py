"""Short online SAE batch-size calibration runs.

This intentionally trains only for a capped number of batches and does not
evaluate, save checkpoints, or log to wandb. It is meant to be run on the
target compute node so GPU memory and throughput match the real job.
"""

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.training.train_sae_online import (
    build_jsonl_train_eval_sources,
    build_train_metrics,
    infer_dim_sparse,
)
from vasae.data.schema import DataConfig
from vasae.engine.trainer import Trainer
from vasae.models.online import attach_sae_embeddings, load_online_llm
from vasae.models.sae import SAEConfig, SAEModel
from vasae.utils.log import get_logger
from vasae.utils.seed import set_seed

logger = get_logger()


def parse_args():
    p = argparse.ArgumentParser(description="Calibrate online SAE batch size")
    p.add_argument("--model-name", required=True)
    p.add_argument("--dtype", default=None, choices=["float16", "bfloat16", "float32"])
    p.add_argument("--layer-idx", type=int, required=True)
    p.add_argument("--corpus-dir", type=Path, required=True)
    p.add_argument("--corpora", nargs="+", default=["fineweb", "dclm", "pile"])
    p.add_argument("--batch-sizes", nargs="+", type=int, required=True)
    p.add_argument("--variant", choices=["plain", "soft"], required=True)
    p.add_argument("--dim-sparse", type=int, required=True)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--train-tokens", type=int, default=1_000_000)
    p.add_argument("--max-batches", type=int, default=200)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--anchor-coeff", type=float, default=0.0)
    p.add_argument("--anchor-every", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=20)
    return p.parse_args()


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_sae(args, llm_ctx):
    anchor_coeff = args.anchor_coeff if args.variant == "soft" else 0.0
    cfg = SAEConfig(
        dim_model=llm_ctx.dim_model,
        dim_sparse=infer_dim_sparse(
            args.dim_sparse,
            tied_decoder=False,
            dim_model=llm_ctx.dim_model,
            vocab_size=llm_ctx.vocab_size,
        ),
        sparsity_type="topk",
        k=args.k,
        nonneg_latents=True,
        anchor_coeff=anchor_coeff,
        anchor_mode="hard",
        anchor_every=args.anchor_every,
    )
    model = SAEModel(cfg).to(args.device)
    return attach_sae_embeddings(model, llm_ctx.embedding, freeze_decoder=False).float()


def run_one(args, llm_ctx, batch_size: int):
    data_cfg = DataConfig(
        max_length=args.max_length,
        train_batchsize=batch_size,
        valid_batchsize=batch_size,
    )
    train_source, _, _ = build_jsonl_train_eval_sources(
        data_cfg=data_cfg,
        nn_model=llm_ctx.nn_model,
        tokenizer=llm_ctx.tokenizer,
        layer_idx=args.layer_idx,
        model_name=args.model_name,
        corpus_dir=args.corpus_dir,
        corpora=args.corpora,
        train_tokens=args.train_tokens,
        valid_tokens=1,
    )
    sae_model = make_sae(args, llm_ctx)
    metrics = build_train_metrics(llm_ctx.lm_head)
    optimizer = optim.Adam(
        [p for p in sae_model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    trainer = Trainer(
        sae_model=sae_model,
        metrics=metrics,
        eval_metrics=metrics,
        device=args.device,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    train_out = trainer.train_epoch(
        train_source,
        optimizer=optimizer,
        max_batches=args.max_batches,
        log_every=args.log_every,
        log_interval_seconds=60,
        epoch=1,
        num_epochs=1,
    )
    elapsed = time.monotonic() - started
    actual_tokens = int(train_out.get("tokens_processed", 0))
    tok_s = actual_tokens / elapsed if elapsed > 0 else float("nan")
    peak_gb = (
        torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    )
    logger.info(
        "CALIB_RESULT model=%s variant=%s batch_size=%d status=ok "
        "elapsed=%.2fs tokens=%d tok/s=%.1f peak_mem=%.2fGiB",
        args.model_name,
        args.variant,
        batch_size,
        elapsed,
        actual_tokens,
        tok_s,
        peak_gb,
    )
    cleanup(trainer, optimizer, metrics, sae_model, train_source)
    return tok_s


def main():
    args = parse_args()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    set_seed(args.seed)
    logger.info(
        "Batch calibration: model=%s variant=%s layer=%d batch_sizes=%s",
        args.model_name,
        args.variant,
        args.layer_idx,
        args.batch_sizes,
    )
    llm_ctx = load_online_llm(
        args.model_name,
        device=args.device,
        dtype_name=args.dtype,
        layer_idx=args.layer_idx,
    )

    results = []
    for batch_size in args.batch_sizes:
        try:
            tok_s = run_one(args, llm_ctx, batch_size)
            results.append((batch_size, tok_s))
        except RuntimeError as exc:
            message = str(exc).splitlines()[0]
            if "out of memory" not in str(exc).lower():
                raise
            logger.info(
                "CALIB_RESULT model=%s variant=%s batch_size=%d status=oom error=%s",
                args.model_name,
                args.variant,
                batch_size,
                message,
            )
            cleanup()

    if not results:
        logger.info("CALIB_BEST model=%s variant=%s status=no_success", args.model_name, args.variant)
        return
    best_batch, best_tok_s = max(results, key=lambda item: item[1])
    logger.info(
        "CALIB_BEST model=%s variant=%s batch_size=%d tok/s=%.1f",
        args.model_name,
        args.variant,
        best_batch,
        best_tok_s,
    )


if __name__ == "__main__":
    main()
