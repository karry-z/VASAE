"""Train the final paper-facing SAE variants: plain, VASAE-soft, and hard_tied_baseline."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path


METHODS = ("plain", "vasae_soft", "hard_tied_baseline")


def parse_dim_sparse(value: str) -> int | str:
    value = value.strip().lower()
    if value in {"vocab", "0"}:
        return "vocab"
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--dim-sparse must be a positive integer, 0, or 'vocab'."
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "--dim-sparse must be a positive integer, 0, or 'vocab'."
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train paper reproduction SAE variants. VASAE-soft uses a learnable "
            "decoder with vocabulary-anchor regularization; hard_tied_baseline is "
            "a separate fixed-decoder baseline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-name", default="gpt2", help="Hugging Face causal LM.")
    parser.add_argument("--layer-idx", type=int, default=11, help="Transformer layer to reconstruct.")
    parser.add_argument("--method", choices=METHODS, default="vasae_soft", help="Training method.")
    parser.add_argument("--dataset", default="wikitext", help="Hugging Face dataset name.")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1", help="Dataset config name; use empty string for none.")
    parser.add_argument("--text-column", default="text", help="Dataset text column.")
    parser.add_argument("--max-length", type=int, default=128, help="Tokenization length.")
    parser.add_argument("--train-samples", type=int, default=8000, help="Training text rows.")
    parser.add_argument("--eval-samples", type=int, default=2000, help="Evaluation text rows.")
    parser.add_argument("--test-samples", type=int, default=1000, help="Final test text rows.")
    parser.add_argument("--batch-size", type=int, default=32, help="Text batch size for activation extraction.")
    parser.add_argument("--dim-sparse", type=parse_dim_sparse, default="vocab", help="Sparse dimension; integer, 0, or 'vocab'.")
    parser.add_argument("--k", type=int, default=32, help="TopK active features per token.")
    parser.add_argument("--anchor-coeff", type=float, default=0.1, help="VASAE-soft vocabulary-anchor coefficient.")
    parser.add_argument("--anchor-every", type=int, default=1, help="Apply anchor loss every N training batches.")
    parser.add_argument("--num-epochs", type=int, default=5, help="Maximum training epochs.")
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience; 0 disables it.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or a torch device string.")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default=None, help="Language-model dtype.")
    parser.add_argument("--save-dir", default="outputs/runs", help="Directory for run outputs.")
    parser.add_argument("--exp-name", default=None, help="Experiment subdirectory name.")
    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    wandb_group.add_argument(
        "--no-wandb",
        action="store_true",
        help="Compatibility flag; W&B is disabled by default unless --wandb is passed.",
    )
    return parser


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s][%(asctime)s][%(name)s] %(message)s",
        datefmt="%Y%m%d %H:%M:%S",
    )
    return logging.getLogger("train_vasae")


def dataset_config_name(value: str | None) -> str | None:
    if value is None or value.strip() == "":
        return None
    return value


def resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype_name: str | None):
    import torch

    return {
        None: None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def load_lm_and_tokenizer(model_name: str, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer


def transformer_layer_count(model) -> int:
    candidates = [
        ("transformer", "h"),
        ("model", "layers"),
        ("gpt_neox", "layers"),
    ]
    for root_name, layers_name in candidates:
        root = getattr(model, root_name, None)
        if root is not None and hasattr(root, layers_name):
            return len(getattr(root, layers_name))
    decoder = getattr(getattr(model, "model", None), "decoder", None)
    if decoder is not None and hasattr(decoder, "layers"):
        return len(decoder.layers)
    raise ValueError(f"Cannot infer transformer layer count for {type(model).__name__}.")


def load_text_dataset(args, logger: logging.Logger):
    import datasets
    from datasets import load_dataset

    datasets.disable_progress_bars()
    config_name = dataset_config_name(args.dataset_config)
    logger.info("Loading dataset %s%s", args.dataset, f"/{config_name}" if config_name else "")
    dataset = load_dataset(args.dataset, config_name, split="train")
    if args.text_column not in dataset.column_names:
        raise ValueError(
            f"Text column {args.text_column!r} not found. Available columns: {dataset.column_names}"
        )
    if args.text_column != "text":
        dataset = dataset.rename_column(args.text_column, "text")
    dataset = dataset.filter(lambda row: isinstance(row["text"], str) and row["text"].strip() != "")
    return dataset


def split_dataset(dataset, train_samples: int, eval_samples: int, test_samples: int):
    total = len(dataset)
    n_train = min(train_samples, total)
    n_eval = min(eval_samples, max(total - n_train, 0))
    n_test = min(test_samples, max(total - n_train - n_eval, 0))
    if n_train == 0:
        raise ValueError("No training rows are available after filtering empty text.")
    train = dataset.select(range(n_train))
    eval_ = dataset.select(range(n_train, n_train + n_eval))
    test = dataset.select(range(n_train + n_eval, n_train + n_eval + n_test))
    return train, eval_, test


def resolve_dim_sparse(dim_sparse_arg: int | str, vocab_size: int) -> int:
    return vocab_size if dim_sparse_arg == "vocab" else int(dim_sparse_arg)


def build_sae(args, dim_input: int, vocab_embedding, device):
    import torch

    from vasae.models import SAEConfig, SAEModel

    vocab_size = vocab_embedding.weight.size(0)
    dim_sparse = resolve_dim_sparse(args.dim_sparse, vocab_size)
    if args.method == "hard_tied_baseline" and dim_sparse != vocab_size:
        raise ValueError(
            "hard_tied_baseline requires --dim-sparse vocab or 0 so sparse features match vocabulary size."
        )

    decoder_mode = "hard_tied_baseline" if args.method == "hard_tied_baseline" else "learnable"
    anchor_coeff = args.anchor_coeff if args.method == "vasae_soft" else 0.0
    cfg = SAEConfig(
        dim_input=dim_input,
        dim_sparse=dim_sparse,
        k=args.k,
        decoder_mode=decoder_mode,
        anchor_coeff=anchor_coeff,
        anchor_every=args.anchor_every,
    )
    sae = SAEModel(cfg).to(device).float()
    embedding = vocab_embedding.to(device)
    if args.method == "hard_tied_baseline":
        sae.attach_tied_decoder_embedding(embedding, freeze=True)
    elif args.method == "vasae_soft" and anchor_coeff > 0:
        sae.attach_vocab_anchor(embedding)
    return sae


def metric_payload(metrics: dict) -> dict:
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def sae_config_payload(config) -> dict:
    return {
        "dim_input": config.dim_input,
        "dim_sparse": config.dim_sparse,
        "k": config.k,
        "nonneg_latents": config.nonneg_latents,
        "mse_reduction": config.mse_reduction,
        "use_abs_topk": config.use_abs_topk,
        "decoder_mode": config.decoder_mode,
        "anchor_coeff": config.anchor_coeff,
        "anchor_mode": config.anchor_mode,
        "anchor_topk": config.anchor_topk,
        "anchor_every": config.anchor_every,
    }


def save_checkpoint(path: Path, sae_model, args, epoch: int, metrics: dict) -> None:
    import torch

    payload = {
        "model_state_dict": sae_model.state_dict(),
        "sae_config": sae_config_payload(sae_model.config),
        "method": args.method,
        "epoch": epoch,
        "metrics": metric_payload(metrics),
    }
    torch.save(payload, path)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_epoch_csv(path: Path, history: list[dict]) -> None:
    if not history:
        return
    fieldnames = sorted({key for row in history for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def compute_feature_stats(sae_model, data_source, device) -> dict:
    import torch

    sae_model.eval()
    feature_counts = torch.zeros(sae_model.config.dim_sparse, device=device)
    l0_sum = 0.0
    n_tokens = 0
    with torch.no_grad():
        for batch in data_source:
            activations = batch["activations"].to(device).float()
            mask = batch.get("attention_mask")
            if mask is not None:
                activations = activations[mask.bool()]
            _, sparse = sae_model.encode(activations)
            nonzero = sparse != 0
            feature_counts += nonzero.float().sum(dim=0)
            l0_sum += nonzero.float().sum(dim=1).sum().item()
            n_tokens += activations.size(0)
    return {
        "dead_feature_rate": float((feature_counts == 0).float().mean().item()),
        "mean_l0": float(l0_sum / max(n_tokens, 1)),
        "tokens_evaluated": int(n_tokens),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = setup_logger()

    import torch
    import torch.optim as optim
    import transformers
    from nnsight import NNsight

    transformers.logging.set_verbosity_error()
    torch.manual_seed(42)

    from vasae.data.activation_source import OnlineActivationSource
    from vasae.engine.trainer import Trainer
    from vasae.metrics.base import MetricComposer
    from vasae.metrics.variance_explained import VarianceExplained

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    exp_name = args.exp_name or f"{args.model_name.replace('/', '_')}_L{args.layer_idx}_{args.method}"
    output_dir = Path(args.save_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading language model %s", args.model_name)
    lm, tokenizer = load_lm_and_tokenizer(args.model_name, device, dtype)
    layer_count = transformer_layer_count(lm)
    if args.layer_idx < 0 or args.layer_idx >= layer_count:
        raise ValueError(f"--layer-idx {args.layer_idx} is outside model layer range 0..{layer_count - 1}.")
    embedding = lm.get_input_embeddings()
    dim_input = embedding.weight.size(1)
    vocab_size = embedding.weight.size(0)
    logger.info("Model hidden dim=%d, vocab=%d, layers=%d", dim_input, vocab_size, layer_count)

    dataset = load_text_dataset(args, logger)
    train_rows, eval_rows, test_rows = split_dataset(
        dataset, args.train_samples, args.eval_samples, args.test_samples
    )
    logger.info("Data split rows: train=%d eval=%d test=%d", len(train_rows), len(eval_rows), len(test_rows))

    nn_model = NNsight(lm)
    train_source = OnlineActivationSource(
        model=nn_model,
        tokenizer=tokenizer,
        layer_idx=args.layer_idx,
        text_dataset=train_rows,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    eval_source = OnlineActivationSource(
        model=nn_model,
        tokenizer=tokenizer,
        layer_idx=args.layer_idx,
        text_dataset=eval_rows,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    test_source = OnlineActivationSource(
        model=nn_model,
        tokenizer=tokenizer,
        layer_idx=args.layer_idx,
        text_dataset=test_rows,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    sae_model = build_sae(args, dim_input, embedding, device)
    logger.info(
        "Training method=%s dim_sparse=%d k=%d decoder_mode=%s anchor_coeff=%.4g",
        args.method,
        sae_model.config.dim_sparse,
        sae_model.config.k,
        sae_model.config.decoder_mode,
        sae_model.config.anchor_coeff,
    )

    optimizer = optim.Adam([p for p in sae_model.parameters() if p.requires_grad], lr=args.lr)
    metrics = MetricComposer([VarianceExplained()])
    trainer = Trainer(
        sae_model=sae_model,
        optimizer=optimizer,
        metrics=metrics,
        eval_metrics=metrics,
        device=str(device),
        logger=logger,
    )

    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError:
            logger.warning("--wandb was requested but wandb is not installed; continuing without W&B logging.")
        else:
            wandb_run = wandb.init(project="VASAE", name=exp_name, config=vars(args))

    config_payload = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "exp_name": exp_name,
        "model": {
            "model_name": args.model_name,
            "layer_idx": args.layer_idx,
            "dim_input": dim_input,
            "vocab_size": vocab_size,
            "n_layers": layer_count,
        },
        "sae_config": sae_config_payload(sae_model.config),
    }
    write_json(output_dir / "config.json", config_payload)

    best_eval_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict] = []
    checkpoint_path = output_dir / "checkpoint.pt"

    for epoch in range(1, args.num_epochs + 1):
        logger.info("Epoch %d/%d", epoch, args.num_epochs)
        train_metrics = trainer.train_epoch(train_source, epoch=epoch, num_epochs=args.num_epochs)
        eval_metrics = trainer.evaluate(eval_source) if len(eval_rows) > 0 else {}
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in metric_payload(train_metrics).items()},
            **{f"eval_{key}": value for key, value in metric_payload(eval_metrics).items()},
        }
        history.append(row)
        logger.info("Train metrics: %s", metric_payload(train_metrics))
        logger.info("Eval metrics: %s", metric_payload(eval_metrics))
        if wandb_run is not None:
            wandb_run.log(row)

        eval_loss = float(eval_metrics.get("loss", train_metrics.get("loss", float("inf"))))
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(checkpoint_path, sae_model, args, epoch, eval_metrics or train_metrics)
            logger.info("Saved checkpoint at epoch %d", epoch)
        else:
            stale_epochs += 1
            if args.patience > 0 and stale_epochs >= args.patience:
                logger.info("Stopping early at epoch %d after %d stale epochs", epoch, stale_epochs)
                break

    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device)
        sae_model.load_state_dict(payload["model_state_dict"])
    else:
        save_checkpoint(checkpoint_path, sae_model, args, args.num_epochs, {})

    test_metrics = trainer.evaluate(test_source) if len(test_rows) > 0 else {}
    feature_stats = compute_feature_stats(sae_model, test_source, device) if len(test_rows) > 0 else {}
    metrics_payload = {
        "best_epoch": best_epoch,
        "best_eval_loss": best_eval_loss,
        "history": history,
        "test": metric_payload(test_metrics),
        "feature_stats": feature_stats,
    }
    write_json(output_dir / "metrics.json", metrics_payload)
    write_epoch_csv(output_dir / "metrics.csv", history)
    logger.info("Final test metrics: %s", metric_payload(test_metrics))
    logger.info("Saved checkpoint, config, and metrics to %s", output_dir)

    if wandb_run is not None:
        wandb_run.log({f"test_{key}": value for key, value in metric_payload(test_metrics).items()})
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
