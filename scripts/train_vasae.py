"""Train the paper-facing SAE variants from text activations."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import datasets
import torch
import torch.optim as optim
import transformers
from datasets import load_dataset
from nnsight import NNsight
from transformers import AutoModelForCausalLM, AutoTokenizer

from vasae.data import OnlineActivationSource
from vasae.engine import MetricComposer, Trainer
from vasae.metrics import VarianceExplained
from vasae.models import SAEConfig, SAEModel


DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
TEXT_COLUMN = "text"
RUNS_DIR = Path("outputs/runs")
METHODS = ("plain", "vasae_soft", "hard_tied_baseline")

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(asctime)s][%(name)s] %(message)s",
    datefmt="%Y%m%d %H:%M:%S",
)
LOGGER = logging.getLogger("train_vasae")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the paper-facing SAE variants on text activations. "
            "The sparse dimension is fixed to the vocabulary size."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--save-dir", type=Path, default=RUNS_DIR, help="Directory for local run outputs.")
    parser.add_argument("--exp-name", default=None, help="Run directory name. Defaults to model/layer/method.")
    parser.add_argument("--model-name", default="gpt2", help="Hugging Face causal LM.")
    parser.add_argument("--layer-idx", type=int, default=11, help="Transformer layer to reconstruct.")
    parser.add_argument("--method", choices=METHODS, default="vasae_soft", help="Training method.")
    parser.add_argument("--dataset", default=DATASET_NAME, help="Hugging Face dataset name.")
    parser.add_argument("--dataset-config", default=DATASET_CONFIG, help="Hugging Face dataset config.")
    parser.add_argument("--text-column", default=TEXT_COLUMN, help="Dataset column containing text.")
    parser.add_argument("--max-length", type=int, default=128, help="Tokenization length.")
    parser.add_argument("--train-samples", type=int, default=8000, help="Training text rows.")
    parser.add_argument("--eval-samples", type=int, default=2000, help="Evaluation text rows.")
    parser.add_argument("--test-samples", type=int, default=1000, help="Final test text rows.")
    parser.add_argument("--batch-size", type=int, default=32, help="Text batch size for activation extraction.")
    parser.add_argument("--k", type=int, default=32, help="TopK active features per token.")
    parser.add_argument(
        "--anchor-coeff",
        type=float,
        default=1e-4,
        help="VASAE-soft vocabulary-anchor coefficient; 1e-4 is the paper-facing minimal-release default.",
    )
    parser.add_argument("--anchor-every", type=int, default=1, help="Apply the anchor loss every N training steps.")
    parser.add_argument("--num-epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or a torch device string.")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default=None, help="Language-model dtype.")
    return parser


def prepare_runtime(args) -> tuple[torch.device, torch.dtype | None]:
    transformers.logging.set_verbosity_error()
    torch.manual_seed(42)

    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device

    dtype = {
        None: None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    return torch.device(device_name), dtype


def transformer_layer_count(model) -> int:
    for root_name, layers_name in (("transformer", "h"), ("model", "layers"), ("gpt_neox", "layers")):
        root = getattr(model, root_name, None)
        if root is not None and hasattr(root, layers_name):
            return len(getattr(root, layers_name))

    decoder = getattr(getattr(model, "model", None), "decoder", None)
    if decoder is not None and hasattr(decoder, "layers"):
        return len(decoder.layers)
    raise ValueError(f"Cannot infer transformer layer count for {type(model).__name__}.")


def load_language_model(args, device: torch.device, dtype):
    LOGGER.info("Loading language model %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()

    layer_count = transformer_layer_count(model)
    if args.layer_idx < 0 or args.layer_idx >= layer_count:
        raise ValueError(f"--layer-idx {args.layer_idx} is outside model layer range 0..{layer_count - 1}.")

    embedding = model.get_input_embeddings()
    LOGGER.info(
        "Model hidden dim=%d, vocab=%d, layers=%d",
        embedding.weight.size(1),
        embedding.weight.size(0),
        layer_count,
    )
    return model, tokenizer, layer_count


def load_text_splits(args):
    dataset_label = f"{args.dataset}/{args.dataset_config}" if args.dataset_config else args.dataset
    LOGGER.info("Loading dataset %s", dataset_label)
    dataset_args = [args.dataset]
    if args.dataset_config:
        dataset_args.append(args.dataset_config)
    dataset = load_dataset(*dataset_args, split="train")
    if args.text_column not in dataset.column_names:
        raise ValueError(f"--text-column {args.text_column!r} not found in dataset columns {dataset.column_names}.")
    dataset = dataset.filter(lambda row: isinstance(row[args.text_column], str) and row[args.text_column].strip() != "")

    total = len(dataset)
    n_train = min(args.train_samples, total)
    n_eval = min(args.eval_samples, max(total - n_train, 0))
    n_test = min(args.test_samples, max(total - n_train - n_eval, 0))
    if n_train == 0:
        raise ValueError("No training rows are available after filtering empty text.")

    train_rows = dataset.select(range(n_train))
    eval_rows = dataset.select(range(n_train, n_train + n_eval))
    test_rows = dataset.select(range(n_train + n_eval, n_train + n_eval + n_test))
    LOGGER.info("Data split rows: train=%d eval=%d test=%d", len(train_rows), len(eval_rows), len(test_rows))
    return train_rows, eval_rows, test_rows


def build_activation_sources(args, model, tokenizer, train_rows, eval_rows, test_rows):
    nn_model = NNsight(model)

    def source(rows):
        return OnlineActivationSource(
            model=nn_model,
            tokenizer=tokenizer,
            layer_idx=args.layer_idx,
            text_dataset=rows,
            batch_size=args.batch_size,
            max_length=args.max_length,
            text_column=args.text_column,
        )

    return source(train_rows), source(eval_rows), source(test_rows)


def build_sae(args, embedding, device: torch.device) -> tuple[SAEModel, dict]:
    dim_input = embedding.weight.size(1)
    dim_sparse = embedding.weight.size(0)
    decoder_mode = "hard_tied_baseline" if args.method == "hard_tied_baseline" else "learnable"
    anchor_coeff = args.anchor_coeff if args.method == "vasae_soft" else 0.0

    model = SAEModel(
        SAEConfig(
            dim_input=dim_input,
            dim_sparse=dim_sparse,
            k=args.k,
            decoder_mode=decoder_mode,
            anchor_coeff=anchor_coeff,
            anchor_every=args.anchor_every,
        )
    ).to(device).float()

    embedding = embedding.to(device)
    if args.method == "hard_tied_baseline":
        model.attach_tied_decoder_embedding(embedding, freeze=True)
    elif args.method == "vasae_soft" and anchor_coeff > 0:
        model.attach_vocab_anchor(embedding)

    config = {
        "dim_input": model.config.dim_input,
        "dim_sparse": model.config.dim_sparse,
        "k": model.config.k,
        "nonneg_latents": model.config.nonneg_latents,
        "mse_reduction": model.config.mse_reduction,
        "use_abs_topk": model.config.use_abs_topk,
        "decoder_mode": model.config.decoder_mode,
        "anchor_coeff": model.config.anchor_coeff,
        "anchor_mode": model.config.anchor_mode,
        "anchor_topk": model.config.anchor_topk,
        "anchor_every": model.config.anchor_every,
    }
    LOGGER.info(
        "Training method=%s dim_sparse=%d k=%d decoder_mode=%s anchor_coeff=%.4g",
        args.method,
        model.config.dim_sparse,
        model.config.k,
        model.config.decoder_mode,
        model.config.anchor_coeff,
    )
    return model, config


def numeric_metrics(metrics: dict) -> dict:
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def train_and_save_best(args, trainer: Trainer, train_source, eval_source, sae_model, sae_config, checkpoint_path: Path):
    history: list[dict] = []
    best_eval_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, args.num_epochs + 1):
        LOGGER.info("Epoch %d/%d", epoch, args.num_epochs)
        train_metrics = trainer.train_epoch(train_source, epoch=epoch, num_epochs=args.num_epochs)
        eval_metrics = trainer.evaluate(eval_source) if len(eval_source) > 0 else {}

        train_numbers = numeric_metrics(train_metrics)
        eval_numbers = numeric_metrics(eval_metrics)
        history_row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_numbers.items()},
            **{f"eval_{key}": value for key, value in eval_numbers.items()},
        }
        history.append(history_row)
        LOGGER.info("Train metrics: %s", train_numbers)
        LOGGER.info("Eval metrics: %s", eval_numbers)

        eval_loss = float(eval_metrics.get("loss", train_metrics.get("loss", float("inf"))))
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": sae_model.state_dict(),
                    "sae_config": sae_config,
                    "method": args.method,
                    "epoch": epoch,
                    "metrics": eval_numbers or train_numbers,
                },
                checkpoint_path,
            )
            LOGGER.info("Saved checkpoint at epoch %d", epoch)

    if not checkpoint_path.exists():
        torch.save(
            {
                "model_state_dict": sae_model.state_dict(),
                "sae_config": sae_config,
                "method": args.method,
                "epoch": args.num_epochs,
                "metrics": {},
            },
            checkpoint_path,
        )
    return history, best_epoch, best_eval_loss


def compute_feature_stats(sae_model: SAEModel, source, device: torch.device) -> dict:
    sae_model.eval()
    feature_counts = torch.zeros(sae_model.config.dim_sparse, device=device)
    l0_sum = 0.0
    n_tokens = 0

    with torch.no_grad():
        for batch in source:
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


def write_metrics(output_dir: Path, history: list[dict], metrics_payload: dict) -> None:
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n")
    if history:
        fieldnames = sorted({key for row in history for key in row})
        with (output_dir / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device, dtype = prepare_runtime(args)

    run_name = args.exp_name or f"{args.model_name.replace('/', '_')}_L{args.layer_idx}_{args.method}"
    output_dir = args.save_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    lm, tokenizer, _ = load_language_model(args, device, dtype)
    train_rows, eval_rows, test_rows = load_text_splits(args)
    train_source, eval_source, test_source = build_activation_sources(
        args, lm, tokenizer, train_rows, eval_rows, test_rows
    )

    embedding = lm.get_input_embeddings()
    sae_model, sae_config = build_sae(args, embedding, device)
    optimizer = optim.Adam([p for p in sae_model.parameters() if p.requires_grad], lr=args.lr)
    metrics = MetricComposer([VarianceExplained()])
    trainer = Trainer(sae_model, optimizer, metrics, eval_metrics=metrics, device=str(device), logger=LOGGER)

    checkpoint_path = output_dir / "checkpoint.pt"
    history, best_epoch, best_eval_loss = train_and_save_best(
        args, trainer, train_source, eval_source, sae_model, sae_config, checkpoint_path
    )

    payload = torch.load(checkpoint_path, map_location=device)
    sae_model.load_state_dict(payload["model_state_dict"])
    test_numbers = numeric_metrics(trainer.evaluate(test_source)) if len(test_rows) > 0 else {}
    feature_stats = compute_feature_stats(sae_model, test_source, device) if len(test_rows) > 0 else {}

    write_metrics(
        output_dir,
        history,
        {
            "best_epoch": best_epoch,
            "best_eval_loss": best_eval_loss,
            "history": history,
            "test": test_numbers,
            "feature_stats": feature_stats,
        },
    )
    LOGGER.info("Final test metrics: %s", test_numbers)
    LOGGER.info("Saved checkpoint, config, and metrics to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
