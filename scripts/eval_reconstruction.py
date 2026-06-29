"""Evaluate reconstruction quality for a saved paper reproduction SAE checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained SAE/VASAE checkpoint and compute reconstruction MSE "
            "and variance explained. CE recovery and logit-lens agreement are "
            "optional because they require language-model forward passes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt or a run directory containing it.")
    parser.add_argument("--model-name", default="gpt2", help="Hugging Face causal LM used for activation extraction.")
    parser.add_argument("--layer-idx", type=int, default=11, help="Transformer layer to evaluate.")
    parser.add_argument("--dataset", default="wikitext", help="Hugging Face dataset name.")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1", help="Dataset config name; use empty string for none.")
    parser.add_argument("--text-column", default="text", help="Dataset text column.")
    parser.add_argument("--max-length", type=int, default=128, help="Tokenization length.")
    parser.add_argument("--samples", type=int, default=1000, help="Evaluation text rows.")
    parser.add_argument("--batch-size", type=int, default=32, help="Text batch size.")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or a torch device string.")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default=None, help="Language-model dtype.")
    parser.add_argument("--output-dir", default=None, help="Directory for reconstruction_metrics.json/csv.")
    parser.add_argument("--logit-lens", action="store_true", help="Compute logit-lens top-1 agreement.")
    parser.add_argument("--ce-recovery", action="store_true", help="Compute CE recovery with identity, SAE, and zero-patched runs.")
    return parser


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s][%(asctime)s][%(name)s] %(message)s",
        datefmt="%Y%m%d %H:%M:%S",
    )
    return logging.getLogger("eval_reconstruction")


def quiet_external_progress() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")


def checkpoint_file(path: Path) -> Path:
    if path.is_dir():
        path = path / "checkpoint.pt"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


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


def load_text_rows(args, logger: logging.Logger):
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
    return dataset.select(range(min(args.samples, len(dataset))))


def load_sae(checkpoint_path: Path, device):
    import torch

    from vasae.models import SAEConfig, SAEModel

    payload = torch.load(checkpoint_path, map_location=device)
    config = SAEConfig(**payload["sae_config"])
    model = SAEModel(config).to(device).float()
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def update_reconstruction_sums(sums: dict, activations, reconstruction) -> None:
    error = reconstruction - activations
    flat = activations.reshape(-1, activations.size(-1))
    sums["sse"] += error.pow(2).sum().item()
    sums["n_elements"] += activations.numel()
    sums["n_tokens"] += flat.size(0)
    sums["sum_x"] += flat.sum(dim=0)
    sums["sum_x2"] += flat.pow(2).sum(dim=0)


def finalize_reconstruction_metrics(sums: dict) -> dict:
    var_sum = (sums["sum_x2"] - sums["sum_x"].pow(2) / max(sums["n_tokens"], 1)).sum().item()
    mse = sums["sse"] / max(sums["n_elements"], 1)
    variance_explained = 1.0 - sums["sse"] / max(var_sum, 1e-8)
    return {
        "mse": float(mse),
        "variance_explained": float(variance_explained),
        "tokens_evaluated": int(sums["n_tokens"]),
    }


def write_outputs(output_dir: Path, results: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reconstruction_metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    metric_row = {**results["config"], **results["metrics"]}
    with (output_dir / "reconstruction_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_row))
        writer.writeheader()
        writer.writerow(metric_row)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    quiet_external_progress()
    logger = setup_logger()

    import torch
    import transformers
    from nnsight import NNsight

    transformers.logging.set_verbosity_error()

    from vasae.data.activation_source import OnlineActivationSource
    from vasae.metrics.ce_loss import CELossRecovered

    checkpoint_path = checkpoint_file(Path(args.checkpoint))
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    sae_model, checkpoint_payload = load_sae(checkpoint_path, device)
    logger.info("Loaded checkpoint %s", checkpoint_path)

    lm, tokenizer = load_lm_and_tokenizer(args.model_name, device, dtype)
    nn_model = NNsight(lm)
    dataset = load_text_rows(args, logger)
    source = OnlineActivationSource(
        model=nn_model,
        tokenizer=tokenizer,
        layer_idx=args.layer_idx,
        text_dataset=dataset,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    sums = {
        "sse": 0.0,
        "n_elements": 0,
        "n_tokens": 0,
        "sum_x": torch.zeros(sae_model.config.dim_input, device=device),
        "sum_x2": torch.zeros(sae_model.config.dim_input, device=device),
    }
    logit_lens_correct = 0
    logit_lens_total = 0
    ce_sums: dict[str, float] = {}
    ce_batches = 0
    lm_head = lm.get_output_embeddings() if args.logit_lens else None
    ce_metric = CELossRecovered(nn_model, args.layer_idx) if args.ce_recovery else None

    with torch.no_grad():
        for batch_idx, batch in enumerate(source, start=1):
            activations = batch["activations"].to(device).float()
            mask = batch.get("attention_mask")
            if mask is not None:
                activations = activations[mask.bool()]
            output = sae_model(activations)
            reconstruction = output.hidden_states_recon
            update_reconstruction_sums(sums, activations, reconstruction)

            if lm_head is not None:
                head_weight = getattr(lm_head, "weight", None)
                head_device = head_weight.device if head_weight is not None else device
                head_dtype = head_weight.dtype if head_weight is not None else activations.dtype
                original_ids = lm_head(activations.to(device=head_device, dtype=head_dtype)).argmax(dim=-1)
                recon_ids = lm_head(reconstruction.to(device=head_device, dtype=head_dtype)).argmax(dim=-1)
                logit_lens_correct += (original_ids == recon_ids).sum().item()
                logit_lens_total += original_ids.numel()

            if ce_metric is not None:
                ce_values = ce_metric.compute(
                    {
                        "input_ids": batch["input_ids"],
                        "attention_mask": batch["attention_mask"],
                        "sae_model": sae_model,
                    }
                )
                for key, value in ce_values.items():
                    ce_sums[key] = ce_sums.get(key, 0.0) + float(value)
                ce_batches += 1

            if batch_idx % 10 == 0:
                logger.info("Evaluated %d batches", batch_idx)

    metrics = finalize_reconstruction_metrics(sums)
    if logit_lens_total > 0:
        metrics["logitlens_acc"] = float(logit_lens_correct / logit_lens_total)
    if ce_batches > 0:
        metrics.update({key: value / ce_batches for key, value in ce_sums.items()})

    results = {
        "config": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_method": checkpoint_payload.get("method"),
            "model_name": args.model_name,
            "layer_idx": args.layer_idx,
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "samples": len(dataset),
            "batch_size": args.batch_size,
            "logit_lens": args.logit_lens,
            "ce_recovery": args.ce_recovery,
        },
        "metrics": metrics,
    }
    write_outputs(output_dir, results)
    logger.info("Saved reconstruction metrics to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
