"""Evaluate reconstruction quality for a saved paper SAE checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import datasets
import torch
import transformers
from datasets import load_dataset
from nnsight import NNsight
from transformers import AutoModelForCausalLM, AutoTokenizer

from vasae.data import OnlineActivationSource
from vasae.metrics import CELossRecovered
from vasae.models import SAEConfig, SAEModel


DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
TEXT_COLUMN = "text"

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(asctime)s][%(name)s] %(message)s",
    datefmt="%Y%m%d %H:%M:%S",
)
LOGGER = logging.getLogger("eval_reconstruction")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate reconstruction MSE, variance explained, and optional LM-facing metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt or a run directory containing it.")
    parser.add_argument("--model-name", default="gpt2", help="Hugging Face causal LM used for activation extraction.")
    parser.add_argument("--layer-idx", type=int, default=11, help="Transformer layer to evaluate.")
    parser.add_argument("--dataset", default=DATASET_NAME, help="Hugging Face dataset name.")
    parser.add_argument("--dataset-config", default=DATASET_CONFIG, help="Hugging Face dataset config.")
    parser.add_argument("--text-column", default=TEXT_COLUMN, help="Dataset column containing text.")
    parser.add_argument("--max-length", type=int, default=128, help="Tokenization length.")
    parser.add_argument("--samples", type=int, default=1000, help="Evaluation text rows.")
    parser.add_argument("--batch-size", type=int, default=32, help="Text batch size.")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or a torch device string.")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default=None, help="Language-model dtype.")
    parser.add_argument("--logit-lens", action="store_true", help="Compute logit-lens top-1 agreement.")
    parser.add_argument("--ce-recovery", action="store_true", help="Compute CE recovery with identity, SAE, and zero-patched runs.")
    return parser


def prepare_runtime(args) -> tuple[torch.device, torch.dtype | None]:
    datasets.disable_progress_bars()
    transformers.logging.set_verbosity_error()

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


def checkpoint_file(path: str) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def load_sae(checkpoint_path: Path, device: torch.device) -> tuple[SAEModel, dict]:
    payload = torch.load(checkpoint_path, map_location=device)
    model = SAEModel(SAEConfig(**payload["sae_config"])).to(device).float()
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    LOGGER.info("Loaded checkpoint %s", checkpoint_path)
    return model, payload


def load_eval_source(args, device: torch.device, dtype):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lm = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
    lm.to(device)
    lm.eval()
    nn_model = NNsight(lm)

    dataset_label = f"{args.dataset}/{args.dataset_config}" if args.dataset_config else args.dataset
    LOGGER.info("Loading dataset %s", dataset_label)
    dataset_args = [args.dataset]
    if args.dataset_config:
        dataset_args.append(args.dataset_config)
    dataset = load_dataset(*dataset_args, split="train")
    if args.text_column not in dataset.column_names:
        raise ValueError(f"--text-column {args.text_column!r} not found in dataset columns {dataset.column_names}.")
    dataset = dataset.filter(lambda row: isinstance(row[args.text_column], str) and row[args.text_column].strip() != "")
    dataset = dataset.select(range(min(args.samples, len(dataset))))

    source = OnlineActivationSource(
        model=nn_model,
        tokenizer=tokenizer,
        layer_idx=args.layer_idx,
        text_dataset=dataset,
        batch_size=args.batch_size,
        max_length=args.max_length,
        text_column=args.text_column,
    )
    return lm, nn_model, dataset, source


def accumulate_reconstruction_metrics(args, sae_model, lm, nn_model, source, device: torch.device) -> dict:
    sse = 0.0
    n_elements = 0
    n_tokens = 0
    sum_x = torch.zeros(sae_model.config.dim_input, device=device)
    sum_x2 = torch.zeros(sae_model.config.dim_input, device=device)

    logit_lens_correct = 0
    logit_lens_total = 0
    lm_head = lm.get_output_embeddings() if args.logit_lens else None

    ce_sums: dict[str, float] = {}
    ce_batches = 0
    ce_metric = CELossRecovered(nn_model, args.layer_idx) if args.ce_recovery else None

    with torch.no_grad():
        for batch_idx, batch in enumerate(source, start=1):
            activations = batch["activations"].to(device).float()
            mask = batch.get("attention_mask")
            if mask is not None:
                activations = activations[mask.bool()]

            reconstruction = sae_model(activations).hidden_states_recon
            error = reconstruction - activations
            flat = activations.reshape(-1, activations.size(-1))
            sse += error.pow(2).sum().item()
            n_elements += activations.numel()
            n_tokens += flat.size(0)
            sum_x += flat.sum(dim=0)
            sum_x2 += flat.pow(2).sum(dim=0)

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
                LOGGER.info("Evaluated %d batches", batch_idx)

    var_sum = (sum_x2 - sum_x.pow(2) / max(n_tokens, 1)).sum().item()
    metrics = {
        "mse": float(sse / max(n_elements, 1)),
        "variance_explained": float(1.0 - sse / max(var_sum, 1e-8)),
        "tokens_evaluated": int(n_tokens),
    }
    if logit_lens_total > 0:
        metrics["logitlens_acc"] = float(logit_lens_correct / logit_lens_total)
    if ce_batches > 0:
        metrics.update({key: value / ce_batches for key, value in ce_sums.items()})
    return metrics


def write_results(checkpoint_path: Path, metrics: dict) -> None:
    output_dir = checkpoint_path.parent
    (output_dir / "reconstruction_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    with (output_dir / "reconstruction_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    LOGGER.info("Saved reconstruction metrics to %s", output_dir)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device, dtype = prepare_runtime(args)
    checkpoint_path = checkpoint_file(args.checkpoint)
    sae_model, _ = load_sae(checkpoint_path, device)
    lm, nn_model, _, source = load_eval_source(args, device, dtype)
    metrics = accumulate_reconstruction_metrics(args, sae_model, lm, nn_model, source, device)
    write_results(checkpoint_path, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
