"""Evaluate a pre-trained SAE model on the same test split used at training time.

Mirrors the final-test section of `scripts/training/train_sae_online.py`:
loads the saved SAE, rebuilds the OnlineActivationSource on the test split
(offsets read from the training `results.json`), and runs Trainer.evaluate with
the same metric composer (LogitLens + VarianceExplained + CELossRecovered),
followed by dead-feature-rate / L0 over the test set.

Example:
    python scripts/eval/eval_sae_online.py \
        --sae-path /scratch/.../009_online_gpt2_L11_k32_a0 \
        --model-name gpt2 --layer-idx 11
"""

import argparse
import json
import os
import re

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

from pathlib import Path

import torch
import torch.optim as optim

from vasae.utils.log import get_logger
from vasae.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a pre-trained SAE model")
    p.add_argument(
        "--sae-path",
        type=str,
        required=True,
        help="Path to SAE model directory (config.json + model.safetensors)",
    )
    p.add_argument("--model-name", type=str, default="gpt2")
    p.add_argument(
        "--layer-idx",
        type=int,
        default=None,
        help="Layer index (if None, parse from directory name)",
    )
    p.add_argument(
        "--dtype", type=str, default=None, choices=["float16", "bfloat16", "float32"]
    )
    p.add_argument(
        "--test-samples",
        type=int,
        default=None,
        help="Override test_samples from training config",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dataset", type=str, default="wikitext")
    p.add_argument("--dataset-config", type=str, default="wikitext-103-raw-v1")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_layer_from_dirname(dirname: str) -> int:
    """Extract layer index from directory name like '009_online_gpt2_L11_k32_a0'."""
    match = re.search(r"_L(\d+)_", dirname)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot parse layer index from directory name: {dirname}")


def main():
    args = parse_args()
    set_seed(args.seed)
    logger = get_logger()
    device = args.device
    sae_path = Path(args.sae_path)

    if args.layer_idx is not None:
        layer_idx = args.layer_idx
    else:
        layer_idx = parse_layer_from_dirname(sae_path.name)

    logger.info(f"Evaluating SAE: {sae_path}")
    logger.info(f"Layer: {layer_idx}, Model: {args.model_name}")

    # --- Read training config (for split offsets) ---
    train_results_path = sae_path / "results.json"
    if not train_results_path.exists():
        raise FileNotFoundError(
            f"{train_results_path} not found; cannot infer train/eval/test split. "
            "Re-run training to produce results.json or pass split sizes explicitly."
        )
    with open(train_results_path) as f:
        train_results = json.load(f)
    train_cfg = train_results.get("config", {})
    try:
        n_train_split = int(train_cfg["train_samples"])
        n_eval_split = int(train_cfg["eval_samples"])
        n_test_cfg = int(train_cfg["test_samples"])
    except KeyError as e:
        raise KeyError(
            f"Training config in {train_results_path} missing key {e!r}; "
            "cannot align eval split with training."
        )
    n_test_request = args.test_samples if args.test_samples is not None else n_test_cfg

    # --- Lazy imports ---
    import datasets
    import transformers
    from datasets import load_dataset, load_from_disk
    from nnsight import NNsight

    datasets.disable_progress_bars()
    transformers.logging.set_verbosity_error()

    from vasae.data.activation_source import OnlineActivationSource
    from vasae.engine.trainer import Trainer
    from vasae.metrics.base import MetricComposer
    from vasae.metrics.ce_loss import CELossRecovered
    from vasae.metrics.logitlens import LogitLens, LogitLensAccMetric
    from vasae.metrics.variance_explained import VarianceExplained
    from vasae.models.factory import get_embedding, get_layers, get_lm_head, load_model
    from vasae.models.sae import SAEModel

    # --- Load LLM ---
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype)

    logger.info(f"Loading {args.model_name}...")
    llm, tokenizer = load_model(args.model_name, device=device, dtype=dtype)
    nn_model = NNsight(llm)
    emb = get_embedding(llm)
    lm_head = get_lm_head(llm)
    n_layers = len(get_layers(llm))
    if layer_idx >= n_layers:
        raise ValueError(f"layer_idx={layer_idx} >= n_layers={n_layers}")

    # --- Load SAE ---
    logger.info("Loading SAE...")
    sae_model = SAEModel.from_pretrained(sae_path).to(device)
    sae_model.eval()

    if sae_model.config.tied_decoder:
        sae_model.attach_embedding(emb, freeze=True)
    if sae_model.config.anchor_coeff > 0:
        sae_model.attach_anchor_embedding(emb)
    sae_model = sae_model.float()

    logger.info(
        f"SAE config: dim_input={sae_model.config.dim_model}, "
        f"dim_sparse={sae_model.config.dim_sparse}, "
        f"tied={sae_model.config.tied_decoder}, k={sae_model.config.k}"
    )

    # --- Load dataset (shared cache, same naming as train_sae_online.py) ---
    save_dir_root = sae_path.parent
    ds_cache_name = f"{args.dataset}_{args.dataset_config or 'default'}".replace(
        "/", "_"
    )
    data_cache_dir = save_dir_root / ".data_cache" / ds_cache_name

    if (data_cache_dir / "dataset_info.json").exists():
        logger.info(f"Loading cached dataset from {data_cache_dir}")
        ds = load_from_disk(str(data_cache_dir))
    else:
        logger.info(f"Loading dataset {args.dataset}...")
        ds = load_dataset(args.dataset, args.dataset_config, split="train")
        ds = ds.filter(lambda x: len(x["text"].strip()) > 50)

    # --- Same split layout as training: skip train+eval, take test ---
    n_total = len(ds)
    n_skip = n_train_split + n_eval_split
    if n_skip >= n_total:
        raise ValueError(
            f"train+eval ({n_skip}) >= dataset size ({n_total}); no test samples left."
        )
    n_test = min(n_test_request, n_total - n_skip)
    test_ds = ds.select(range(n_skip, n_skip + n_test))
    logger.info(
        f"Data split: skip={n_skip} (train={n_train_split}+eval={n_eval_split}), "
        f"test={n_test}"
    )

    test_source = OnlineActivationSource(
        model=nn_model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        text_dataset=test_ds,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # --- Metrics (same composition as training final test) ---
    test_metrics = MetricComposer(
        [
            LogitLensAccMetric(LogitLens(lm_head)),
            VarianceExplained(),
            CELossRecovered(nn_model, layer_idx=layer_idx),
        ]
    )

    # Trainer.evaluate doesn't touch the optimizer, but constructor requires one.
    dummy_optimizer = optim.SGD(
        [p for p in sae_model.parameters() if p.requires_grad], lr=0.0
    )
    trainer = Trainer(
        sae_model=sae_model,
        optimizer=dummy_optimizer,
        metrics=test_metrics,
        eval_metrics=test_metrics,
        device=device,
        logger=logger,
    )

    logger.info("=== Test ===")
    test_out = trainer.evaluate(test_source)
    logger.info(
        f"[Test] loss={test_out['loss']:.4f} "
        f"VE={test_out.get('variance_explained', 0):.4f} "
        f"logitlens={test_out.get('logitlens_acc', 0) * 100:.2f}% "
        f"CE_recovered={test_out.get('loss_recovered', 0):.4f}"
    )

    # --- Dead feature rate & L0 over the test set ---
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

    # --- Save ---
    results = {
        "config": {
            "sae_path": str(sae_path),
            "model_name": args.model_name,
            "layer_idx": layer_idx,
            "test_samples": n_test,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "seed": args.seed,
            "split_train_samples": n_train_split,
            "split_eval_samples": n_eval_split,
        },
        "test": {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in test_out.items()
        },
        "dead_rate": dead_rate,
        "l0": l0,
    }
    results_path = sae_path / "results_eval.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
