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
import re
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers import AutoTokenizer

from vasae.data.schema import DataConfig
from vasae.engine.trainer import Trainer
from vasae.metrics.base import MetricComposer
from vasae.utils.log import get_logger
from vasae.utils.seed import set_seed

if TYPE_CHECKING:
    from nnsight import NNsight

logger = get_logger()


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
    p.add_argument("--test-batchsize", type=int, default=32)
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


def read_training_split_config(sae_path: Path) -> tuple[int, int, int]:
    """Read train/eval/test split sizes from a saved SAE result file.

    Parameters
    ----------
    sae_path
        Directory containing the trained SAE and its ``results.json``.

    Returns
    -------
    tuple[int, int, int]
        The train, eval, and test sample counts from the training config.

    Raises
    ------
    FileNotFoundError
        If ``results.json`` is missing.
    KeyError
        If the saved config does not contain the required split keys.
    """
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
        return (
            int(train_cfg["train_samples"]),
            int(train_cfg["eval_samples"]),
            int(train_cfg["test_samples"]),
        )
    except KeyError as e:
        raise KeyError(
            f"Training config in {train_results_path} missing key {e!r}; "
            "cannot align eval split with training."
        )


def load_llm(
    model_name: str, device: str, dtype_name: str | None, layer_idx: int
) -> tuple[nn.Module, AutoTokenizer, "NNsight", nn.Linear]:
    """Load the language model and model-dependent evaluation handles.

    Parameters
    ----------
    model_name
        Hugging Face model name or local model path.
    device
        Device used to load the model.
    dtype_name
        Optional dtype name from the CLI.
    layer_idx
        Transformer layer index to evaluate.

    Returns
    -------
    tuple
        ``(llm, tokenizer, nn_model, lm_head)`` ready for online evaluation.

    Raises
    ------
    ValueError
        If ``layer_idx`` is outside the model's layer range.
    """
    from nnsight import NNsight

    from vasae.models.factory import get_layers, get_lm_head, load_model

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_name)

    llm, tokenizer = load_model(model_name, device=device, dtype=dtype)
    nn_model = NNsight(llm)
    lm_head = get_lm_head(llm)
    n_layers = len(get_layers(llm))
    if layer_idx >= n_layers:
        raise ValueError(f"layer_idx={layer_idx} >= n_layers={n_layers}")

    return llm, tokenizer, nn_model, lm_head


def load_sae_model(sae_path: Path, llm, device: str):
    """Load a pretrained SAE and attach model-dependent embeddings.

    Parameters
    ----------
    sae_path
        Directory containing the trained SAE checkpoint.
    llm
        Language model whose embeddings should be attached when needed.
    device
        Device used to load the SAE.

    Returns
    -------
    SAEModel
        The SAE in eval mode, cast to float32, with optional embeddings attached.
    """
    from vasae.models.factory import get_embedding
    from vasae.models.sae import SAEModel

    emb = get_embedding(llm)
    sae_model = SAEModel.from_pretrained(sae_path).to(device)
    sae_model.eval()

    if sae_model.config.tied_decoder:
        sae_model.attach_embedding(emb, freeze=True)
    if sae_model.config.anchor_coeff > 0:
        sae_model.attach_anchor_embedding(emb)

    return sae_model.float()


def build_test_source(
    data_cfg: DataConfig,
    sae_path: Path,
    nn_model: "NNsight",
    tokenizer: AutoTokenizer,
    layer_idx: int,
    n_train_split: int,
    n_eval_split: int,
    n_test_request: int,
):
    """Build the online activation source for the training-aligned test split.

    Parameters
    ----------
    data_cfg
        Dataset configuration (source, batching, max length).
    sae_path
        Directory containing the trained SAE.
    nn_model
        NNsight-wrapped language model used to extract activations.
    tokenizer
        Tokenizer used by the online activation source.
    layer_idx
        Transformer layer index whose activations should be extracted.
    n_train_split
        Number of training samples used by the original training run.
    n_eval_split
        Number of eval samples used by the original training run.
    n_test_request
        Requested number of test samples.

    Returns
    -------
    tuple
        ``(test_source, n_test)`` where ``n_test`` is the actual selected size.

    Raises
    ------
    ValueError
        If the saved train/eval split consumes the whole dataset.
    """
    from datasets import load_dataset, load_from_disk

    from vasae.data.activation_source import OnlineActivationSource

    save_dir_root = sae_path.parent
    ds_cache_name = (
        f"{data_cfg.dataset}_{data_cfg.dataset_config or 'default'}".replace("/", "_")
    )
    data_cache_dir = save_dir_root / ".data_cache" / ds_cache_name

    if (data_cache_dir / "dataset_info.json").exists():
        logger.info(f"Loading cached dataset from {data_cache_dir}")
        ds = load_from_disk(str(data_cache_dir))
    else:
        logger.info(f"Loading dataset {data_cfg.dataset}...")
        ds = load_dataset(data_cfg.dataset, data_cfg.dataset_config, split="train")
        ds = ds.filter(lambda x: len(x["text"].strip()) > 50)

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
        batch_size=data_cfg.test_batchsize,
        max_length=data_cfg.max_length,
    )
    return test_source, n_test


def build_test_metrics(nn_model, lm_head, layer_idx: int) -> MetricComposer:
    """Build the metric composer used for online SAE test evaluation.

    Parameters
    ----------
    nn_model
        NNsight-wrapped language model used by CE loss recovery.
    lm_head
        Language model head used by logit lens accuracy.
    layer_idx
        Transformer layer index to evaluate.

    Returns
    -------
    MetricComposer
        Composer with logit lens accuracy, variance explained, and CE recovery.
    """
    from vasae.metrics.activity import ActivityStats
    from vasae.metrics.ce_loss import CELossRecovered
    from vasae.metrics.logitlens import LogitLens, LogitLensAccMetric
    from vasae.metrics.variance_explained import VarianceExplained

    return MetricComposer(
        [
            LogitLensAccMetric(LogitLens(lm_head)),
            VarianceExplained(),
            CELossRecovered(nn_model, layer_idx=layer_idx),
            ActivityStats(),
        ]
    )


def save_eval_results(
    results_path: Path,
    config: dict,
    test_out: dict,
) -> None:
    """Save online SAE evaluation results.

    Parameters
    ----------
    results_path
        Output JSON path.
    config
        Evaluation configuration to write under the ``config`` key.
    test_out
        Metrics returned by ``Trainer.evaluate``.
    """
    results = {
        "config": config,
        "test": {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in test_out.items()
        },
        "dead_rate": test_out["dead_rate"],
        "l0": test_out["l0"],
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    sae_path = Path(args.sae_path)

    if args.layer_idx is not None:
        layer_idx = args.layer_idx
    else:
        layer_idx = parse_layer_from_dirname(sae_path.name)

    logger.info(f"Evaluating SAE: {sae_path}")
    logger.info(f"Layer: {layer_idx}, Model: {args.model_name}")

    n_train_split, n_eval_split, n_test_cfg = read_training_split_config(sae_path)
    n_test_request = args.test_samples if args.test_samples is not None else n_test_cfg

    data_cfg = DataConfig(
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        max_length=args.max_length,
        test_batchsize=args.test_batchsize,
    )

    # --- Lazy imports ---
    import datasets
    import transformers

    # datasets.disable_progress_bars()
    # transformers.logging.set_verbosity_error()

    logger.info(f"Loading {args.model_name}...")
    llm, tokenizer, nn_model, lm_head = load_llm(
        args.model_name, device, args.dtype, layer_idx
    )

    logger.info("Loading SAE...")
    sae_model = load_sae_model(sae_path, llm, device)
    logger.info(
        f"SAE config: dim_input={sae_model.config.dim_model}, "
        f"dim_sparse={sae_model.config.dim_sparse}, "
        f"tied={sae_model.config.tied_decoder}, k={sae_model.config.k}"
    )

    test_source, n_test = build_test_source(
        data_cfg,
        sae_path,
        nn_model,
        tokenizer,
        layer_idx=layer_idx,
        n_train_split=n_train_split,
        n_eval_split=n_eval_split,
        n_test_request=n_test_request,
    )

    test_metrics = build_test_metrics(nn_model, lm_head, layer_idx)
    trainer = Trainer(
        sae_model=sae_model,
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

    logger.info(
        f"Dead feature rate: {test_out['dead_rate']:.4f}, "
        f"L0: {test_out['l0']:.2f} "
        f"(over {test_out['n_samples']} samples)"
    )

    config = {
        "sae_path": str(sae_path),
        "model_name": args.model_name,
        "layer_idx": layer_idx,
        "test_samples": n_test,
        "test_batchsize": data_cfg.test_batchsize,
        "max_length": data_cfg.max_length,
        "dataset": data_cfg.dataset,
        "dataset_config": data_cfg.dataset_config,
        "seed": args.seed,
        "split_train_samples": n_train_split,
        "split_eval_samples": n_eval_split,
    }
    results_path = sae_path / "results_eval.json"
    save_eval_results(results_path, config, test_out)


if __name__ == "__main__":
    main()
