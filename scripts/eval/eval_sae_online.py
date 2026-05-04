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
import logging
import re
from pathlib import Path

from vasae.data.online_sources import load_hf_text_dataset
from vasae.data.schema import DataConfig
from vasae.models.online import attach_sae_embeddings, load_online_llm
from vasae.utils.log import get_logger
from vasae.utils.seed import set_seed

logger = logging.getLogger(__name__)

CORPUS_CHOICES = ("fineweb", "dclm", "pile")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a pre-trained SAE model")
    p.add_argument(
        "--sae-path",
        type=str,
        required=True,
        help="Path to SAE model directory (config.json + model.safetensors)",
    )
    p.add_argument(
        "--data-source",
        type=str,
        default="auto",
        choices=["auto", "hf", "jsonl"],
        help="Evaluation data source. auto infers from training results.json when possible.",
    )
    p.add_argument("--model-name", type=str, default=None)
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
    p.add_argument(
        "--corpus",
        choices=CORPUS_CHOICES,
        default=None,
        help="Heldout corpus to evaluate for --data-source jsonl.",
    )
    p.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Root directory containing <corpus>/raw/heldout.jsonl for --data-source jsonl.",
    )
    p.add_argument(
        "--eval-tokens",
        type=int,
        default=1_000_000,
        help="Token budget for JSONL heldout evaluation.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    return value


def save_eval_results(results_path: Path, config: dict, test_out: dict):
    results = {
        "config": jsonable(config),
        "test": {
            key: float(value) if isinstance(value, (int, float)) else jsonable(value)
            for key, value in test_out.items()
        },
        "dead_rate": test_out["dead_rate"],
        "l0": test_out["l0"],
    }
    with results_path.open("w") as handle:
        json.dump(results, handle, indent=2)
    logger.info(f"Results saved to {results_path}")


def read_training_config(sae_path: Path) -> dict:
    with (sae_path / "results.json").open() as handle:
        return json.load(handle)["config"]


def infer_data_source(requested: str, train_cfg: dict) -> str:
    if requested == "auto":
        return train_cfg["data_source"]
    return requested


def parse_layer_from_dirname(dirname: str) -> int:
    return int(re.search(r"_L(\d+)_", dirname).group(1))


def read_training_split_config(sae_path: Path) -> tuple[int, int, int]:
    train_cfg = read_training_config(sae_path)
    return (
        int(train_cfg["train_samples"]),
        int(train_cfg["eval_samples"]),
        int(train_cfg["test_samples"]),
    )


def load_sae_for_online_eval(sae_path: Path, embedding, device: str):
    from vasae.models.sae import SAEModel

    sae_model = SAEModel.from_pretrained(sae_path).to(device)
    sae_model.eval()
    attach_sae_embeddings(sae_model, embedding, freeze_decoder=True)
    return sae_model.float()


def build_hf_test_source(
    *,
    data_cfg,
    cache_root,
    nn_model,
    tokenizer,
    layer_idx: int,
    n_train_split: int,
    n_eval_split: int,
    n_test_request: int,
):
    from vasae.data.activation_source import OnlineActivationSource

    ds = load_hf_text_dataset(data_cfg, cache_root)
    n_total = len(ds)
    n_skip = n_train_split + n_eval_split
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


def build_jsonl_heldout_source(
    *,
    corpus: str,
    corpus_dir: Path,
    eval_tokens: int,
    nn_model,
    tokenizer,
    layer_idx: int,
    batch_size: int,
    max_length: int,
):
    from vasae.data.corpus_windows import HeldoutCorpusSource, corpus_jsonl

    return HeldoutCorpusSource(
        model=nn_model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        jsonl_path=corpus_jsonl(corpus_dir, corpus, "heldout"),
        token_budget=eval_tokens,
        batch_size=batch_size,
        max_length=max_length,
    )


def build_test_metrics(nn_model, lm_head, layer_idx: int):
    from vasae.metrics.activity import ActivityStats
    from vasae.metrics.base import MetricComposer
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


def main():
    args = parse_args()
    set_seed(args.seed)
    get_logger()
    device = args.device
    sae_path = Path(args.sae_path)
    train_cfg = read_training_config(sae_path)
    data_source = infer_data_source(args.data_source, train_cfg)

    if args.model_name is not None:
        model_name = args.model_name
    else:
        model_name = train_cfg["model_name"]

    if args.layer_idx is not None:
        layer_idx = args.layer_idx
    elif "layer_idx" in train_cfg:
        layer_idx = int(train_cfg["layer_idx"])
    else:
        layer_idx = parse_layer_from_dirname(sae_path.name)

    logger.info(f"Evaluating SAE: {sae_path}")
    logger.info(f"Layer: {layer_idx}, Model: {model_name}, Data source: {data_source}")

    # --- Lazy imports ---
    import transformers

    if data_source == "hf":
        import datasets

        datasets.disable_progress_bars()
    transformers.logging.set_verbosity_error()

    from vasae.engine.trainer import Trainer

    llm_ctx = load_online_llm(
        model_name,
        device=device,
        dtype_name=args.dtype,
        layer_idx=layer_idx,
    )

    logger.info("Loading SAE...")
    sae_model = load_sae_for_online_eval(sae_path, llm_ctx.embedding, device)
    logger.info(
        f"SAE config: dim_input={sae_model.config.dim_model}, "
        f"dim_sparse={sae_model.config.dim_sparse}, "
        f"tied={sae_model.config.tied_decoder}, k={sae_model.config.k}"
    )

    if data_source == "hf":
        n_train_split, n_eval_split, n_test_cfg = read_training_split_config(sae_path)
        n_test_request = (
            args.test_samples if args.test_samples is not None else n_test_cfg
        )
        data_cfg = DataConfig(
            dataset=args.dataset,
            dataset_config=args.dataset_config,
            max_length=args.max_length,
            test_batchsize=args.test_batchsize,
        )
        test_source, n_test = build_hf_test_source(
            data_cfg=data_cfg,
            cache_root=sae_path.parent,
            nn_model=llm_ctx.nn_model,
            tokenizer=llm_ctx.tokenizer,
            layer_idx=layer_idx,
            n_train_split=n_train_split,
            n_eval_split=n_eval_split,
            n_test_request=n_test_request,
        )
        config = {
            "data_source": data_source,
            "sae_path": str(sae_path),
            "model_name": model_name,
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
    else:
        from vasae.data.corpus_windows import default_corpus_dir

        corpus_dir = (
            args.corpus_dir
            or (Path(train_cfg["corpus_dir"]) if train_cfg.get("corpus_dir") else None)
            or default_corpus_dir()
        )
        test_source = build_jsonl_heldout_source(
            corpus=args.corpus,
            corpus_dir=corpus_dir,
            eval_tokens=args.eval_tokens,
            nn_model=llm_ctx.nn_model,
            tokenizer=llm_ctx.tokenizer,
            layer_idx=layer_idx,
            batch_size=args.test_batchsize,
            max_length=args.max_length,
        )
        config = {
            "data_source": data_source,
            "sae_path": str(sae_path),
            "model_name": model_name,
            "layer_idx": layer_idx,
            "corpus": args.corpus,
            "eval_tokens": args.eval_tokens,
            "corpus_dir": corpus_dir,
            "batch_size": args.test_batchsize,
            "max_length": args.max_length,
            "seed": args.seed,
        }
        results_path = sae_path / f"results_eval_{args.corpus}.json"

    test_metrics = build_test_metrics(
        llm_ctx.nn_model,
        llm_ctx.lm_head,
        layer_idx,
    )
    trainer = Trainer(
        sae_model=sae_model,
        metrics=test_metrics,
        eval_metrics=test_metrics,
        device=device,
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

    save_eval_results(results_path, config, test_out)


if __name__ == "__main__":
    main()
