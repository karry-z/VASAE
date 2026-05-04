"""Online SAE training with nnsight — supports any HuggingFace causal LM.

Extracts activations on-the-fly, trains a vocab-aligned SAE,
and saves training/eval metrics for later test-time evaluation.

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
import logging
from collections.abc import Sequence

# Disable progress bars before any HF/tqdm imports
# os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# os.environ["TQDM_DISABLE"] = "1"

# Preload CUDA libraries before torch import (some nodes lack LD_LIBRARY_PATH)
# import ctypes
# import site

# _sp = site.getsitepackages()[0]
# for _lib in [
#     "nvidia/cusparselt/lib/libcusparseLt.so.0",
#     "nvidia/cusparse/lib/libcusparse.so.12",
# ]:
#     _path = os.path.join(_sp, _lib)
#     if os.path.exists(_path):
#         ctypes.CDLL(_path)

from pathlib import Path
from typing import TYPE_CHECKING

import torch.optim as optim

from vasae.data.online_sources import load_hf_text_dataset
from vasae.models.online import attach_sae_embeddings, load_online_llm
from vasae.utils.log import get_logger
from vasae.utils.seed import set_seed

if TYPE_CHECKING:
    from nnsight import NNsight
    from transformers import PreTrainedTokenizerBase

    from vasae.data.corpus_windows import BalancedMixtureSource
    from vasae.data.schema import DataConfig

logger = get_logger()

CORPUS_CHOICES = ("fineweb", "dclm", "pile")


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
    p.add_argument(
        "--data-source",
        type=str,
        default="hf",
        choices=["hf", "jsonl"],
        help="Training data source: HuggingFace dataset split or local JSONL corpus mixture.",
    )
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
    p.add_argument("--valid-batchsize", type=int, default=32)
    p.add_argument("--test-batchsize", type=int, default=32)
    p.add_argument("--train-samples", type=int, default=8000)
    p.add_argument("--eval-samples", type=int, default=2000)
    p.add_argument("--test-samples", type=int, default=1000)
    p.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Root directory containing <corpus>/raw/{train,heldout}.jsonl for --data-source jsonl.",
    )
    p.add_argument(
        "--corpora",
        nargs="+",
        choices=CORPUS_CHOICES,
        default=list(CORPUS_CHOICES),
        help="Corpora to use for balanced JSONL mixture training.",
    )
    p.add_argument(
        "--train-tokens",
        type=int,
        default=200_000_000,
        help="Total token budget for balanced JSONL mixture training.",
    )
    p.add_argument(
        "--valid-tokens",
        type=int,
        default=300_000,
        help="Total token budget for balanced JSONL mixture validation.",
    )

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


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    return value


def save_training_results(
    results_path: Path,
    config: dict,
    stopped_epoch: int,
    eval_out: dict,
):
    results = {
        "config": jsonable(config),
        "stopped_epoch": stopped_epoch,
        "last_eval": {
            key: float(value) if isinstance(value, (int, float)) else value
            for key, value in eval_out.items()
        },
    }
    with results_path.open("w") as handle:
        json.dump(results, handle, indent=2)
    logger.info(f"Results saved to {results_path}")


def infer_dim_sparse(
    requested_dim: int,
    tied_decoder: bool,
    dim_model: int,
    vocab_size: int,
) -> int:
    if requested_dim > 0:
        return requested_dim
    if tied_decoder:
        return vocab_size
    return 8 * dim_model


def build_hf_train_eval_sources(
    *,
    data_cfg,
    cache_root,
    nn_model,
    tokenizer,
    layer_idx: int,
    train_samples: int,
    eval_samples: int,
    test_samples: int,
):
    from vasae.data.activation_source import OnlineActivationSource

    ds = load_hf_text_dataset(data_cfg, cache_root)
    n_total = len(ds)
    n_train = min(train_samples, n_total)
    n_eval = min(eval_samples, n_total - n_train)
    n_test = min(test_samples, n_total - n_train - n_eval)
    train_ds = ds.select(range(n_train))
    eval_ds = ds.select(range(n_train, n_train + n_eval))
    logger.info(f"Data split: train={n_train}, eval={n_eval}, test={n_test}")

    train_source = OnlineActivationSource(
        model=nn_model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        text_dataset=train_ds,
        batch_size=data_cfg.train_batchsize,
        max_length=data_cfg.max_length,
    )
    valid_source = OnlineActivationSource(
        model=nn_model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        text_dataset=eval_ds,
        batch_size=data_cfg.valid_batchsize,
        max_length=data_cfg.max_length,
    )
    return (
        train_source,
        valid_source,
        {
            "n_train": n_train,
            "n_eval": n_eval,
            "n_test": n_test,
        },
    )


def build_jsonl_train_eval_sources(
    *,
    data_cfg: "DataConfig",
    nn_model: "NNsight",
    tokenizer: "PreTrainedTokenizerBase",
    layer_idx: int,
    model_name: str,
    corpus_dir: Path | None,
    corpora: Sequence[str],
    train_tokens: int,
    valid_tokens: int,
) -> tuple[
    "BalancedMixtureSource",
    "BalancedMixtureSource",
    dict[str, str | list[str] | int],
]:
    from vasae.data.corpus_windows import (
        BalancedMixtureSource,
        corpus_jsonl,
        default_corpus_dir,
    )

    corpus_dir = corpus_dir or default_corpus_dir()
    corpora = tuple(corpora)
    train_paths = {
        corpus: corpus_jsonl(corpus_dir, corpus, "train") for corpus in corpora
    }
    heldout_paths = {
        corpus: corpus_jsonl(corpus_dir, corpus, "heldout") for corpus in corpora
    }

    logger.info(
        "Training JSONL mixture with %s total tokens, model=%s, layer=%s, paths=%s",
        f"{train_tokens:,}",
        model_name,
        layer_idx,
        {key: str(value) for key, value in train_paths.items()},
    )
    train_source = BalancedMixtureSource(
        model=nn_model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        corpus_paths=train_paths,
        total_token_budget=train_tokens,
        batch_size=data_cfg.train_batchsize,
        max_length=data_cfg.max_length,
    )
    valid_source = BalancedMixtureSource(
        model=nn_model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        corpus_paths=heldout_paths,
        total_token_budget=valid_tokens,
        batch_size=data_cfg.valid_batchsize,
        max_length=data_cfg.max_length,
    )
    return (
        train_source,
        valid_source,
        {
            "corpus_dir": str(corpus_dir),
            "corpora": list(corpora),
            "train_tokens": train_tokens,
            "valid_tokens": valid_tokens,
        },
    )


def build_train_metrics(lm_head):
    from vasae.metrics.base import MetricComposer
    from vasae.metrics.logitlens import LogitLens, LogitLensAccMetric
    from vasae.metrics.variance_explained import VarianceExplained

    return MetricComposer(
        [
            LogitLensAccMetric(LogitLens(lm_head)),
            VarianceExplained(),
        ]
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    logger.info(f"use device: {device}")

    save_dir: Path = Path(args.save_dir) / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output will be saved in: {save_dir}")

    # --- Lazy imports so --help don't need to wait so long.
    import transformers  # transformers ~8s
    import wandb  # wandb ~2s

    from vasae.data.schema import DataConfig
    from vasae.engine.trainer import Trainer
    from vasae.models.sae import SAEConfig, SAEModel

    # --- Load LLM (model-agnostic) ---
    llm_ctx = load_online_llm(
        args.model_name,
        device=device,
        dtype_name=args.dtype,
        layer_idx=args.layer_idx,
    )

    # --- Data config ---
    data_cfg = DataConfig(
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        text_column=args.text_column,
        max_length=args.max_length,
        train_batchsize=args.train_batchsize,
        valid_batchsize=args.valid_batchsize,
        test_batchsize=args.test_batchsize,
    )

    if args.data_source == "hf":
        train_source, valid_source, data_summary = build_hf_train_eval_sources(
            data_cfg=data_cfg,
            cache_root=args.save_dir,
            nn_model=llm_ctx.nn_model,
            tokenizer=llm_ctx.tokenizer,
            layer_idx=args.layer_idx,
            train_samples=args.train_samples,
            eval_samples=args.eval_samples,
            test_samples=args.test_samples,
        )
    else:
        train_source, valid_source, data_summary = build_jsonl_train_eval_sources(
            data_cfg=data_cfg,
            nn_model=llm_ctx.nn_model,
            tokenizer=llm_ctx.tokenizer,
            layer_idx=args.layer_idx,
            model_name=args.model_name,
            corpus_dir=args.corpus_dir,
            corpora=args.corpora,
            train_tokens=args.train_tokens,
            valid_tokens=args.valid_tokens,
        )
    # --- Build SAE ---
    dim_sparse = infer_dim_sparse(
        args.dim_sparse,
        args.tied_decoder,
        llm_ctx.dim_model,
        llm_ctx.vocab_size,
    )

    sae_cfg = SAEConfig(
        dim_model=llm_ctx.dim_model,
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

    sae_model = attach_sae_embeddings(
        sae_model,
        llm_ctx.embedding,
        freeze_decoder=args.freeze_decoder,
    ).float()

    logger.info(
        f"SAE: dim_sparse={dim_sparse}, tied={args.tied_decoder}, "
        f"sparsity={args.sparsity_type}, k={args.k}"
    )

    # --- Metrics ---
    train_metrics = build_train_metrics(llm_ctx.lm_head)
    eval_metrics = build_train_metrics(llm_ctx.lm_head)

    # --- Trainer ---
    optimizer = optim.Adam(
        [p for p in sae_model.parameters() if p.requires_grad], lr=args.lr
    )
    trainer = Trainer(
        sae_model=sae_model,
        metrics=train_metrics,
        eval_metrics=eval_metrics,
        device=device,
    )

    # --- wandb ---
    if not args.no_wandb:
        wandb.init(
            project="VASAE",
            name=args.exp_name,
            group=args.wandb_group,
            config=jsonable(vars(args)),
        )
    else:
        wandb.init(mode="disabled")

    def load_best_sae(checkpoint_dir: Path) -> SAEModel:
        sae_model = SAEModel.from_pretrained(checkpoint_dir).to(device)
        return attach_sae_embeddings(
            sae_model,
            llm_ctx.embedding,
            freeze_decoder=args.freeze_decoder,
        ).float()

    # --- Fit (with optional early stopping) ---
    fit_out = trainer.fit(
        train_source=train_source,
        eval_source=valid_source,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        max_batches=args.max_batches,
        patience=args.patience,
        save_dir=save_dir,
        log_fn=wandb.log,
        load_best_model_fn=load_best_sae,
    )
    logger.info(f"Model saved to {save_dir}")
    save_training_results(
        save_dir / "results.json",
        config={**vars(args), **data_summary},
        stopped_epoch=fit_out["stopped_epoch"],
        eval_out=fit_out["eval"],
    )

    wandb.finish()


if __name__ == "__main__":
    main()
