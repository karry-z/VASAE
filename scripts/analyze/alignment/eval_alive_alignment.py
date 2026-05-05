"""Evaluate alive-feature vocab alignment on held-out corpora.

For each layer/corpus pair, this script marks SAE features as alive when they
activate at least once on valid held-out tokens, then reports the fraction of
alive features whose decoder direction has geometric alignment >= threshold.
"""

import argparse
import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch

from vasae.analysis.alignment import compute_geometric_alignment
from vasae.analysis.alive_alignment import compute_alive_alignment_stats
from vasae.analysis.sae_loader import get_decoder_features, load_sae_for_analysis
from vasae.data.corpus_windows import HeldoutCorpusSource, corpus_jsonl
from vasae.models.factory import get_embedding
from vasae.models.online import load_online_llm
from vasae.utils.log import get_logger
from vasae.utils.seed import set_seed

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate alive-feature alignment for held-out corpora"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/projects/b5bq/VASAE/F001_Benchmarking_mix"),
    )
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 3, 6, 9, 11])
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=None,
        help="Optional single layer override for Slurm array tasks.",
    )
    parser.add_argument(
        "--corpora",
        type=str,
        nargs="+",
        default=["fineweb", "dclm", "pile"],
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Optional single corpus override for Slurm array tasks.",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("/projects/b5bq/VASAE/Dataset/data"),
    )
    parser.add_argument("--eval-tokens", type=int, default=1_000_000)
    parser.add_argument("--alignment-threshold", type=float, default=0.8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp/F001_Benchmarking/alive_alignment/gpt2_mix"),
    )
    parser.add_argument("--variant", type=str, default="soft")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip model evaluation and only regenerate CSV/plots from JSON outputs.",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip CSV/plot aggregation after evaluation.",
    )
    return parser.parse_args()


def jsonable(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    return value


def model_tag(model_name: str) -> str:
    return "gpt2" if "gpt2" in model_name.lower() else "llama"


def discover_checkpoint(
    results_dir: Path,
    model_name: str,
    layer_idx: int,
    variant: str,
) -> Path:
    tag = model_tag(model_name)
    preferred = results_dir / f"001Fmix_{tag}_L{layer_idx}_{variant}"
    if preferred.is_dir():
        return preferred

    pattern = re.compile(rf".*{tag}.*_L{layer_idx}_{re.escape(variant)}$")
    matches = sorted(path for path in results_dir.iterdir() if path.is_dir() and pattern.match(path.name))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint found for model={model_name}, layer={layer_idx}, "
            f"variant={variant} in {results_dir}"
        )
    raise ValueError(
        f"Multiple checkpoints found for layer={layer_idx}, variant={variant}: {matches}"
    )


def result_path(output_dir: Path, layer_idx: int, corpus: str) -> Path:
    return output_dir / f"L{layer_idx}" / f"{corpus}.json"


@torch.no_grad()
def collect_alive_mask(
    *,
    sae,
    source: HeldoutCorpusSource,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    n_features = int(sae.config.dim_sparse)
    alive_mask = torch.zeros(n_features, dtype=torch.bool, device=device)
    tokens_processed = 0

    for batch_idx, batch in enumerate(source):
        if batch_idx % 25 == 0:
            logger.info("  Alive pass batch %d, tokens=%d", batch_idx, tokens_processed)

        activations = batch["activations"]
        if isinstance(activations, tuple):
            activations = activations[0]
        activations = activations.detach()
        if activations.dim() == 2:
            valid_activations = activations.float()
            valid_count = activations.shape[0]
        else:
            mask = batch["attention_mask"].bool()
            valid_count = int(mask.sum().item())
            valid_activations = activations[mask].float()

        if valid_count == 0:
            continue

        _, z = sae.encode(valid_activations)
        alive_mask |= (z > 0).any(dim=0)
        tokens_processed += valid_count

        del activations, valid_activations, z, batch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return alive_mask.cpu(), tokens_processed


def evaluate_one(
    *,
    layer_idx: int,
    corpus: str,
    args,
    llm_ctx,
    embedding_weight: torch.Tensor,
    device: torch.device,
) -> dict:
    out_path = result_path(args.output_dir, layer_idx, corpus)
    if out_path.exists() and not args.force:
        logger.info("Skipping existing result: %s", out_path)
        with out_path.open() as handle:
            return json.load(handle)

    checkpoint = discover_checkpoint(
        args.results_dir,
        args.model_name,
        layer_idx,
        args.variant,
    )
    logger.info("=== Layer %d, corpus=%s ===", layer_idx, corpus)
    logger.info("Checkpoint: %s", checkpoint)

    sae = load_sae_for_analysis(checkpoint, device=device)
    n_features = int(sae.config.dim_sparse)

    logger.info("Computing geometric alignment")
    geo = compute_geometric_alignment(
        get_decoder_features(sae),
        embedding_weight,
        top_k=1,
        device=device,
    )
    alignment_scores = geo.max_sims
    aligned_mask = alignment_scores >= args.alignment_threshold

    heldout_path = corpus_jsonl(args.corpus_dir, corpus, "heldout")
    source = HeldoutCorpusSource(
        model=llm_ctx.nn_model,
        tokenizer=llm_ctx.tokenizer,
        layer_idx=layer_idx,
        jsonl_path=heldout_path,
        token_budget=args.eval_tokens,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    logger.info("Collecting alive features from %s", heldout_path)
    alive_mask, tokens_processed = collect_alive_mask(
        sae=sae,
        source=source,
        device=device,
    )
    stats = compute_alive_alignment_stats(
        alive_mask,
        alignment_scores,
        threshold=args.alignment_threshold,
    )

    alive_features = alive_mask.nonzero(as_tuple=True)[0].tolist()
    aligned_features = aligned_mask.nonzero(as_tuple=True)[0].tolist()
    alive_aligned_features = (alive_mask & aligned_mask).nonzero(as_tuple=True)[
        0
    ].tolist()

    result = {
        "layer": layer_idx,
        "layer_idx": layer_idx,
        "corpus": corpus,
        "variant": args.variant,
        "checkpoint": str(checkpoint),
        "n_features": n_features,
        "n_alive": stats.n_alive,
        "n_aligned": stats.n_aligned,
        "n_alive_aligned": stats.n_alive_aligned,
        "alive_alignment_rate": stats.alive_alignment_rate,
        "dead_rate": stats.dead_rate,
        "alignment_threshold": args.alignment_threshold,
        "eval_tokens_requested": args.eval_tokens,
        "tokens_processed": tokens_processed,
        "alive_features": alive_features,
        "aligned_features": aligned_features,
        "alive_aligned_features": alive_aligned_features,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(jsonable(result), handle, indent=2)

    logger.info(
        "Saved %s: alive=%d aligned=%d alive+aligned=%d rate=%.2f%%",
        out_path,
        stats.n_alive,
        stats.n_aligned,
        stats.n_alive_aligned,
        stats.alive_alignment_rate * 100,
    )

    del sae, geo
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def load_task_results(output_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(output_dir.glob("L*/*.json")):
        with path.open() as handle:
            row = json.load(handle)
        if {"layer", "corpus", "alive_alignment_rate"}.issubset(row):
            rows.append(row)
    return rows


def write_csv(rows: list[dict], output_dir: Path) -> Path:
    csv_path = output_dir / "alive_alignment_per_layer.csv"
    fieldnames = [
        "layer",
        "corpus",
        "variant",
        "n_features",
        "n_alive",
        "n_aligned",
        "n_alive_aligned",
        "alive_alignment_rate",
        "alive_alignment_rate_pct",
        "dead_rate",
        "alignment_threshold",
        "eval_tokens_requested",
        "tokens_processed",
        "checkpoint",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["corpus"], item["layer"])):
            out = {key: row.get(key, "") for key in fieldnames}
            out["alive_alignment_rate_pct"] = row["alive_alignment_rate"] * 100
            writer.writerow(out)
    return csv_path


def plot_rates(rows: list[dict], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    corpora = sorted({row["corpus"] for row in rows})
    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    for corpus in corpora:
        corpus_rows = sorted(
            [row for row in rows if row["corpus"] == corpus],
            key=lambda item: item["layer"],
        )
        xs = [row["layer"] for row in corpus_rows]
        ys = [row["alive_alignment_rate"] * 100 for row in corpus_rows]
        ax.plot(xs, ys, marker="o", linewidth=2, label=corpus)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Alive feature alignment rate (%)")
    ax.set_title("GPT-2 Mix: Alive Feature Alignment")
    ax.set_xticks(sorted({row["layer"] for row in rows}))
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Held-out corpus")
    fig.tight_layout()

    fig.savefig(output_dir / "alive_alignment_rate_by_layer.pdf", bbox_inches="tight")
    fig.savefig(
        output_dir / "alive_alignment_rate_by_layer.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def aggregate_outputs(output_dir: Path) -> None:
    rows = load_task_results(output_dir)
    if not rows:
        logger.warning("No task JSON files found under %s", output_dir)
        return
    csv_path = write_csv(rows, output_dir)
    plot_rates(rows, output_dir)
    logger.info("Wrote %s", csv_path)
    logger.info("Wrote alive_alignment_rate_by_layer.{png,pdf}")


def main():
    args = parse_args()
    set_seed(args.seed)
    get_logger()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        aggregate_outputs(args.output_dir)
        return

    layers = [args.layer_idx] if args.layer_idx is not None else args.layers
    corpora = [args.corpus] if args.corpus is not None else args.corpora
    if any(corpus is None for corpus in corpora):
        raise ValueError("corpus cannot be None")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("Loading model once for layers=%s corpora=%s", layers, corpora)

    llm_ctx = load_online_llm(
        args.model_name,
        device=str(device),
        dtype_name=args.dtype,
        layer_idx=layers[0],
    )
    embedding_weight = get_embedding(llm_ctx.llm).weight.data

    for layer_idx in layers:
        for corpus in corpora:
            evaluate_one(
                layer_idx=layer_idx,
                corpus=corpus,
                args=args,
                llm_ctx=llm_ctx,
                embedding_weight=embedding_weight,
                device=device,
            )

    if not args.no_aggregate:
        aggregate_outputs(args.output_dir)


if __name__ == "__main__":
    main()
