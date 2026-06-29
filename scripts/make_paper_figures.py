"""Generate final paper-facing figures from cleaned summaries under experiments/paper."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path


METHOD_LABELS = {
    "plain": "plain TopK SAE",
    "soft": "VASAE-soft",
    "vasae_soft": "VASAE-soft",
    "hard": "hard_tied_baseline",
    "hard_tied_baseline": "hard_tied_baseline",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read cleaned paper summaries and regenerate final reconstruction/alignment figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--paper-dir", default="experiments/paper", help="Root directory with cleaned paper summaries.")
    parser.add_argument("--output-dir", default=None, help="Figure output directory; defaults to <paper-dir>/figures.")
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated output formats.")
    parser.add_argument("--dpi", type=int, default=200, help="Raster figure DPI.")
    return parser


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s][%(asctime)s][%(name)s] %(message)s",
        datefmt="%Y%m%d %H:%M:%S",
    )
    return logging.getLogger("make_paper_figures")


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if value is None or value == "":
                continue
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return rows


def format_list(value: str) -> list[str]:
    return [item.strip().lstrip(".") for item in value.split(",") if item.strip()]


def save_figure(fig, output_dir: Path, stem: str, formats: list[str], dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", dpi=dpi, bbox_inches="tight")


def plot_reconstruction(rows: list[dict], output_dir: Path, formats: list[str], dpi: int, logger: logging.Logger) -> None:
    if not rows:
        logger.warning("No reconstruction per-layer CSV found; skipping reconstruction figures.")
        return

    import matplotlib.pyplot as plt

    metrics = [
        ("variance_explained", "Variance explained", "reconstruction_variance_explained"),
        ("loss_recovered", "CE recovery", "reconstruction_ce_recovery"),
        ("logitlens_acc", "Logit-lens agreement", "reconstruction_logitlens_agreement"),
    ]
    models = sorted({row["model"] for row in rows})
    variants = ["plain", "soft", "hard", "vasae_soft", "hard_tied_baseline"]
    colors = {
        "plain": "#2b6cb0",
        "soft": "#2f855a",
        "vasae_soft": "#2f855a",
        "hard": "#744210",
        "hard_tied_baseline": "#744210",
    }

    for metric, ylabel, stem in metrics:
        fig, axes = plt.subplots(1, len(models), figsize=(5 * max(len(models), 1), 3.2), squeeze=False)
        for axis, model in zip(axes[0], models):
            model_rows = [row for row in rows if row["model"] == model and metric in row]
            for variant in variants:
                series = sorted(
                    [row for row in model_rows if row.get("variant") == variant],
                    key=lambda row: row.get("layer", 0),
                )
                if not series:
                    continue
                axis.plot(
                    [row["layer"] for row in series],
                    [row[metric] for row in series],
                    marker="o",
                    linewidth=1.8,
                    markersize=3.5,
                    label=METHOD_LABELS.get(variant, variant),
                    color=colors.get(variant),
                )
            axis.set_title(str(model))
            axis.set_xlabel("Layer")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.25)
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 3), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        save_figure(fig, output_dir, stem, formats, dpi)
        plt.close(fig)
        logger.info("Wrote %s", stem)


def read_alignment_rows(paper_dir: Path) -> list[dict]:
    rows = []
    root = paper_dir / "alignment" / "results"
    if not root.exists():
        return rows
    for result_path in sorted(root.glob("*/*/results.json")):
        with result_path.open() as handle:
            data = json.load(handle)
        layer_name = result_path.parent.name
        model_name = result_path.parent.parent.name
        layer = int(layer_name[1:]) if layer_name.startswith("L") and layer_name[1:].isdigit() else None
        geometric = data.get("geometric", {})
        summary = data.get("summary", {})
        rows.append(
            {
                "model": model_name,
                "layer": layer,
                "aligned_pct": geometric.get("aligned_pct"),
                "coverage_pct": geometric.get("coverage_pct"),
                "max_sim_mean": geometric.get("max_sim_mean", summary.get("max_cosine_mean")),
                "max_sim_median": geometric.get("max_sim_median"),
            }
        )
    return rows


def plot_alignment(rows: list[dict], output_dir: Path, formats: list[str], dpi: int, logger: logging.Logger) -> None:
    if not rows:
        logger.warning("No alignment result JSON files found; skipping alignment figures.")
        return

    import matplotlib.pyplot as plt

    metrics = [
        ("aligned_pct", "Aligned features (%)", "alignment_aligned_pct"),
        ("coverage_pct", "Vocabulary coverage (%)", "alignment_vocab_coverage"),
        ("max_sim_mean", "Mean max cosine", "alignment_mean_max_cosine"),
    ]
    models = sorted({row["model"] for row in rows})
    colors = ["#2b6cb0", "#2f855a", "#744210", "#805ad5"]
    for metric, ylabel, stem in metrics:
        fig, axis = plt.subplots(figsize=(6.4, 3.6))
        for idx, model in enumerate(models):
            series = sorted(
                [row for row in rows if row["model"] == model and row.get(metric) is not None],
                key=lambda row: row.get("layer") if row.get("layer") is not None else -1,
            )
            if not series:
                continue
            axis.plot(
                [row["layer"] for row in series],
                [row[metric] for row in series],
                marker="o",
                linewidth=1.8,
                markersize=3.5,
                label=model,
                color=colors[idx % len(colors)],
            )
        axis.set_xlabel("Layer")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
        fig.tight_layout()
        save_figure(fig, output_dir, stem, formats, dpi)
        plt.close(fig)
        logger.info("Wrote %s", stem)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = setup_logger()

    import matplotlib

    matplotlib.use("Agg")

    paper_dir = Path(args.paper_dir)
    output_dir = Path(args.output_dir) if args.output_dir else paper_dir / "figures"
    formats = format_list(args.formats)
    reconstruction_rows = read_csv_rows(paper_dir / "reconstruction" / "results_per_layer.csv")
    alignment_rows = read_alignment_rows(paper_dir)
    plot_reconstruction(reconstruction_rows, output_dir, formats, args.dpi, logger)
    plot_alignment(alignment_rows, output_dir, formats, args.dpi, logger)
    logger.info("Paper figures written to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
