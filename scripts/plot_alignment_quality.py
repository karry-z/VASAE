"""Generate figures for 002_F Alignment Quality Analysis.

Reads per-layer JSON + PT results and produces:
  - Figure 1: s(i) distribution histogram (VASAE-Soft vs plain SAE, 3 representative layers)
  - Figure 2: Feature category distribution by layer (stacked bar)
  - Case study table (printed to stdout as markdown)

Usage:
    uv run python scripts/plot_alignment_quality.py \
        --input-dir exp/002_F_AlignmentAnalysis/gpt2 \
        --model-label GPT-2 \
        --output-dir exp/002_F_AlignmentAnalysis/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from shared_utils.log import get_logger

log = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Plot 002_F alignment quality figures")
    p.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory with per-layer results (L0/, L1/, ...)",
    )
    p.add_argument(
        "--model-label", type=str, default="GPT-2", help="Model label for figure titles"
    )
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args()


def load_layer_results(input_dir: Path):
    """Load all per-layer JSON results."""
    results = {}
    for d in sorted(input_dir.iterdir()):
        if d.is_dir() and d.name.startswith("L"):
            rpath = d / "results.json"
            if rpath.exists():
                with open(rpath) as f:
                    r = json.load(f)
                    results[r["layer_idx"]] = r
    return results


def load_max_sims(input_dir: Path, layer_idx: int):
    """Load full max_sims arrays from PT file."""
    pt_path = input_dir / f"L{layer_idx}" / "max_sims.pt"
    if pt_path.exists():
        return torch.load(pt_path, weights_only=True)
    return None


def select_representative_layers(layers: list[int], n: int = 3):
    """Select shallow, middle, deep representative layers."""
    if len(layers) <= n:
        return layers
    indices = [0, len(layers) // 2, len(layers) - 1]
    return [layers[i] for i in indices]


def plot_geometric_histograms(
    results: dict, input_dir: Path, model_label: str, output_dir: Path
):
    """Figure 1: s(i) distribution histograms for representative layers."""
    layers = sorted(results.keys())
    rep_layers = select_representative_layers(layers)

    fig, axes = plt.subplots(1, len(rep_layers), figsize=(4.5 * len(rep_layers), 4))
    if len(rep_layers) == 1:
        axes = [axes]

    bins = np.linspace(0, 1, 51)

    for ax, layer_idx in zip(axes, rep_layers):
        pt_data = load_max_sims(input_dir, layer_idx)
        if pt_data is None:
            ax.set_title(f"L{layer_idx} (no data)")
            continue

        sims = pt_data["max_sims"].numpy()
        ax.hist(
            sims,
            bins=bins,
            alpha=0.7,
            color="#4C72B0",
            label="VASAE-Soft",
            edgecolor="none",
        )

        if "baseline_max_sims" in pt_data:
            bl_sims = pt_data["baseline_max_sims"].numpy()
            ax.hist(
                bl_sims,
                bins=bins,
                alpha=0.5,
                color="#AAAAAA",
                label="Plain SAE",
                edgecolor="none",
            )

        ax.set_xlabel("$s(i) = \\max_v \\cos(d_i, e_v)$")
        ax.set_ylabel("Feature count")
        ax.set_title(f"L{layer_idx}")
        ax.axvline(x=0.8, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax.legend(fontsize=8)

    fig.suptitle(f"{model_label}: Geometric Alignment Distribution", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_geometric_distribution.pdf", bbox_inches="tight")
    fig.savefig(
        output_dir / "fig1_geometric_distribution.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    log.info("Saved fig1_geometric_distribution")


def plot_category_distribution(results: dict, model_label: str, output_dir: Path):
    """Figure 2: Feature category stacked bar by layer."""
    layers = sorted(results.keys())

    cat_keys = ["dual", "input_detector", "output_controller", "non_functional"]
    cat_labels = ["Dual", "Input Detector", "Output Controller", "Non-functional"]
    cat_colors = ["#4C72B0", "#55A868", "#C44E52", "#CCB974"]

    data = {k: [] for k in cat_keys}
    for l in layers:
        cats = results[l]["categories"]
        # Use sum of categories as denominator — this handles both old results
        # (where n_categorized was not saved) and new results correctly.
        n_tested = results[l].get("n_categorized", 0)
        if n_tested == 0:
            n_tested = sum(cats.get(k, 0) for k in cat_keys)
        for k in cat_keys:
            count = cats.get(k, 0)
            data[k].append(count / max(n_tested, 1) * 100)

    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.7), 5))

    bottoms = np.zeros(len(layers))
    for k, label, color in zip(cat_keys, cat_labels, cat_colors):
        vals = np.array(data[k])
        ax.bar(
            range(len(layers)),
            vals,
            bottom=bottoms,
            label=label,
            color=color,
            alpha=0.85,
        )
        bottoms += vals

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
    ax.set_ylabel("Proportion of Aligned Features (%)")
    ax.set_title(f"{model_label}: Feature Functional Category by Layer")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(output_dir / "fig2_category_distribution.pdf", bbox_inches="tight")
    fig.savefig(
        output_dir / "fig2_category_distribution.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    log.info("Saved fig2_category_distribution")


def plot_functional_rates(results: dict, model_label: str, output_dir: Path):
    """Figure 3: Output control rate and alive+aligned count by layer."""
    layers = sorted(results.keys())
    top1 = [results[l]["output_control"]["top1_match_rate"] for l in layers]
    top5 = [results[l]["output_control"]["top5_match_rate"] for l in layers]
    alive_aligned = [results[l]["n_alive_aligned"] for l in layers]

    fig, ax1 = plt.subplots(figsize=(max(7, len(layers) * 0.6), 4.5))

    # Left axis: rates
    ax1.plot(
        range(len(layers)),
        top1,
        "o-",
        color="#C44E52",
        label="Top-1 Match",
        markersize=5,
    )
    ax1.plot(
        range(len(layers)),
        top5,
        "s--",
        color="#C44E52",
        alpha=0.6,
        label="Top-5 Match",
        markersize=4,
    )
    ax1.set_ylabel("Output Control Match Rate (%)", color="#C44E52")
    ax1.tick_params(axis="y", labelcolor="#C44E52")
    ax1.set_ylim(0, max(top5) * 1.3)

    # Right axis: alive+aligned count
    ax2 = ax1.twinx()
    ax2.bar(
        range(len(layers)),
        alive_aligned,
        color="#4C72B0",
        alpha=0.25,
        label="Alive+Aligned",
    )
    ax2.set_ylabel("Alive+Aligned Feature Count", color="#4C72B0")
    ax2.tick_params(axis="y", labelcolor="#4C72B0")

    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
    ax1.set_xlabel("Layer")
    ax1.set_title(f"{model_label}: Output Control Rate & Feature Count by Layer")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_dir / "fig3_functional_rates.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig3_functional_rates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved fig3_functional_rates")


def print_summary_table(results: dict, model_label: str):
    """Print cross-layer summary table."""
    layers = sorted(results.keys())
    log.info("\n=== %s Cross-Layer Summary ===", model_label)
    log.info(
        "%-5s %8s %8s %10s %10s %10s",
        "Layer",
        "Align%",
        "Alive%",
        "InputDet%",
        "OutTop1%",
        "OutTop5%",
    )
    for l in layers:
        r = results[l]
        log.info(
            "L%-4d %7.1f%% %7.1f%% %9.1f%% %9.1f%% %9.1f%%",
            l,
            r["geometric"]["aligned_pct"],
            r["n_alive"] / r["n_features"] * 100,
            r["input_detection"]["detection_rate"],
            r["output_control"]["top1_match_rate"],
            r["output_control"]["top5_match_rate"],
        )


def print_case_studies(results: dict, model_label: str):
    """Print case study examples as markdown."""
    log.info("\n### %s Case Studies\n", model_label)
    log.info(
        "| Layer | Feature | Category | Token | s(i) | P_i | Top-1 Match | Ablation Top-5 |"
    )
    log.info(
        "| ----- | ------- | -------- | ----- | ---- | --- | ----------- | -------------- |"
    )

    for layer_idx in sorted(results.keys()):
        examples = results[layer_idx].get("examples", [])
        for ex in examples:
            tok = f"`{ex['aligned_token']}`"
            ablation = ", ".join(f"`{t}`" for t in ex.get("ablation_top5", []))
            log.info(
                "| L%d | %d | %s | %s | %.2f | %.2f | %s | %s |",
                ex["layer"],
                ex["feature_id"],
                ex["category"],
                tok,
                ex["geo_max_sim"],
                ex["P_i"],
                "Y" if ex["top1_match"] else "N",
                ablation,
            )


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_layer_results(input_dir)
    if not results:
        log.error("No results found in %s", input_dir)
        return

    log.info("Loaded results for %d layers: %s", len(results), sorted(results.keys()))

    plot_geometric_histograms(results, input_dir, args.model_label, output_dir)
    plot_category_distribution(results, args.model_label, output_dir)
    plot_functional_rates(results, args.model_label, output_dir)
    print_summary_table(results, args.model_label)
    print_case_studies(results, args.model_label)

    log.info("\nAll figures saved to %s", output_dir)


if __name__ == "__main__":
    main()
    main()
