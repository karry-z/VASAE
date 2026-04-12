"""
Plot final results for the 010_SoftAlignSweep experiment.

Sweep: 12 layers × 3 k values (8, 16, 32) × 3 anchor coeffs (1e-3, 1e-4, 1e-5) = 108 runs.
Reads results.json from each run directory.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LAYERS = list(range(12))
KS = [8, 16, 32]
ANCHORS = [1e-3, 1e-4, 1e-5]
ANCHOR_LABELS = ["1e-3", "1e-4", "1e-5"]

# Metrics to plot (key in results.json -> display name)
TEST_METRICS = {
    "loss_recovered": "Loss Recovered",
    "variance_explained": "Variance Explained",
    "logitlens_acc": "Logit Lens Accuracy",
    "loss_reconst": "Reconstruction Loss (MSE)",
    "ce_sae": "CE (SAE)",
    "ce_id": "CE (Identity)",
    "ce_zero": "CE (Zero)",
}


def load_all_results(base_dir: Path):
    """Load results.json for all 108 runs. Returns dict[layer][k][anchor] -> test metrics."""
    data = {}
    for layer in LAYERS:
        data[layer] = {}
        for k in KS:
            data[layer][k] = {}
            for anchor in ANCHORS:
                anchor_tag = ANCHOR_LABELS[ANCHORS.index(anchor)]
                exp_name = f"010_soft_gpt2_L{layer}_k{k}_a{anchor_tag}"
                result_path = base_dir / exp_name / "results.json"
                if result_path.exists():
                    with open(result_path) as f:
                        r = json.load(f)
                    data[layer][k][anchor] = r.get("test", {})
                else:
                    print(f"Missing: {exp_name}")
                    data[layer][k][anchor] = None
    return data


def plot_metric_by_layer(data, metric_key, metric_name, output_dir: Path):
    """One figure: x=layer, lines for each (k, anchor) combo.
    Subplots: one per k value, lines colored by anchor coeff."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle(f"{metric_name} vs Layer", fontsize=14, fontweight="bold")

    colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    markers = ["o", "s", "^"]

    for i, k in enumerate(KS):
        ax = axes[i]
        for j, anchor in enumerate(ANCHORS):
            xs, ys = [], []
            for layer in LAYERS:
                d = data[layer][k][anchor]
                if d and metric_key in d:
                    xs.append(layer)
                    ys.append(d[metric_key])
            ax.plot(xs, ys, color=colors[j], marker=markers[j], markersize=5,
                    linewidth=1.5, label=f"anchor={ANCHOR_LABELS[j]}")
        ax.set_xlabel("Layer")
        ax.set_title(f"k = {k}")
        ax.set_xticks(LAYERS)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel(metric_name)
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / f"sweep_{metric_key}_by_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(data, metric_key, metric_name, output_dir: Path):
    """Heatmap: rows = layer, cols = (k, anchor) combos."""
    n_combos = len(KS) * len(ANCHORS)
    matrix = np.full((len(LAYERS), n_combos), np.nan)
    col_labels = []

    for ci, (ki, k) in enumerate(enumerate(KS)):
        for aj, anchor in enumerate(ANCHORS):
            col_idx = ci * len(ANCHORS) + aj
            col_labels.append(f"k={k}\na={ANCHOR_LABELS[aj]}")
            for li, layer in enumerate(LAYERS):
                d = data[layer][k][anchor]
                if d and metric_key in d:
                    matrix[li, col_idx] = d[metric_key]

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(n_combos))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(LAYERS)))
    ax.set_yticklabels([f"L{l}" for l in LAYERS])
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Layer")
    ax.set_title(f"{metric_name} — Heatmap", fontsize=13, fontweight="bold")

    # Annotate cells
    for li in range(len(LAYERS)):
        for ci in range(n_combos):
            val = matrix[li, ci]
            if not np.isnan(val):
                ax.text(ci, li, f"{val:.3f}", ha="center", va="center", fontsize=6,
                        color="black" if 0.3 < (val - np.nanmin(matrix)) / (np.nanmax(matrix) - np.nanmin(matrix) + 1e-9) < 0.7 else "white")

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / f"sweep_{metric_key}_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ce_comparison(data, output_dir: Path):
    """Plot CE_sae vs CE_id and CE_zero to show how much SAE degrades LM performance."""
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), sharex=True)
    fig.suptitle("Cross-Entropy: Identity vs SAE vs Zero — by Layer", fontsize=14, fontweight="bold")

    colors_k = {"8": "#e41a1c", "16": "#377eb8", "32": "#4daf4a"}
    linestyles = {1e-3: "-", 1e-4: "--", 1e-5: ":"}

    for li, layer in enumerate(LAYERS):
        row, col = li // 4, li % 4
        ax = axes[row, col]

        # Plot CE_id and CE_zero as horizontal reference (same for all configs at this layer)
        ce_id_vals = []
        ce_zero_vals = []
        for k in KS:
            for anchor in ANCHORS:
                d = data[layer][k][anchor]
                if d:
                    if "ce_id" in d:
                        ce_id_vals.append(d["ce_id"])
                    if "ce_zero" in d:
                        ce_zero_vals.append(d["ce_zero"])

        if ce_id_vals:
            ax.axhline(np.mean(ce_id_vals), color="gray", linestyle="-", linewidth=1, alpha=0.7, label="CE_id" if li == 0 else None)
        if ce_zero_vals:
            ax.axhline(np.mean(ce_zero_vals), color="gray", linestyle="--", linewidth=1, alpha=0.5, label="CE_zero" if li == 0 else None)

        # Plot CE_sae for each config
        for k in KS:
            for anchor in ANCHORS:
                d = data[layer][k][anchor]
                if d and "ce_sae" in d:
                    ax.scatter(k, d["ce_sae"], color=colors_k[str(k)],
                               marker="o" if anchor == 1e-3 else ("s" if anchor == 1e-4 else "^"),
                               s=40, zorder=5)

        ax.set_title(f"Layer {layer}", fontsize=10)
        if row == 2:
            ax.set_xlabel("k")
        if col == 0:
            ax.set_ylabel("Cross-Entropy")
        ax.set_xticks(KS)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "sweep_ce_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_table(data, output_dir: Path):
    """Best config per layer based on loss_recovered, printed and saved."""
    lines = ["=" * 80]
    lines.append(f"{'Layer':<6} {'Best k':<8} {'Best anchor':<12} {'Loss Rec.':<12} {'VE':<10} {'LL Acc':<10} {'CE_sae':<10}")
    lines.append("-" * 80)

    for layer in LAYERS:
        best_lr, best_k, best_a = -1, None, None
        for k in KS:
            for anchor in ANCHORS:
                d = data[layer][k][anchor]
                if d and "loss_recovered" in d and d["loss_recovered"] > best_lr:
                    best_lr = d["loss_recovered"]
                    best_k = k
                    best_a = anchor
        if best_k is not None:
            d = data[layer][best_k][best_a]
            lines.append(f"L{layer:<5} k={best_k:<6} a={best_a:<11} {best_lr:<12.4f} {d.get('variance_explained', 0):<10.4f} {d.get('logitlens_acc', 0):<10.4f} {d.get('ce_sae', 0):<10.4f}")

    lines.append("=" * 80)
    text = "\n".join(lines)
    print(text)
    with open(output_dir / "best_configs.txt", "w") as f:
        f.write(text + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot 010_SoftAlignSweep results")
    parser.add_argument("--base-dir", type=str,
                        default="/scratch/b5bq/pu22650.b5bq/VASAE_out/010_soft_align",
                        help="Directory containing run outputs")
    parser.add_argument("--output-dir", type=str,
                        default="exp/010_p_SoftAlignSweep/figures",
                        help="Directory to save figures")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    data = load_all_results(base_dir)

    print("Plotting line charts...")
    for metric_key, metric_name in TEST_METRICS.items():
        plot_metric_by_layer(data, metric_key, metric_name, output_dir)

    print("Plotting heatmaps...")
    for metric_key in ["loss_recovered", "variance_explained", "logitlens_acc"]:
        plot_heatmaps(data, metric_key, TEST_METRICS[metric_key], output_dir)

    print("Plotting CE comparison...")
    plot_ce_comparison(data, output_dir)

    print("\nBest config per layer (by loss_recovered on test set):")
    plot_summary_table(data, output_dir)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
