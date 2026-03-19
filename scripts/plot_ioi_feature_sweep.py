"""Post-processing for the per-feature IOI causal intervention sweep.

Reads layer_*.json files produced by eval_ioi_feature_sweep.py and generates:
1. Markdown table of top-20 features by mean Recovery
2. Plot 1: max Recovery vs layer (line chart)
3. Plot 2: Recovery heatmap (features × prompts) at best layer

Example:
    uv run python scripts/plot_ioi_feature_sweep.py \
        --input-dir /scratch/b5bq/pu22650.b5bq/VASAE_out/ioi_feature_sweep \
        --output-dir exp/IOI/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vasae.utils.log import get_logger

logger = get_logger("plot_ioi_feature_sweep")


def load_all_layers(input_dir: Path) -> dict[int, dict]:
    """Load layer_*.json files, return {layer_idx: data}."""
    layers = {}
    for p in sorted(input_dir.glob("layer_*.json")):
        with p.open() as f:
            data = json.load(f)
        layers[data["layer_idx"]] = data
    return layers


def flatten_features(data: dict) -> list[dict]:
    """Flatten per-example features into a list with example index."""
    rows = []
    for ex_idx, ex in enumerate(data["examples"]):
        for feat in ex["features"]:
            rows.append({**feat, "example_idx": ex_idx})
    return rows


def top_features_table(all_layers: dict[int, dict], top_k: int = 20) -> str:
    """Build markdown table of top features by mean Recovery across all layers."""
    # Aggregate: (layer, feature_id) -> list of recovery values and other stats
    agg: dict[tuple[int, int], list[dict]] = {}
    for layer_idx, data in all_layers.items():
        for row in flatten_features(data):
            key = (layer_idx, row["feature_id"])
            agg.setdefault(key, []).append(row)

    # Compute mean recovery per (layer, feature)
    summary = []
    for (layer_idx, fid), rows in agg.items():
        recoveries = [r["recovery"] for r in rows]
        mean_rec = np.mean(recoveries)
        mean_strength = np.mean([r["strength"] for r in rows])
        mean_du_clean = np.mean([r["du_clean"] for r in rows])
        mean_du_corr = np.mean([r["du_corr"] for r in rows])
        mean_du_interv = np.mean([r["du_intervened"] for r in rows])
        mean_effect = np.mean([r["effect"] for r in rows])
        mean_specificity = np.mean([r["specificity"] for r in rows])
        summary.append({
            "feature_id": fid,
            "layer": layer_idx,
            "strength": mean_strength,
            "du_clean": mean_du_clean,
            "du_corr": mean_du_corr,
            "du_intervened": mean_du_interv,
            "effect": mean_effect,
            "recovery": mean_rec,
            "specificity": mean_specificity,
        })

    summary.sort(key=lambda x: x["recovery"], reverse=True)
    top = summary[:top_k]

    lines = [
        "| Feature ID | Layer | Feature Strength | $\\Delta u_{\\text{clean}}$ | $\\Delta u_{\\text{corr}}$ | $\\Delta u_{\\text{clean, intervened}}$ | Effect | Recovery | Specificity Score |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in top:
        lines.append(
            f"| {row['feature_id']} | {row['layer']} | {row['strength']:.4f} "
            f"| {row['du_clean']:.4f} | {row['du_corr']:.4f} "
            f"| {row['du_intervened']:.4f} | {row['effect']:.4f} "
            f"| {row['recovery']:.4f} | {row['specificity']:.4f} |"
        )
    return "\n".join(lines)


def plot_max_recovery_vs_layer(all_layers: dict[int, dict], output_path: Path):
    """Line chart: max_{f,p} Recovery(l,f,p) vs layer l."""
    layers_sorted = sorted(all_layers.keys())
    max_recoveries = []
    for l in layers_sorted:
        features = flatten_features(all_layers[l])
        if features:
            max_rec = max(r["recovery"] for r in features)
        else:
            max_rec = 0.0
        max_recoveries.append(max_rec)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers_sorted, max_recoveries, marker="o", linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Max Recovery")
    ax.set_title("Max Recovery by Layer")
    ax.set_xticks(layers_sorted)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved max recovery plot to %s", output_path)


def plot_recovery_heatmap(data: dict, layer_idx: int, output_path: Path, top_n: int = 50):
    """Heatmap: Recovery(f, p) at given layer, top features by mean recovery."""
    features = flatten_features(data)
    if not features:
        logger.warning("No features found for layer %d, skipping heatmap", layer_idx)
        return

    # Get all unique feature IDs and their mean recovery
    from collections import defaultdict
    feat_recoveries: dict[int, list[float]] = defaultdict(list)
    for r in features:
        feat_recoveries[r["feature_id"]].append(r["recovery"])

    feat_mean = {fid: np.mean(recs) for fid, recs in feat_recoveries.items()}
    top_feats = sorted(feat_mean, key=feat_mean.get, reverse=True)[:top_n]
    feat_to_row = {fid: i for i, fid in enumerate(top_feats)}

    n_examples = data["n_prompts"]
    matrix = np.full((len(top_feats), n_examples), np.nan)

    for r in features:
        fid = r["feature_id"]
        if fid in feat_to_row:
            matrix[feat_to_row[fid], r["example_idx"]] = r["recovery"]

    fig, ax = plt.subplots(figsize=(max(10, n_examples * 0.15), max(6, len(top_feats) * 0.2)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_xlabel("Prompt Index")
    ax.set_ylabel("Feature ID")
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats, fontsize=6)
    ax.set_title(f"Recovery Heatmap — Layer {layer_idx} (top {len(top_feats)} features)")
    fig.colorbar(im, ax=ax, label="Recovery")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved heatmap to %s", output_path)


def parse_args():
    p = argparse.ArgumentParser(description="Plot IOI feature sweep results")
    p.add_argument("--input-dir", type=str, required=True, help="Dir with layer_*.json files")
    p.add_argument("--output-dir", type=str, required=True, help="Dir for figures and table")
    p.add_argument("--top-k", type=int, default=20, help="Top features for table")
    p.add_argument("--heatmap-top-n", type=int, default=50, help="Top features for heatmap")
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_layers = load_all_layers(input_dir)
    if not all_layers:
        raise ValueError(f"No layer_*.json files found in {input_dir}")
    logger.info("Loaded %d layers: %s", len(all_layers), sorted(all_layers.keys()))

    # Table
    table_md = top_features_table(all_layers, top_k=args.top_k)
    table_path = output_dir / "top_features_table.md"
    table_path.write_text(table_md)
    logger.info("Table:\n%s", table_md)

    # Plot 1: max recovery vs layer
    plot_max_recovery_vs_layer(all_layers, output_dir / "max_recovery_vs_layer.pdf")

    # Plot 2: heatmap at best layer
    best_layer = max(all_layers.keys(), key=lambda l: max(
        (r["recovery"] for r in flatten_features(all_layers[l])), default=0.0
    ))
    logger.info("Best layer by max recovery: %d", best_layer)
    plot_recovery_heatmap(
        all_layers[best_layer], best_layer,
        output_dir / f"recovery_heatmap_layer{best_layer}.pdf",
        top_n=args.heatmap_top_n,
    )

    logger.info("Done. Figures saved to %s", output_dir)


if __name__ == "__main__":
    main()
