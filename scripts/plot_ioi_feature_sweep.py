"""Plot IOI feature sweep results from vocab-aligned SAE.

Reads per-layer JSON files and produces:
  - Figure 1: Max/Mean CR vs layer + causal feature count
  - Figure 2: CR heatmap for a selected layer (top features x prompts)
  - Figure 3: Alignment (max_sim) vs causal effect (mean CR) scatter
  - Figure 4: Inhibitory features (negative CR) fraction
  - Table: Top causal features with aligned token info

Usage:
    uv run python scripts/plot_ioi_feature_sweep.py \
        --input-dir /scratch/b5bq/pu22650.b5bq/VASAE_out/021_ioi_feature_sweep \
        --output-dir exp/021_IOI/figures
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "figure.dpi": 150,
})


def load_all_layers(input_dir: Path, n_layers: int = 12):
    data = {}
    for layer in range(n_layers):
        p = input_dir / f"layer_{layer}.json"
        if p.exists():
            with p.open() as f:
                data[layer] = json.load(f)
    return data


def aggregate_features(data: dict) -> dict:
    """Per-layer: aggregate per-feature stats across prompts."""
    layer_stats = {}
    for layer, d in sorted(data.items()):
        by_feat = defaultdict(lambda: {
            "crs": [], "kls": [], "specs": [], "strengths": [],
            "sim": 0.0, "tok": "", "tok_id": -1, "n_prompts": 0,
        })
        for ex in d["examples"]:
            for feat in ex["features"]:
                fid = feat["feature_id"]
                info = by_feat[fid]
                info["crs"].append(feat["recovery"])
                info["kls"].append(feat["kl_divergence"])
                info["specs"].append(feat["specificity"])
                info["strengths"].append(feat["strength"])
                info["sim"] = feat.get("max_sim", 0.0)
                info["tok"] = feat.get("aligned_token", "?")
                info["tok_id"] = feat.get("aligned_token_id", -1)
                info["n_prompts"] += 1
        summary = {}
        for fid, info in by_feat.items():
            summary[fid] = {
                "mean_cr": np.mean(info["crs"]),
                "std_cr": np.std(info["crs"]),
                "mean_kl": np.mean(info["kls"]),
                "mean_spec": np.mean(info["specs"]),
                "mean_strength": np.mean(info["strengths"]),
                "max_sim": info["sim"],
                "aligned_token": info["tok"],
                "aligned_token_id": info["tok_id"],
                "n_prompts": info["n_prompts"],
            }
        layer_stats[layer] = summary
    return layer_stats


def plot_cr_vs_layer(layer_stats: dict, output_dir: Path):
    """Figure 1: Max CR and causal feature count vs layer."""
    layers = sorted(layer_stats.keys())
    max_crs = []
    n_causal_03 = []
    n_causal_05 = []

    for layer in layers:
        feats = layer_stats[layer]
        crs = [f["mean_cr"] for f in feats.values()]
        max_crs.append(max(crs) if crs else 0)
        n_causal_03.append(sum(1 for c in crs if c >= 0.3))
        n_causal_05.append(sum(1 for c in crs if c >= 0.5))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(layers, max_crs, "o-", color="C0", label="Max CR")
    ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="CR = 1")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Max mean CR")
    ax1.set_title("Max Corruption Recovery vs Layer")
    ax1.legend()
    ax1.set_xticks(layers)

    ax2.bar([l - 0.15 for l in layers], n_causal_03, width=0.3, color="C1",
            label=r"CR $\geq$ 0.3", alpha=0.8)
    ax2.bar([l + 0.15 for l in layers], n_causal_05, width=0.3, color="C3",
            label=r"CR $\geq$ 0.5", alpha=0.8)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Number of features")
    ax2.set_title("Causal Feature Count vs Layer")
    ax2.legend()
    ax2.set_xticks(layers)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig1_cr_vs_layer.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_cr_vs_layer")


def plot_heatmap(data: dict, layer_stats: dict, layer: int, output_dir: Path,
                 n_features: int = 50):
    """Figure 2: CR heatmap for top features at a given layer."""
    d = data[layer]
    feats = layer_stats[layer]

    sorted_feats = sorted(feats.items(), key=lambda x: -x[1]["mean_cr"])[:n_features]
    feat_ids = [fid for fid, _ in sorted_feats]
    feat_id_set = set(feat_ids)

    n_prompts = len(d["examples"])
    matrix = np.full((len(feat_ids), n_prompts), np.nan)
    feat_idx_map = {fid: i for i, fid in enumerate(feat_ids)}

    for pi, ex in enumerate(d["examples"]):
        for feat in ex["features"]:
            fid = feat["feature_id"]
            if fid in feat_idx_map:
                matrix[feat_idx_map[fid], pi] = feat["recovery"]

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=2,
                   interpolation="nearest")

    ylabels = []
    for fid in feat_ids:
        tok = feats[fid]["aligned_token"]
        sim = feats[fid]["max_sim"]
        ylabels.append(f"F{fid} ({tok.strip()}, s={sim:.2f})")

    ax.set_yticks(range(len(feat_ids)))
    ax.set_yticklabels(ylabels, fontsize=6)
    ax.set_xlabel("Prompt index")
    ax.set_ylabel("Feature (sorted by mean CR)")
    ax.set_title(f"Corruption Recovery Heatmap — L{layer} (top {n_features} features)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Corruption Recovery")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig2_heatmap_L{layer}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved fig2_heatmap_L{layer}")


def plot_sim_vs_cr(layer_stats: dict, output_dir: Path):
    """Figure 3: Geometric alignment (max_sim) vs causal effect scatter."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    selected_layers = [0, 2, 5, 7, 9, 11]

    for ax, layer in zip(axes.flat, selected_layers):
        feats = layer_stats.get(layer, {})
        sims = [f["max_sim"] for f in feats.values()]
        crs = [f["mean_cr"] for f in feats.values()]

        ax.scatter(sims, crs, s=8, alpha=0.5, c="C0", edgecolors="none")
        ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
        ax.axhline(0.3, color="red", linestyle="--", alpha=0.3, label="CR=0.3")
        ax.axvline(0.8, color="green", linestyle="--", alpha=0.3, label="sim=0.8")
        ax.set_xlabel(r"max cos sim $s(i)$")
        ax.set_ylabel("mean CR")
        ax.set_title(f"Layer {layer}")
        ax.set_xlim(-0.05, 1.05)

    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Geometric Alignment vs Causal Effect", fontsize=13)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig3_sim_vs_cr.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_sim_vs_cr")


def plot_negative_cr(layer_stats: dict, output_dir: Path):
    """Figure 4: Fraction of features with negative CR vs layer."""
    layers = sorted(layer_stats.keys())
    neg_frac = []
    neg_strong_frac = []

    for layer in layers:
        feats = layer_stats[layer]
        total = len(feats)
        neg = sum(1 for f in feats.values() if f["mean_cr"] < 0)
        neg_strong = sum(1 for f in feats.values() if f["mean_cr"] < -0.1)
        neg_frac.append(neg / total * 100 if total > 0 else 0)
        neg_strong_frac.append(neg_strong / total * 100 if total > 0 else 0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([l - 0.15 for l in layers], neg_frac, width=0.3, color="C0",
           label="CR < 0", alpha=0.7)
    ax.bar([l + 0.15 for l in layers], neg_strong_frac, width=0.3, color="C3",
           label=r"CR < $-$0.1", alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction of active features (%)")
    ax.set_title("Inhibitory Features (negative CR)")
    ax.legend()
    ax.set_xticks(layers)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig4_negative_cr.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig4_negative_cr")


def write_top_features_table(layer_stats: dict, output_dir: Path, top_n: int = 20):
    """Write markdown table of top causal features across all layers."""
    all_feats = []
    for layer, feats in layer_stats.items():
        for fid, info in feats.items():
            all_feats.append((layer, fid, info))

    all_feats.sort(key=lambda x: -x[2]["mean_cr"])

    lines = [
        "| Layer | Feature | Aligned Token | $s(i)$ | Mean CR | Std CR | Specificity | Active Prompts |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for layer, fid, info in all_feats[:top_n]:
        tok = info["aligned_token"].strip()
        lines.append(
            f"| L{layer} | {fid} | {tok!r} | {info['max_sim']:.2f} | "
            f"{info['mean_cr']:.2f} | {info['std_cr']:.2f} | "
            f"{info['mean_spec']:.1f} | {info['n_prompts']}/73 |"
        )

    table = "\n".join(lines)
    (output_dir / "top_features_table.md").write_text(table)
    print("Saved top_features_table.md")
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--heatmap-layer", type=int, default=1)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_all_layers(input_dir, args.n_layers)
    print(f"Loaded {len(data)} layers")

    layer_stats = aggregate_features(data)

    plot_cr_vs_layer(layer_stats, output_dir)
    plot_heatmap(data, layer_stats, args.heatmap_layer, output_dir)
    plot_sim_vs_cr(layer_stats, output_dir)
    plot_negative_cr(layer_stats, output_dir)
    table = write_top_features_table(layer_stats, output_dir)
    print("\nTop features:\n")
    print(table)


if __name__ == "__main__":
    main()
