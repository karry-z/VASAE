"""Aggregate and visualize results from experiments 012a/b/c.

Produces:
  1. margin_by_layer.png — margin distributions with null baselines
  2. hub_distribution.png — t_geo hub count distribution
  3. mean_act_alignment.png — cos(d_i, mu_i) and t_geo==t_mean by layer
  4. context_position_match.png — t_geo match rate at each context position
  5. feature_category_by_layer.png — stacked bar of feature categories (012c)
  6. consistency_heatmap.png — geo/input/causal consistency across 12 layers

Usage:
    python scripts/plot/plot_tgeo_meaning.py \
        --weight-dir exp/012_p_TgeoMeaning/weight_only \
        --data-dir exp/012_p_TgeoMeaning/data \
        --io-dir exp/012_p_TgeoMeaning/io_full \
        --output-dir exp/012_p_TgeoMeaning/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description="012d: Aggregate + visualize t_geo analysis")
    p.add_argument("--weight-dir", type=str, required=True,
                   help="012a output dir (weight_only)")
    p.add_argument("--data-dir", type=str, required=True,
                   help="012b output dir (data/L{layer}_k32)")
    p.add_argument("--io-dir", type=str, required=True,
                   help="012c output dir (io_full/L{layer}_k32)")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--layers", type=str, default="0-11")
    p.add_argument("--data-layers", type=str, default="2,6,11",
                   help="Layers with data-dependent analysis (012b)")
    return p.parse_args()


def parse_layer_range(s: str) -> list[int]:
    if "," in s:
        return [int(x) for x in s.split(",")]
    parts = s.split("-")
    return list(range(int(parts[0]), int(parts[1]) + 1))


def load_json_safe(path):
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None


def plot_margin_by_layer(weight_dir, layers, output_dir):
    """Fig 1: Margin distributions by layer with null baselines."""
    tensors = torch.load(weight_dir / "tensors.pt", map_location="cpu", weights_only=True)

    fig, ax = plt.subplots(figsize=(14, 6))

    positions = []
    labels = []
    data_real = []
    data_rot = []
    data_rand = []

    for i, layer in enumerate(layers):
        key = f"L{layer}"
        if key not in tensors:
            continue
        t = tensors[key]
        data_real.append(t["margins"].numpy())
        data_rot.append(t["null_rot_margins"].flatten().numpy())
        data_rand.append(t["null_rand_margins"].flatten().numpy())
        positions.append(i)
        labels.append(f"L{layer}")

    if not positions:
        print("  No margin data found, skipping margin plot")
        return

    width = 0.25
    bp1 = ax.boxplot(data_real, positions=[p - width for p in positions],
                     widths=width * 0.8, patch_artist=True, showfliers=False)
    bp2 = ax.boxplot(data_rot, positions=positions,
                     widths=width * 0.8, patch_artist=True, showfliers=False)
    bp3 = ax.boxplot(data_rand, positions=[p + width for p in positions],
                     widths=width * 0.8, patch_artist=True, showfliers=False)

    for bp, color in [(bp1, "#2196F3"), (bp2, "#FF9800"), (bp3, "#9E9E9E")]:
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Alignment Margin (sim_1st - sim_2nd)")
    ax.set_title("t_geo Alignment Margin by Layer (Real vs Null Baselines)")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]],
              ["Real", "Null (rotated)", "Null (random)"], loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "margin_by_layer.png", dpi=150)
    plt.close(fig)
    print("  Saved margin_by_layer.png")


def plot_hub_distribution(weight_dir, layers, output_dir):
    """Fig 2: Hub count distribution."""
    hub_stats = load_json_safe(weight_dir / "hub_stats.json")
    if not hub_stats:
        print("  No hub stats found, skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: coverage by layer
    ax = axes[0]
    layer_labels = []
    coverages = []
    for layer in layers:
        key = str(layer)
        if key in hub_stats:
            layer_labels.append(f"L{layer}")
            coverages.append(hub_stats[key]["coverage_pct"])
    ax.bar(layer_labels, coverages, color="#4CAF50", alpha=0.7)
    ax.set_ylabel("Vocab Coverage (%)")
    ax.set_title("t_geo Vocab Coverage by Layer")
    ax.grid(axis="y", alpha=0.3)

    # Right: embedding norm comparison (t_geo tokens vs all tokens)
    ax = axes[1]
    tgeo_norms = []
    all_norms = []
    for layer in layers:
        key = str(layer)
        if key in hub_stats:
            tgeo_norms.append(hub_stats[key]["tgeo_mean_emb_norm"])
            all_norms.append(hub_stats[key]["all_mean_emb_norm"])

    if layer_labels:
        x = np.arange(len(layer_labels))
        w = 0.35
        ax.bar(x - w/2, tgeo_norms, w, label="t_geo tokens", color="#2196F3", alpha=0.7)
        ax.bar(x + w/2, all_norms, w, label="All tokens", color="#9E9E9E", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels)
        ax.set_ylabel("Mean Embedding Norm")
        ax.set_title("Embedding Norm: t_geo Tokens vs All")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "hub_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved hub_distribution.png")


def plot_mean_act_alignment(data_dir, data_layers, output_dir):
    """Fig 3: cos(d_i, mu_i) and t_geo==t_mean rate by layer."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = []
    cos_means = []
    cos_medians = []
    tmean_pcts = []

    for layer in data_layers:
        result_path = Path(data_dir) / f"L{layer}_k32" / "tgeo_data_analysis.json"
        data = load_json_safe(result_path)
        if data is None:
            continue
        a3 = data["analysis_3_mean_direction"]
        labels.append(f"L{layer}")
        cos_means.append(a3["cos_d_mu_mean"])
        cos_medians.append(a3["cos_d_mu_median"])
        tmean_pcts.append(a3["tgeo_eq_tmean_pct"])

    if not labels:
        print("  No data analysis results found, skipping mean_act_alignment plot")
        return

    # Left: cos(d_i, mu_i)
    ax = axes[0]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, cos_means, w, label="Mean", color="#2196F3", alpha=0.7)
    ax.bar(x + w/2, cos_medians, w, label="Median", color="#FF9800", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("cos(d_i, mu_i)")
    ax.set_title("Decoder-MeanAct Cosine Similarity")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Right: t_geo == t_mean rate
    ax = axes[1]
    ax.bar(labels, tmean_pcts, color="#4CAF50", alpha=0.7)
    ax.set_ylabel("Match Rate (%)")
    ax.set_title("t_geo == t_mean Rate")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "mean_act_alignment.png", dpi=150)
    plt.close(fig)
    print("  Saved mean_act_alignment.png")


def plot_context_position_match(data_dir, data_layers, output_dir):
    """Fig 4: t_geo match rate at each context position."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    for idx, layer in enumerate(data_layers):
        result_path = Path(data_dir) / f"L{layer}_k32" / "tgeo_data_analysis.json"
        data = load_json_safe(result_path)
        if data is None:
            continue

        a4 = data["analysis_4_context_position"]
        offsets = a4["window_offsets"]
        rates = [a4["position_match_rates_pct"][f"offset_{o}"] for o in offsets]
        rand_rates = [a4["random_baseline_match_rates_pct"][f"offset_{o}"] for o in offsets]

        color = colors[idx % len(colors)]
        ax.plot(offsets, rates, "o-", label=f"L{layer} (real)", color=color)
        ax.plot(offsets, rand_rates, "x--", label=f"L{layer} (random)", color=color, alpha=0.5)

    ax.set_xlabel("Position Offset (0=current, +1=next)")
    ax.set_ylabel("t_geo Match Rate (%)")
    ax.set_title("t_geo == token at Context Position")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(y=1/50257*100, color="gray", linestyle=":", label="chance (1/vocab)")

    fig.tight_layout()
    fig.savefig(output_dir / "context_position_match.png", dpi=150)
    plt.close(fig)
    print("  Saved context_position_match.png")


def plot_feature_categories(io_dir, layers, output_dir):
    """Fig 5: Stacked bar chart of feature categories by layer."""
    categories = ["token_feature", "output_feature", "input_feature",
                   "context_feature", "unaligned", "dead"]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#9E9E9E", "#E0E0E0"]

    layer_labels = []
    cat_data = {cat: [] for cat in categories}

    for layer in layers:
        result_path = Path(io_dir) / f"L{layer}_k32" / "io_decomposition_results.json"
        data = load_json_safe(result_path)
        if data is None:
            continue
        layer_labels.append(f"L{layer}")
        cats = data.get("categories", {})
        n_total = data["config"]["n_features"]
        for cat in categories:
            count = cats.get(cat, 0)
            cat_data[cat].append(count / n_total * 100)

    if not layer_labels:
        print("  No IO decomposition results found, skipping category plot")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(layer_labels))
    bottom = np.zeros(len(layer_labels))

    for cat, color in zip(categories, colors):
        vals = np.array(cat_data[cat])
        ax.bar(x, vals, bottom=bottom, label=cat, color=color, alpha=0.8)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.set_ylabel("Feature Proportion (%)")
    ax.set_title("Feature Categories by Layer (IO Decomposition)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "feature_category_by_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved feature_category_by_layer.png")


def plot_consistency_heatmap(io_dir, layers, output_dir):
    """Fig 6: Heatmap of geo/input/causal consistency across layers."""
    metrics = ["geo_eq_logit_top1_pct", "geo_eq_input_top1_pct",
               "geo_eq_causal_top1_pct", "input_eq_causal_top1_pct"]
    metric_labels = ["geo=logit", "geo=input", "geo=causal", "input=causal"]

    layer_labels = []
    data_matrix = []

    for layer in layers:
        result_path = Path(io_dir) / f"L{layer}_k32" / "io_decomposition_results.json"
        data = load_json_safe(result_path)
        if data is None:
            continue
        layer_labels.append(f"L{layer}")
        cons = data.get("consistency", {})
        row = [cons.get(m, 0) for m in metrics]
        data_matrix.append(row)

    if not layer_labels:
        print("  No consistency data found, skipping heatmap")
        return

    matrix = np.array(data_matrix)

    fig, ax = plt.subplots(figsize=(8, max(6, len(layer_labels) * 0.5 + 1)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(layer_labels)))
    ax.set_yticklabels(layer_labels)

    # Annotate cells
    for i in range(len(layer_labels)):
        for j in range(len(metric_labels)):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    color=color, fontsize=9)

    ax.set_title("View Consistency (%) by Layer")
    fig.colorbar(im, ax=ax, label="%")

    fig.tight_layout()
    fig.savefig(output_dir / "consistency_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved consistency_heatmap.png")


def generate_report(weight_dir, data_dir, io_dir, layers, data_layers, output_dir):
    """Generate a text summary of hypothesis verdicts."""
    margin_stats = load_json_safe(Path(weight_dir) / "margin_stats.json") or {}
    hub_stats = load_json_safe(Path(weight_dir) / "hub_stats.json") or {}

    lines = ["# Experiment 012: t_geo Meaning Diagnosis — Summary Report\n"]

    # H3: artifact?
    lines.append("## H3: t_geo is embedding density artifact")
    for layer in layers:
        key = str(layer)
        if key in margin_stats:
            ms = margin_stats[key]
            real = ms["median_margin"]
            null_rot = ms["null_rotated_median_margin"]
            null_rand = ms["null_random_median_margin"]
            lines.append(f"  L{layer}: median_margin={real:.4f} "
                         f"(null_rot={null_rot:.4f}, null_rand={null_rand:.4f})")

    # H4: geometry bias?
    lines.append("\n## H4: t_geo tokens have biased embedding geometry")
    for layer in layers:
        key = str(layer)
        if key in hub_stats:
            hs = hub_stats[key]
            lines.append(f"  L{layer}: tgeo_norm={hs['tgeo_mean_emb_norm']:.3f} "
                         f"(all={hs['all_mean_emb_norm']:.3f}), "
                         f"tgeo_knn={hs['tgeo_mean_knn_sim']:.3f} "
                         f"(all={hs['all_mean_knn_sim']:.3f}), "
                         f"coverage={hs['coverage_pct']:.1f}%")

    # H1 + H2 from data analysis
    lines.append("\n## H1: t_geo = mean activation direction")
    lines.append("## H2: t_geo = next-token")
    for layer in data_layers:
        result_path = Path(data_dir) / f"L{layer}_k32" / "tgeo_data_analysis.json"
        data = load_json_safe(result_path)
        if data:
            a3 = data["analysis_3_mean_direction"]
            a4 = data["analysis_4_context_position"]
            lines.append(f"  L{layer}:")
            lines.append(f"    H1: t_geo==t_mean={a3['tgeo_eq_tmean_pct']:.1f}%, "
                         f"cos(d,mu)={a3['cos_d_mu_mean']:.4f}")
            next_key = "offset_1"
            next_rate = a4["position_match_rates_pct"].get(next_key, 0)
            next_rand = a4["random_baseline_match_rates_pct"].get(next_key, 0)
            lines.append(f"    H2: t_geo==t_next={next_rate:.3f}% "
                         f"(random={next_rand:.3f}%)")

    # H5 from IO decomposition
    lines.append("\n## H5: Layer dependence (feature categories)")
    for layer in layers:
        result_path = Path(io_dir) / f"L{layer}_k32" / "io_decomposition_results.json"
        data = load_json_safe(result_path)
        if data:
            cats = data.get("categories", {})
            cons = data.get("consistency", {})
            n = data["config"]["n_features"]
            alive = data.get("feature_stats", {}).get("alive", "?")
            lines.append(f"  L{layer}: alive={alive}, "
                         f"token_feat={cats.get('token_feature', 0)}, "
                         f"context_feat={cats.get('context_feature', 0)}, "
                         f"geo=causal={cons.get('geo_eq_causal_top1_pct', 0):.1f}%")

    report_text = "\n".join(lines)
    report_path = output_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"  Saved report.md")
    print(report_text)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = parse_layer_range(args.layers)
    data_layers = parse_layer_range(args.data_layers)
    weight_dir = Path(args.weight_dir)
    data_dir = Path(args.data_dir)
    io_dir = Path(args.io_dir)

    print("=== Generating plots ===\n")

    print("1. Margin by layer...")
    plot_margin_by_layer(weight_dir, layers, output_dir)

    print("2. Hub distribution...")
    plot_hub_distribution(weight_dir, layers, output_dir)

    print("3. Mean activation alignment...")
    plot_mean_act_alignment(data_dir, data_layers, output_dir)

    print("4. Context position match...")
    plot_context_position_match(data_dir, data_layers, output_dir)

    print("5. Feature categories by layer...")
    plot_feature_categories(io_dir, layers, output_dir)

    print("6. Consistency heatmap...")
    plot_consistency_heatmap(io_dir, layers, output_dir)

    print("\n=== Generating report ===\n")
    generate_report(weight_dir, data_dir, io_dir, layers, data_layers, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
