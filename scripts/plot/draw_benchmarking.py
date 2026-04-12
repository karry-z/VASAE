import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.scale import FuncScale


def set_neurips_style():
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "legend.title_fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.3,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


LABEL_MAP = {
    "hard": "VASAE-Hard",
    "plain": "Plain SAE",
    "soft": "VASAE-Soft",
}

STYLE_MAP = {
    "plain": dict(color="#ff7f0e", linestyle="--", marker="s", markersize=5.5, zorder=2),
    "soft": dict(color="#2ca02c", linestyle="-", marker="^", markersize=6.5, zorder=3),
    "hard": dict(color="#1f77b4", linestyle=":", marker="o", markersize=4.5, zorder=1),
}

FALLBACK_STYLES = [
    dict(color="#1f77b4", linestyle="-", marker="o"),
    dict(color="#ff7f0e", linestyle="--", marker="s"),
    dict(color="#2ca02c", linestyle=":", marker="^"),
    dict(color="#d62728", linestyle="-.", marker="D"),
    dict(color="#9467bd", linestyle=(0, (3, 1, 1, 1)), marker="v"),
]


NEG_SCALE = 0.35


def custom_forward(y):
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)

    mask_mid = (y >= 0) & (y <= 1)
    mask_hi = y > 1
    mask_lo = y < 0

    out[mask_mid] = y[mask_mid]
    out[mask_hi] = 1.0 + np.log10(y[mask_hi])
    out[mask_lo] = -NEG_SCALE * np.log10(1.0 - y[mask_lo])

    return out


def custom_inverse(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)

    mask_mid = (z >= 0) & (z <= 1)
    mask_hi = z > 1
    mask_lo = z < 0

    out[mask_mid] = z[mask_mid]
    out[mask_hi] = 10 ** (z[mask_hi] - 1.0)
    out[mask_lo] = 1.0 - 10 ** (-z[mask_lo] / NEG_SCALE)

    return out


def apply_adaptive_yaxis(ax, values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        ax.set_ylim(0, 1)
        return

    vmin = values.min()
    vmax = values.max()

    # 全在 [0, 1] 内：线性坐标并自动缩放
    if vmin >= 0 and vmax <= 1:
        span = max(vmax - vmin, 1e-6)
        pad = 0.05 * span
        lo = max(0.0, vmin - pad)
        hi = min(1.0, vmax + pad)

        if hi - lo < 0.1:
            center = 0.5 * (lo + hi)
            lo = max(0.0, center - 0.05)
            hi = min(1.0, center + 0.05)

        ax.set_yscale("linear")
        ax.set_ylim(lo, hi)

        ticks = np.linspace(lo, hi, 5)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:.2f}" for t in ticks])
        return

    # We plot metrics using a piecewise nonlinear y-axis: values in [0,1] are shown linearly, while values outside this range are logarithmically compressed. This improves readability without distorting the main region of interest.

    ax.set_yscale(FuncScale(ax, (custom_forward, custom_inverse)))

    ticks = []
    if vmin < 0:
        neg_ticks = [-100, -30, -10, -3, -1]
        ticks += [t for t in neg_ticks if vmin <= t < 0]

    ticks += [0.0, 0.5, 1.0]

    if vmax > 1:
        pos_ticks = [2, 3, 5, 10, 20, 50, 100]
        ticks += [t for t in pos_ticks if 1 < t <= vmax]

    ticks = sorted(set(ticks))
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])


def plot_one_metric(df_model, metric_col, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    all_values = []
    # Draw in order: hard (bottom), plain (middle), soft (top) so soft is visible on overlap
    draw_order = ["hard", "plain", "soft"]
    grouped = dict(list(df_model.groupby("variant")))

    for i, variant in enumerate(draw_order):
        if variant not in grouped:
            continue
        subdf = grouped[variant].sort_values("layer")
        style = STYLE_MAP.get(variant, FALLBACK_STYLES[i % len(FALLBACK_STYLES)])
        label = LABEL_MAP.get(variant, variant)

        x = subdf["layer"].to_numpy()
        y = subdf[metric_col].to_numpy(dtype=float)
        all_values.append(y)

        ax.plot(
            x,
            y,
            label=label,
            markeredgewidth=0.8,
            **style,
        )

    all_values = np.concatenate(all_values) if all_values else np.array([0.0, 1.0])

    apply_adaptive_yaxis(ax, all_values)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", axis="both", alpha=0.25)
    ax.legend(title="Variant")

    # x 轴整数刻度，层数多时间隔显示
    layers = np.sort(df_model["layer"].unique())
    if len(layers) > 16:
        step = max(1, len(layers) // 8)
        tick_layers = layers[::step]
        if layers[-1] not in tick_layers:
            tick_layers = np.append(tick_layers, layers[-1])
        ax.set_xticks(tick_layers)
    else:
        ax.set_xticks(layers)

    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    args = parser.parse_args()

    set_neurips_style()

    df = pd.read_csv(args.csv)
    df_model = df[df["model"] == args.model].copy()

    if df_model.empty:
        raise ValueError(f"没有找到 model={args.model} 的数据")

    os.makedirs(args.output, exist_ok=True)

    metrics = [
        ("variance_explained", "VE", "variance_explained"),
        ("loss_recovered", "CE Recovered", "ce_recovered"),
        ("logitlens_acc", "LogitLens Acc", "logitlens_acc"),
    ]

    for col, ylabel, name in metrics:
        out_path = os.path.join(args.output, f"{args.model}_{name}.pdf")
        plot_one_metric(df_model, col, ylabel, out_path)
        print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
