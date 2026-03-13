"""
Plot Variance Explained and Loss Recovered vs Layer for the decomposition sweep.

Reads results.json and loss_recovered.json from exp/sweep_decompose/layer_*/dpca_*/.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(base_dir: Path, layers, d_pca_values):
    """Load results.json and loss_recovered.json for all layer × d_pca combinations."""
    ve_data = {}       # d_pca -> {layer: ve}
    ve_sparse_data = {}
    lr_data = {}       # d_pca -> {layer: loss_recovered}
    lr_sparse_data = {}

    for d_pca in d_pca_values:
        ve_data[d_pca] = {}
        ve_sparse_data[d_pca] = {}
        lr_data[d_pca] = {}
        lr_sparse_data[d_pca] = {}

        for layer in layers:
            result_path = base_dir / f"layer_{layer}" / f"dpca_{d_pca}" / "results.json"
            lr_path = base_dir / f"layer_{layer}" / f"dpca_{d_pca}" / "loss_recovered.json"

            if result_path.exists():
                with open(result_path) as f:
                    r = json.load(f)
                ve_data[d_pca][layer] = r.get("ve")
                ve_sparse_data[d_pca][layer] = r.get("ve_sparse")

            if lr_path.exists():
                with open(lr_path) as f:
                    r = json.load(f)
                lr_data[d_pca][layer] = r.get("loss_recovered")
                lr_sparse_data[d_pca][layer] = r.get("loss_recovered_sparse")

    return ve_data, ve_sparse_data, lr_data, lr_sparse_data


def plot_metric(data, sparse_data, layers, d_pca_values, ylabel, title, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    for d_pca in d_pca_values:
        x = [l for l in layers if l in data[d_pca]]
        y = [data[d_pca][l] for l in x]
        if x:
            ax.plot(x, y, marker="o", label=f"d_pca={d_pca}")

    # sparse-only baseline (use first d_pca that has data)
    for d_pca in d_pca_values:
        x = [l for l in layers if l in sparse_data[d_pca]]
        y = [sparse_data[d_pca][l] for l in x]
        if x:
            ax.plot(x, y, marker="x", linestyle="--", color="gray", label="sparse-only")
            break

    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(layers)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default="/scratch/b5bq/pu22650.b5bq/VASAE_out/sweep_decompose")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir

    layers = list(range(12))
    d_pca_values = [16, 32, 64, 128, 256, 512]

    ve_data, ve_sparse_data, lr_data, lr_sparse_data = load_results(
        base_dir, layers, d_pca_values
    )

    plot_metric(
        ve_data, ve_sparse_data, layers, d_pca_values,
        ylabel="Variance Explained",
        title="Variance Explained vs Layer (Sparse + PCA Decomposition)",
        output_path=output_dir / "variance_explained.png",
    )

    plot_metric(
        lr_data, lr_sparse_data, layers, d_pca_values,
        ylabel="Loss Recovered",
        title="Loss Recovered vs Layer (Sparse + PCA Decomposition)",
        output_path=output_dir / "loss_recovered.png",
    )


if __name__ == "__main__":
    main()
