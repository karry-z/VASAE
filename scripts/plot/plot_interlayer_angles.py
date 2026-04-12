"""Compare the angle distribution of post-residual-stream vectors
across consecutive layers of GPT-2.

For each pair of adjacent layers (layer i, layer i+1), we compute the
angle between the output vectors at the same sequence position, then
plot the distribution of those angles.

Uses pre-collected memmap activations from collect_gpt2_activations.py.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_layer_activations(data_dir: Path):
    """Load all layer activations from memmap files.

    Returns:
        dict[str, np.memmap] – layer_name -> (num_examples, seq_len, hidden_dim)
    """
    meta_path = data_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    activations = {}
    for layer_name, info in meta.items():
        mm = np.memmap(
            data_dir / info["path"],
            mode="r",
            dtype=info["dtype"],
            shape=tuple(info["shape"]),
        )
        activations[layer_name] = mm

    return activations


def compute_angles(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute the angle (in degrees) between vectors along the last dim.

    Args:
        v1, v2: shape (..., d)

    Returns:
        angles in degrees, shape (...)
    """
    v1 = torch.from_numpy(v1)
    v2 = torch.from_numpy(v2)
    cos_sim = torch.nn.functional.cosine_similarity(v1, v2, dim=-1)
    cos_sim = cos_sim.clamp(-1.0, 1.0)
    return torch.acos(cos_sim).rad2deg().numpy()


def plot_angle_distributions(angles_per_pair, save_path):
    """Plot histograms of angle distributions for consecutive layer pairs."""
    num_pairs = len(angles_per_pair)
    ncols = 3
    nrows = (num_pairs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax, (label, angles) in zip(axes, angles_per_pair.items()):
        ax.hist(angles, bins=80, density=True, alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Angle (degrees)")
        ax.set_ylabel("Density")
        mean_val = angles.mean()
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.0,
                    label=f"mean={mean_val:.1f}")
        ax.legend(fontsize=8)

    for idx in range(len(angles_per_pair), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Inter-layer angle distribution (post residual stream)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Figure saved to {save_path}")


def plot_summary(angles_per_pair, save_path):
    """Box plot summarising all layer pairs on one figure."""
    labels = list(angles_per_pair.keys())
    data = [angles_per_pair[l] for l in labels]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#7faadb")
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Angle (degrees)")
    ax.set_title("Inter-layer angle distribution (post residual stream)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Summary figure saved to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot inter-layer angle distributions for GPT-2"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=r"/scratch/b5bq/pu22650.b5bq/activations_gpt2_Geralt-Targaryen_openwebtext2",
        help="Directory containing pre-collected memmap activations",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of samples to use (0 = all)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="outputs/interlayer_angles",
        help="Directory to save figures",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load pre-collected activations
    print(f"Loading activations from {data_dir} ...")
    activations = load_layer_activations(data_dir)

    # Sort layer names by index
    layer_names = sorted(
        activations.keys(), key=lambda n: int(n.split(".")[-1])
    )
    print(f"Layers: {layer_names}")

    n_samples = activations[layer_names[0]].shape[0]
    if args.num_samples > 0:
        n_samples = min(args.num_samples, n_samples)
    print(f"Using {n_samples} samples")

    # Compute angles between consecutive layers
    angles_per_pair = {}
    for i in range(len(layer_names) - 1):
        cur_layer = layer_names[i]
        nxt_layer = layer_names[i + 1]
        cur_idx = int(cur_layer.split(".")[-1])
        nxt_idx = int(nxt_layer.split(".")[-1])

        v_cur = np.array(activations[cur_layer][:n_samples])
        v_nxt = np.array(activations[nxt_layer][:n_samples])

        angles = compute_angles(v_cur, v_nxt)  # (N, seq_len)
        label = f"Layer {cur_idx} -> {nxt_idx}"
        angles_per_pair[label] = angles.flatten()
        print(
            f"  {label}: mean={angles_per_pair[label].mean():.2f}, "
            f"std={angles_per_pair[label].std():.2f}"
        )

    # Plot
    plot_angle_distributions(
        angles_per_pair, save_dir / "interlayer_angle_hist.png"
    )
    plot_summary(angles_per_pair, save_dir / "interlayer_angle_boxplot.png")


if __name__ == "__main__":
    main()
