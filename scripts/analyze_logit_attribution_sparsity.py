"""Analyze logit attribution sparsity of SAE decoder features across layers.

For each SAE feature (decoder column d_i), computes the logit attribution
vector la_i = W_U @ d_i and measures how sparse/peaked that distribution is.
Also computes geometric alignment (max cosine sim to W_E) for correlation.

Usage:
    python scripts/analyze_logit_attribution_sparsity.py \
        --base-dir /scratch/b5bq/pu22650.b5bq/VASAE_out/010_soft_align \
        --model-name gpt2 \
        --output-dir exp/011_p_IODecomposition/logit_attr
"""

import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from vasae.models.factory import get_embedding, get_lm_head, load_model
from vasae.models.sae import SAEModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze logit attribution sparsity of SAE decoder features"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/scratch/b5bq/pu22650.b5bq/VASAE_out/010_soft_align",
        help="Base directory containing sweep model directories",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        help="HuggingFace model name for loading W_U and W_E",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exp/011_p_IODecomposition/logit_attr",
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices (default: 0-11)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=32,
        help="k value in sweep naming",
    )
    parser.add_argument(
        "--alpha",
        type=str,
        default="1e-4",
        help="Alpha value in sweep naming",
    )
    return parser.parse_args()


@torch.no_grad()
def compute_logit_attribution_stats(
    decoder_weight: torch.Tensor,
    W_U: torch.Tensor,
    device: torch.device,
    batch_size: int = 1024,
) -> dict:
    """Compute logit attribution sparsity metrics for all decoder features.

    Args:
        decoder_weight: (dim_input, dim_sparse) - decoder weight matrix
        W_U: (vocab_size, dim_input) - unembedding weight matrix
        device: computation device
        batch_size: process features in batches

    Returns:
        dict with per-feature entropy, max/mean ratio, top1/top5 concentration
    """
    # decoder columns: transpose to (n_features, dim_input)
    features = decoder_weight.T.to(device)  # (n_features, dim_input)
    W_U_dev = W_U.to(device)  # (vocab_size, dim_input)
    n_features = features.size(0)

    all_entropy = []
    all_max_mean_ratio = []
    all_top1_conc = []
    all_top5_conc = []
    all_max_logit = []
    all_max_token_id = []

    for start in range(0, n_features, batch_size):
        end = min(start + batch_size, n_features)
        batch_d = features[start:end]  # (batch, dim_input)

        # Logit attribution: la_i = W_U @ d_i^T -> (batch, vocab_size)
        la = batch_d @ W_U_dev.T  # (batch, vocab_size)

        # Softmax probabilities for entropy
        probs = F.softmax(la, dim=1)  # (batch, vocab_size)
        log_probs = F.log_softmax(la, dim=1)
        entropy = -(probs * log_probs).sum(dim=1)  # (batch,)

        # Max / mean ratio (on absolute logits)
        la_abs = la.abs()
        max_val, max_idx = la_abs.max(dim=1)
        mean_val = la_abs.mean(dim=1)
        max_mean_ratio = max_val / (mean_val + 1e-10)

        # Top-1 and top-5 concentration (on softmax probs)
        top5_vals, _ = probs.topk(5, dim=1)
        top1_conc = top5_vals[:, 0]  # top-1 probability
        top5_conc = top5_vals.sum(dim=1)  # sum of top-5 probabilities

        # Track the actual max logit value and token
        real_max_val, real_max_idx = la.max(dim=1)

        all_entropy.append(entropy.cpu())
        all_max_mean_ratio.append(max_mean_ratio.cpu())
        all_top1_conc.append(top1_conc.cpu())
        all_top5_conc.append(top5_conc.cpu())
        all_max_logit.append(real_max_val.cpu())
        all_max_token_id.append(real_max_idx.cpu())

    return {
        "entropy": torch.cat(all_entropy),
        "max_mean_ratio": torch.cat(all_max_mean_ratio),
        "top1_concentration": torch.cat(all_top1_conc),
        "top5_concentration": torch.cat(all_top5_conc),
        "max_logit": torch.cat(all_max_logit),
        "max_token_id": torch.cat(all_max_token_id),
    }


@torch.no_grad()
def compute_geo_alignment(
    decoder_weight: torch.Tensor,
    W_E: torch.Tensor,
    device: torch.device,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Compute max cosine similarity of each decoder feature to W_E rows.

    Args:
        decoder_weight: (dim_input, dim_sparse)
        W_E: (vocab_size, dim_input)
        device: computation device

    Returns:
        (n_features,) tensor of max cosine similarities
    """
    features = decoder_weight.T.to(device)  # (n_features, dim_input)
    W_E_dev = W_E.to(device)

    features_norm = F.normalize(features, dim=1)
    W_E_norm = F.normalize(W_E_dev, dim=1)

    n_features = features_norm.size(0)
    max_sims = []

    for start in range(0, n_features, batch_size):
        end = min(start + batch_size, n_features)
        batch = features_norm[start:end]
        sim = batch @ W_E_norm.T  # (batch, vocab_size)
        max_sims.append(sim.max(dim=1)[0].cpu())

    return torch.cat(max_sims)


def summarize_tensor(t: torch.Tensor) -> dict:
    """Compute summary statistics for a 1D tensor."""
    return {
        "mean": t.mean().item(),
        "std": t.std().item(),
        "median": t.median().item(),
        "min": t.min().item(),
        "max": t.max().item(),
        "p5": t.quantile(0.05).item(),
        "p25": t.quantile(0.25).item(),
        "p75": t.quantile(0.75).item(),
        "p95": t.quantile(0.95).item(),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parse layers
    if args.layers is not None:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = list(range(12))

    # Load GPT-2 for W_U and W_E
    print(f"Loading {args.model_name} for W_U and W_E...")
    lm_model, tokenizer = load_model(args.model_name, device=str(device))
    W_U = get_lm_head(lm_model).weight.data.clone()  # (vocab_size, dim_input)
    W_E = get_embedding(lm_model).weight.data.clone()  # (vocab_size, dim_input)
    print(f"  W_U shape: {W_U.shape}")
    print(f"  W_E shape: {W_E.shape}")

    # Free LM memory
    del lm_model
    torch.cuda.empty_cache()

    # Collect per-layer results
    all_layer_stats = {}
    all_layer_tensors = {}

    for layer in layers:
        run_name = f"010_soft_{args.model_name}_L{layer}_k{args.k}_a{args.alpha}"
        sae_path = Path(args.base_dir) / run_name

        if not sae_path.exists():
            print(f"[SKIP] {run_name}: path not found at {sae_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Layer {layer}: loading {run_name}")

        sae = SAEModel.from_pretrained(sae_path).to(device).eval()
        decoder_weight = sae.decoder.weight.data  # (dim_input, dim_sparse)
        print(f"  Decoder shape: {decoder_weight.shape}")

        # Logit attribution sparsity
        print(f"  Computing logit attribution stats...")
        la_stats = compute_logit_attribution_stats(decoder_weight, W_U, device)

        # Geo alignment
        print(f"  Computing geo alignment (max cos sim to W_E)...")
        geo_align = compute_geo_alignment(decoder_weight, W_E, device)

        # Summarize
        layer_summary = {
            "entropy": summarize_tensor(la_stats["entropy"]),
            "max_mean_ratio": summarize_tensor(la_stats["max_mean_ratio"]),
            "top1_concentration": summarize_tensor(la_stats["top1_concentration"]),
            "top5_concentration": summarize_tensor(la_stats["top5_concentration"]),
            "max_logit": summarize_tensor(la_stats["max_logit"]),
            "geo_alignment": summarize_tensor(geo_align),
            "n_features": decoder_weight.shape[1],
        }

        # Correlation between entropy and alignment
        corr = torch.corrcoef(
            torch.stack([la_stats["entropy"], geo_align])
        )[0, 1].item()
        layer_summary["entropy_alignment_correlation"] = corr

        all_layer_stats[f"layer_{layer}"] = layer_summary
        all_layer_tensors[layer] = {
            "entropy": la_stats["entropy"],
            "max_mean_ratio": la_stats["max_mean_ratio"],
            "top1_concentration": la_stats["top1_concentration"],
            "top5_concentration": la_stats["top5_concentration"],
            "geo_alignment": geo_align,
        }

        # Print summary
        print(f"  Entropy:           mean={layer_summary['entropy']['mean']:.2f}, "
              f"median={layer_summary['entropy']['median']:.2f}")
        print(f"  Max/Mean ratio:    mean={layer_summary['max_mean_ratio']['mean']:.2f}, "
              f"median={layer_summary['max_mean_ratio']['median']:.2f}")
        print(f"  Top-1 conc:        mean={layer_summary['top1_concentration']['mean']:.4f}, "
              f"median={layer_summary['top1_concentration']['median']:.4f}")
        print(f"  Top-5 conc:        mean={layer_summary['top5_concentration']['mean']:.4f}, "
              f"median={layer_summary['top5_concentration']['median']:.4f}")
        print(f"  Geo alignment:     mean={layer_summary['geo_alignment']['mean']:.4f}, "
              f"median={layer_summary['geo_alignment']['median']:.4f}")
        print(f"  Entropy-Align corr: {corr:.4f}")

        # Free SAE memory
        del sae
        torch.cuda.empty_cache()

    if not all_layer_stats:
        print("\nNo layers found. Check --base-dir and naming convention.")
        return

    # Save JSON results
    results_path = output_dir / "logit_attribution_stats.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "config": {
                    "base_dir": args.base_dir,
                    "model_name": args.model_name,
                    "k": args.k,
                    "alpha": args.alpha,
                    "layers": layers,
                },
                "per_layer": all_layer_stats,
            },
            f,
            indent=2,
        )
    print(f"\nJSON results saved to {results_path}")

    # Save raw tensors
    tensors_path = output_dir / "logit_attribution_tensors.pt"
    torch.save(all_layer_tensors, tensors_path)
    print(f"Raw tensors saved to {tensors_path}")

    # Plot: entropy distribution by layer
    print("Generating entropy distribution plot...")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sorted_layers = sorted(all_layer_tensors.keys())

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # --- Panel 1: Entropy violin/box by layer ---
        ax = axes[0, 0]
        entropy_data = [all_layer_tensors[l]["entropy"].numpy() for l in sorted_layers]
        bp = ax.boxplot(
            entropy_data,
            labels=[str(l) for l in sorted_layers],
            showfliers=False,
            patch_artist=True,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.7)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Entropy of softmax(la_i)")
        ax.set_title("Logit Attribution Entropy by Layer")

        # --- Panel 2: Top-5 concentration by layer ---
        ax = axes[0, 1]
        top5_data = [all_layer_tensors[l]["top5_concentration"].numpy() for l in sorted_layers]
        bp = ax.boxplot(
            top5_data,
            labels=[str(l) for l in sorted_layers],
            showfliers=False,
            patch_artist=True,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("coral")
            patch.set_alpha(0.7)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Top-5 Probability Mass")
        ax.set_title("Top-5 Concentration by Layer")

        # --- Panel 3: Entropy vs Geo Alignment scatter (all layers) ---
        ax = axes[1, 0]
        cmap = plt.cm.viridis
        for i, l in enumerate(sorted_layers):
            color = cmap(i / max(len(sorted_layers) - 1, 1))
            ent = all_layer_tensors[l]["entropy"].numpy()
            geo = all_layer_tensors[l]["geo_alignment"].numpy()
            # subsample for plotting
            n = len(ent)
            idx = torch.randperm(n)[:min(500, n)].numpy()
            ax.scatter(geo[idx], ent[idx], s=3, alpha=0.3, color=color, label=f"L{l}")
        ax.set_xlabel("Geo Alignment (max cos sim to W_E)")
        ax.set_ylabel("Entropy of softmax(la_i)")
        ax.set_title("Entropy vs Geo Alignment")
        ax.legend(fontsize=6, ncol=3, loc="upper right", markerscale=3)

        # --- Panel 4: Summary line plots ---
        ax = axes[1, 1]
        mean_entropy = [all_layer_stats[f"layer_{l}"]["entropy"]["mean"] for l in sorted_layers]
        mean_geo = [all_layer_stats[f"layer_{l}"]["geo_alignment"]["mean"] for l in sorted_layers]
        mean_top5 = [all_layer_stats[f"layer_{l}"]["top5_concentration"]["mean"] for l in sorted_layers]
        corrs = [all_layer_stats[f"layer_{l}"]["entropy_alignment_correlation"] for l in sorted_layers]

        ax2 = ax.twinx()
        ln1 = ax.plot(sorted_layers, mean_entropy, "o-", color="steelblue", label="Mean Entropy")
        ln2 = ax.plot(sorted_layers, mean_top5, "s-", color="coral", label="Mean Top-5 Conc")
        ln3 = ax2.plot(sorted_layers, mean_geo, "^-", color="green", label="Mean Geo Align")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Entropy / Top-5 Conc")
        ax2.set_ylabel("Geo Alignment")
        ax.set_title("Summary Metrics by Layer")
        lns = ln1 + ln2 + ln3
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, fontsize=8, loc="best")
        ax.set_xticks(sorted_layers)

        fig.tight_layout()
        plot_path = output_dir / "entropy_distribution_by_layer.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Plot saved to {plot_path}")
    except ImportError:
        print("matplotlib not available, skipping plot")

    print("\nDone.")


if __name__ == "__main__":
    main()
