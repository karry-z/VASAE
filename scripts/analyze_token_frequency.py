"""Analyze token frequency bias in feature-vocab alignment.

Check whether high-alignment features preferentially align to rare tokens.
Compare plain SAE (001) vs anchor SAE (002) alignment distributions.

Usage:
    python scripts/analyze_token_frequency.py \
        --data-dir /path/to/activations \
        --alignment-dirs dir1:label1 dir2:label2 ... \
        --output-dir /path/to/output \
        --blackbox-model-dir /path/to/BlackBoxModels/gpt2
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

from vasae.models.factory import BlackBoxModelConfig, load_embeding_layer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze token frequency bias in alignment"
    )
    parser.add_argument("--data-dir", type=str,
                        default="/scratch/b5bq/pu22650.b5bq/activations_gpt2_Geralt-Targaryen_openwebtext2")
    parser.add_argument("--alignment-dirs", nargs="+", required=True,
                        help="Pairs of dir:label, e.g. /path/to/001_analysis:plain /path/to/002_analysis:anchor_1e-3")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--blackbox-model-dir", type=str,
                        default="/scratch/b5bq/pu22650.b5bq/VASAE_out/BlackBoxModels/gpt2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--knn-k", type=int, default=10,
                        help="k for kNN isolation analysis in embedding space")
    return parser.parse_args()


def compute_token_frequencies(data_dir: Path) -> Counter:
    """Compute token frequencies from data_info.json display_text."""
    data_info_path = data_dir / "data_info.json"
    print(f"Loading data_info from {data_info_path}...")
    with open(data_info_path) as f:
        data_info = json.load(f)

    freq = Counter()
    for item in data_info:
        text = item.get("display_text", "")
        if isinstance(text, list):
            for t in text:
                freq[t] += 1
        elif isinstance(text, str):
            # tokenize with GPT-2 tokenizer
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            token_ids = tokenizer.encode(text)
            for tid in token_ids:
                freq[tid] += 1
    return freq


def load_alignment(alignment_dir: str) -> dict:
    """Load alignment_results.json and max_sims.pt from a directory."""
    d = Path(alignment_dir)
    results_path = d / "alignment_results.json"
    max_sims_path = d / "max_sims.pt"

    with open(results_path) as f:
        results = json.load(f)

    max_sims = None
    if max_sims_path.exists():
        max_sims = torch.load(max_sims_path, map_location="cpu", weights_only=True)

    return {"results": results, "max_sims": max_sims, "dir": d}


@torch.no_grad()
def compute_knn_distances(embedding_weight: torch.Tensor, k: int, device: torch.device) -> torch.Tensor:
    """Compute mean k-NN distance for each token in embedding space."""
    emb = F.normalize(embedding_weight.to(device), dim=1)
    n = emb.size(0)
    knn_dists = torch.zeros(n, device=device)

    batch_size = 1024
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = emb[start:end]
        sim = batch @ emb.T  # (batch, n)
        # Get top-(k+1) and exclude self
        topk_sims, _ = sim.topk(k + 1, dim=1)
        # Mean of k nearest (excluding self which has sim=1.0)
        knn_dists[start:end] = topk_sims[:, 1:].mean(dim=1)

    return knn_dists.cpu()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Parse alignment dirs
    alignment_configs = []
    for item in args.alignment_dirs:
        parts = item.split(":")
        if len(parts) == 2:
            alignment_configs.append({"dir": parts[0], "label": parts[1]})
        else:
            alignment_configs.append({"dir": parts[0], "label": Path(parts[0]).name})

    # Load token frequencies
    token_freq = compute_token_frequencies(Path(args.data_dir))
    total_tokens = sum(token_freq.values())
    print(f"Total token occurrences: {total_tokens}")
    print(f"Unique tokens seen: {len(token_freq)}")

    # Compute frequency rank for all tokens (0-indexed, 0=most frequent)
    vocab_size = 50257
    freq_array = torch.zeros(vocab_size)
    for tid, count in token_freq.items():
        if isinstance(tid, int) and tid < vocab_size:
            freq_array[tid] = count
        elif isinstance(tid, str):
            # Skip string tokens - they need tokenization
            pass

    _, freq_rank = freq_array.sort(descending=True)
    token_to_rank = torch.zeros(vocab_size, dtype=torch.long)
    token_to_rank[freq_rank] = torch.arange(vocab_size)

    # Load embedding for isolation analysis
    print("Loading vocab embeddings...")
    bb_cfg = BlackBoxModelConfig(name="gpt2", dir=Path(args.blackbox_model_dir))
    emb = load_embeding_layer(bb_cfg)

    print("Computing kNN distances in embedding space...")
    knn_dists = compute_knn_distances(emb.weight.data, args.knn_k, device)

    # Analyze each alignment config
    comparison = {}
    for cfg in alignment_configs:
        label = cfg["label"]
        print(f"\n=== Analyzing: {label} ===")

        alignment = load_alignment(cfg["dir"])
        max_sims = alignment["max_sims"]

        if max_sims is None:
            print(f"  Skipping {label}: no max_sims.pt found")
            continue

        # For each feature, find its top-1 aligned token
        # We need to recompute or load topk_indices
        # Try loading from the alignment dir
        results = alignment["results"]

        # Reconstruct top-1 token from examples if available, otherwise recompute
        # For simplicity, we'll look at strong-alignment features from examples
        examples = results.get("examples", {})
        strong_feats = examples.get("strong", [])

        # Get all features above threshold 0.5
        high_mask = max_sims >= 0.5
        high_indices = high_mask.nonzero(as_tuple=True)[0]
        n_high = high_indices.size(0)

        # We need top-1 token for each feature - check for saved data
        # The alignment script saves topk_indices implicitly through examples
        # For a proper analysis, we'd need the full topk_indices tensor
        # Let's compute frequency stats from the examples we have

        all_feat_stats = []
        for cat_name in ["strong", "medium", "weak", "none"]:
            feats = examples.get(cat_name, [])
            for feat in feats:
                top_tokens = feat.get("top_tokens", [])
                if top_tokens:
                    top1_id = top_tokens[0]["token_id"]
                    rank = token_to_rank[top1_id].item() if top1_id < vocab_size else -1
                    freq = freq_array[top1_id].item() if top1_id < vocab_size else 0
                    isolation = knn_dists[top1_id].item() if top1_id < vocab_size else 0
                    all_feat_stats.append({
                        "feature_id": feat["feature_id"],
                        "max_sim": feat["max_sim"],
                        "top1_token_id": top1_id,
                        "top1_token": top_tokens[0]["token"],
                        "frequency_rank": rank,
                        "frequency_count": freq,
                        "frequency_pct": freq / max(total_tokens, 1) * 100,
                        "knn_isolation": isolation,
                        "category": cat_name,
                    })

        comparison[label] = {
            "n_features": int(max_sims.size(0)),
            "n_high_alignment": n_high,
            "statistics": {
                "mean_max_sim": max_sims.mean().item(),
                "strong_pct": (max_sims >= 0.8).sum().item() / max_sims.size(0) * 100,
            },
            "feature_frequency_stats": all_feat_stats,
        }

        # Print summary
        if all_feat_stats:
            ranks = [s["frequency_rank"] for s in all_feat_stats if s["category"] == "strong"]
            if ranks:
                import statistics
                print(f"  Strong-aligned features: {len(ranks)}")
                print(f"  Median frequency rank of aligned tokens: {statistics.median(ranks):.0f}")
                print(f"  Mean frequency rank: {statistics.mean(ranks):.0f}")

    # Save comparison
    freq_path = output_dir / "freq_comparison.json"
    with open(freq_path, "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\nFrequency comparison saved to {freq_path}")

    # Save isolation analysis
    isolation_stats = {
        "knn_k": args.knn_k,
        "mean_knn_distance_all_tokens": knn_dists.mean().item(),
        "median_knn_distance_all_tokens": knn_dists.median().item(),
    }
    isolation_path = output_dir / "isolation_analysis.json"
    with open(isolation_path, "w") as f:
        json.dump(isolation_stats, f, indent=2)
    print(f"Isolation analysis saved to {isolation_path}")

    # Plot histogram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Frequency rank histogram
        ax = axes[0]
        for label, data in comparison.items():
            strong_ranks = [
                s["frequency_rank"]
                for s in data["feature_frequency_stats"]
                if s["category"] == "strong"
            ]
            if strong_ranks:
                ax.hist(strong_ranks, bins=50, alpha=0.5, label=label, edgecolor="black")
        ax.set_xlabel("Token Frequency Rank (0=most frequent)")
        ax.set_ylabel("Count")
        ax.set_title("Frequency Rank of Top-1 Aligned Tokens (Strong Features)")
        ax.legend()

        # kNN isolation
        ax = axes[1]
        for label, data in comparison.items():
            strong_iso = [
                s["knn_isolation"]
                for s in data["feature_frequency_stats"]
                if s["category"] == "strong"
            ]
            if strong_iso:
                ax.hist(strong_iso, bins=50, alpha=0.5, label=label, edgecolor="black")
        ax.set_xlabel("Mean kNN Cosine Similarity")
        ax.set_ylabel("Count")
        ax.set_title("Embedding Isolation of Aligned Tokens (Strong Features)")
        ax.legend()

        fig.tight_layout()
        fig.savefig(output_dir / "freq_histogram.png", dpi=150)
        plt.close(fig)
        print(f"Histogram saved to {output_dir / 'freq_histogram.png'}")
    except ImportError:
        print("matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
