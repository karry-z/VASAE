"""Context-based interpretability analysis for trained SAE models.

For high-alignment features, extract top activating contexts and verify
semantic relevance. Report feature usage and dead feature statistics.

Usage:
    python scripts/analyze_context_interpretability.py \
        --model-path /path/to/sae.pth \
        --alignment-path /path/to/alignment_results.json \
        --data-dir /path/to/activations \
        --layer-name transformer.h.11 \
        --output-dir /path/to/output
"""

import argparse
import heapq
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

from vasae.configs.data import DataConfig
from vasae.data.dataset import GPT2LayerActivations
from vasae.models.sae_hf import SAEConfig, SAEModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Context-based interpretability analysis of SAE features"
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--alignment-path", type=str, required=True,
                        help="Path to alignment_results.json from analyze_feature_vocab_alignment")
    parser.add_argument("--data-dir", type=str,
                        default="/scratch/b5bq/pu22650.b5bq/activations_gpt2_Geralt-Targaryen_openwebtext2")
    parser.add_argument("--layer-name", type=str, default="transformer.h.11")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--blackbox-model-dir", type=str,
                        default="/scratch/b5bq/pu22650.b5bq/VASAE_out/BlackBoxModels/gpt2")
    parser.add_argument("--dim-input", type=int, default=768)
    parser.add_argument("--dim-sparse", type=int, default=50257)
    parser.add_argument("--sparsity-type", type=str, default="topk")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--num-top-features", type=int, default=200,
                        help="Number of top-aligned features to track")
    parser.add_argument("--top-contexts", type=int, default=20,
                        help="Number of top activating contexts per feature")
    parser.add_argument("--context-window", type=int, default=10,
                        help="Number of tokens before/after activation position")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-batchsize", type=int, default=0,
                        help="Limit number of batches (0=all)")
    return parser.parse_args()


def load_sae(args) -> SAEModel:
    cfg = SAEConfig(
        dim_input=args.dim_input,
        dim_sparse=args.dim_sparse,
        encoder_type="linear",
        sparsity_type=args.sparsity_type,
        k=args.k,
        nonneg_latents=True,
        tied_decoder=False,
        use_lowrank=False,
    )
    model = SAEModel(cfg)
    state_dict = torch.load(args.model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def get_high_alignment_features(alignment_path: str, num_top: int) -> list[int]:
    """Get feature indices with highest alignment from alignment_results.json."""
    with open(alignment_path) as f:
        results = json.load(f)

    # Load max_sims tensor from same directory
    alignment_dir = Path(alignment_path).parent
    max_sims_path = alignment_dir / "max_sims.pt"
    if max_sims_path.exists():
        max_sims = torch.load(max_sims_path, map_location="cpu", weights_only=True)
        _, top_indices = max_sims.topk(min(num_top, max_sims.size(0)))
        return top_indices.tolist()

    # Fallback: use examples from alignment_results.json
    strong_examples = results.get("examples", {}).get("strong", [])
    return [e["feature_id"] for e in strong_examples[:num_top]]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading SAE model...")
    model = load_sae(args).to(device)

    # Get high-alignment features
    print("Loading alignment results...")
    tracked_features = get_high_alignment_features(args.alignment_path, args.num_top_features)
    tracked_set = set(tracked_features)
    print(f"Tracking {len(tracked_features)} high-alignment features")

    # Load alignment info for aligned tokens
    alignment_dir = Path(args.alignment_path).parent
    max_sims_path = alignment_dir / "max_sims.pt"
    # Load the full alignment results for top-1 token info
    with open(args.alignment_path) as f:
        alignment_results = json.load(f)

    # Load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Load dataset
    print("Loading dataset...")
    data_cfg = DataConfig(
        train_batchsize=1,
        valid_batchsize=1,
        test_batchsize=1,
        use_centralize=False,
        layer_name=args.layer_name,
        data_dir=Path(args.data_dir),
    )
    dataset = GPT2LayerActivations(data_cfg)
    print(f"Dataset size: {len(dataset)}")

    # Initialize tracking structures
    # For each tracked feature: min-heap of (activation_value, sample_idx, position)
    top_k = args.top_contexts
    context_heaps: dict[int, list] = {f: [] for f in tracked_features}

    # Global feature usage stats
    n_features = args.dim_sparse
    feature_activation_count = torch.zeros(n_features, dtype=torch.long)
    feature_activation_sum = torch.zeros(n_features, dtype=torch.float64)
    feature_activation_max = torch.zeros(n_features) - float("inf")
    total_positions = 0

    # Process dataset
    print("Processing dataset...")
    n_samples = len(dataset)
    if args.max_batchsize > 0:
        n_samples = min(n_samples, args.max_batchsize)

    for sample_idx in range(n_samples):
        if sample_idx % 500 == 0:
            print(f"  Processing sample {sample_idx}/{n_samples}")

        item = dataset[sample_idx]
        activations = item["activations"].to(device)  # (seq_len, dim_input)
        display_text = item["display_text"]

        with torch.no_grad():
            _, sparse_acts = model.encode(activations)  # (seq_len, dim_sparse)

        sparse_cpu = sparse_acts.cpu()
        seq_len = sparse_cpu.size(0)
        total_positions += seq_len

        # Update global stats
        active_mask = sparse_cpu != 0
        feature_activation_count += active_mask.sum(dim=0).long()
        feature_activation_sum += sparse_cpu.abs().sum(dim=0).double()
        batch_max = sparse_cpu.abs().max(dim=0)[0]
        feature_activation_max = torch.maximum(feature_activation_max, batch_max)

        # Track top contexts for high-alignment features
        for feat_id in tracked_features:
            feat_acts = sparse_cpu[:, feat_id]  # (seq_len,)
            nonzero_mask = feat_acts != 0
            if not nonzero_mask.any():
                continue

            nonzero_indices = nonzero_mask.nonzero(as_tuple=True)[0]
            for pos in nonzero_indices:
                pos = pos.item()
                val = feat_acts[pos].item()
                entry = (abs(val), sample_idx, pos, val, display_text)

                heap = context_heaps[feat_id]
                if len(heap) < top_k:
                    heapq.heappush(heap, entry)
                elif abs(val) > heap[0][0]:
                    heapq.heapreplace(heap, entry)

    print(f"Processed {n_samples} samples, {total_positions} total positions")

    # Compute statistics
    dead_features = (feature_activation_count == 0).sum().item()
    alive_features = n_features - dead_features
    mean_activation_freq = feature_activation_count.float() / max(total_positions, 1)

    feature_usage_stats = {
        "total_features": n_features,
        "dead_features": dead_features,
        "dead_feature_pct": dead_features / n_features * 100,
        "alive_features": alive_features,
        "total_positions": total_positions,
        "mean_activation_frequency": mean_activation_freq.mean().item(),
        "median_activation_frequency": mean_activation_freq.median().item(),
        "p95_activation_frequency": mean_activation_freq.quantile(0.95).item(),
        "activation_frequency_histogram": {
            "never_active": (feature_activation_count == 0).sum().item(),
            "rare (<0.1%)": ((mean_activation_freq > 0) & (mean_activation_freq < 0.001)).sum().item(),
            "low [0.1%, 1%)": ((mean_activation_freq >= 0.001) & (mean_activation_freq < 0.01)).sum().item(),
            "medium [1%, 10%)": ((mean_activation_freq >= 0.01) & (mean_activation_freq < 0.1)).sum().item(),
            "high (>=10%)": (mean_activation_freq >= 0.1).sum().item(),
        },
    }

    print(f"\n=== Feature Usage Stats ===")
    print(f"  Dead features: {dead_features} ({dead_features/n_features*100:.1f}%)")
    print(f"  Alive features: {alive_features}")
    print(f"  Mean activation freq: {mean_activation_freq.mean().item():.6f}")

    # Build context results
    window = args.context_window
    context_results = {}
    consistency_total = 0
    consistency_hit = 0

    for feat_id in tracked_features:
        heap = context_heaps[feat_id]
        if not heap:
            context_results[str(feat_id)] = {"top_contexts": [], "n_activations": 0}
            continue

        sorted_contexts = sorted(heap, key=lambda x: -x[0])
        contexts = []
        for abs_val, sample_idx, pos, val, display_text in sorted_contexts:
            # Extract token window
            tokens = display_text.split() if isinstance(display_text, str) else display_text
            # display_text is typically a list of token strings or a single string
            # We'll use the raw string and mark position
            start = max(0, pos - window)
            end = min(len(display_text) if isinstance(display_text, list) else pos + window + 1, pos + window + 1)

            ctx = {
                "activation_value": round(val, 4),
                "sample_idx": sample_idx,
                "position": pos,
                "context_window": display_text[start:end] if isinstance(display_text, list) else display_text,
            }
            contexts.append(ctx)

        context_results[str(feat_id)] = {
            "top_contexts": contexts,
            "n_activations": int(feature_activation_count[feat_id].item()),
            "mean_activation": float(feature_activation_sum[feat_id] / max(feature_activation_count[feat_id].item(), 1)),
            "max_activation": float(feature_activation_max[feat_id].item()),
        }

    # Save results
    context_path = output_dir / "context_results.json"
    with open(context_path, "w") as f:
        json.dump(context_results, f, indent=2, ensure_ascii=False)
    print(f"\nContext results saved to {context_path}")

    usage_path = output_dir / "feature_usage_stats.json"
    with open(usage_path, "w") as f:
        json.dump(feature_usage_stats, f, indent=2)
    print(f"Feature usage stats saved to {usage_path}")

    # Consistency stats (tracked features only)
    consistency_stats = {
        "tracked_features": len(tracked_features),
        "features_with_activations": sum(
            1 for f in tracked_features if feature_activation_count[f] > 0
        ),
        "features_dead": sum(
            1 for f in tracked_features if feature_activation_count[f] == 0
        ),
    }
    consistency_path = output_dir / "consistency_stats.json"
    with open(consistency_path, "w") as f:
        json.dump(consistency_stats, f, indent=2)
    print(f"Consistency stats saved to {consistency_path}")


if __name__ == "__main__":
    main()
