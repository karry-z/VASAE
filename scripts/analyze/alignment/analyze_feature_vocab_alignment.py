"""Analyze cosine similarity between plain SAE decoder features and GPT-2 vocab embeddings.

Usage:
    python scripts/analyze/alignment/analyze_feature_vocab_alignment.py \
        --model-path /path/to/sae.pth \
        --blackbox-model-dir /path/to/BlackBoxModels/gpt2 \
        --output-dir /path/to/output \
        --top-k 10
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

from vasae.analysis.alignment import compute_geometric_alignment
from vasae.analysis.stats import summarize_tensor as compute_statistics_shared
from vasae.models.factory import BlackBoxModelConfig, load_embedding_layer
from vasae.models.sae import SAEConfig, SAEModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze feature-vocab alignment of a trained SAE"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to legacy state_dict (.pth/.pt) or HF SAE directory/model.safetensors",
    )
    parser.add_argument(
        "--blackbox-model-dir",
        type=str,
        default="/scratch/b5bq/pu22650.b5bq/VASAE_out/BlackBoxModels/gpt2",
        help="Directory containing emb.pth and unemb.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top tokens to report per feature",
    )
    parser.add_argument(
        "--dim-input", type=int, default=768, help="Input dimension of the SAE"
    )
    parser.add_argument(
        "--dim-sparse", type=int, default=50257, help="Sparse dimension of the SAE"
    )
    parser.add_argument(
        "--sparsity-type",
        type=str,
        default="topk",
        help="Sparsity type used in training",
    )
    parser.add_argument("--k", type=int, default=8, help="k used in training")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--embedding-override",
        type=str,
        default=None,
        help="Path to custom embedding tensor (.pt) to use instead of GPT-2 W_E",
    )
    return parser.parse_args()


def load_sae(args) -> SAEModel:
    """Load SAE from HF safetensors directory or legacy state_dict file."""
    model_path = Path(args.model_path)
    suffix = model_path.suffix.lower()

    if model_path.is_dir() or suffix == ".safetensors":
        pretrained_dir = model_path if model_path.is_dir() else model_path.parent
        if not (pretrained_dir / "config.json").exists():
            raise FileNotFoundError(
                f"Expected config.json in {pretrained_dir} when loading via from_pretrained"
            )
        model = SAEModel.from_pretrained(pretrained_dir)
        model.eval()
        return model

    if suffix in {".pth", ".pt"}:
        cfg = SAEConfig(
            dim_input=args.dim_input,
            dim_sparse=args.dim_sparse,
            encoder_type="linear",
            sparsity_type=args.sparsity_type,
            k=args.k,
            nonneg_latents=True,
            tied_decoder=False,
        )
        model = SAEModel(cfg)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    raise ValueError(
        f"Unsupported checkpoint format: {model_path}. "
        "Expected one of: .pth/.pt file, .safetensors file, or HF model directory"
    )


@torch.no_grad()
def compute_alignment(
    decoder_weight: torch.Tensor,
    embedding_weight: torch.Tensor,
    top_k: int,
    device: torch.device,
    batch_size: int = 1024,
) -> dict:
    """Compute cosine similarity between decoder features and vocab embeddings."""
    features = decoder_weight.T  # (dim_sparse, dim_input)
    result = compute_geometric_alignment(
        features, embedding_weight, top_k=top_k, batch_size=batch_size, device=device
    )
    return {
        "max_sims": result.max_sims,
        "topk_sims": result.topk_sims,
        "topk_indices": result.topk_indices,
    }


def compute_statistics(max_sims: torch.Tensor) -> dict:
    """Compute summary statistics of max cosine similarities."""
    return compute_statistics_shared(max_sims)


def compute_alignment_categories(max_sims: torch.Tensor) -> dict:
    """Categorize features by alignment strength."""
    n = max_sims.size(0)
    strong = (max_sims >= 0.8).sum().item()
    medium = ((max_sims >= 0.5) & (max_sims < 0.8)).sum().item()
    weak = ((max_sims >= 0.3) & (max_sims < 0.5)).sum().item()
    none_ = (max_sims < 0.3).sum().item()
    return {
        "strong (>=0.8)": {"count": strong, "pct": strong / n * 100},
        "medium [0.5, 0.8)": {"count": medium, "pct": medium / n * 100},
        "weak [0.3, 0.5)": {"count": weak, "pct": weak / n * 100},
        "none (<0.3)": {"count": none_, "pct": none_ / n * 100},
    }


def save_histogram(max_sims: torch.Tensor, output_dir: Path):
    """Save a histogram of max cosine similarities."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(max_sims.numpy(), bins=100, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Max Cosine Similarity")
        ax.set_ylabel("Number of Features")
        ax.set_title("Distribution of Max Cosine Similarity (Feature vs Vocab)")
        ax.axvline(x=0.8, color="r", linestyle="--", label="Strong (0.8)")
        ax.axvline(x=0.5, color="orange", linestyle="--", label="Medium (0.5)")
        ax.axvline(x=0.3, color="gray", linestyle="--", label="Weak (0.3)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "max_sim_histogram.png", dpi=150)
        plt.close(fig)
        print(f"Histogram saved to {output_dir / 'max_sim_histogram.png'}")
    except ImportError:
        print("matplotlib not available, skipping histogram")


def format_examples(
    topk_indices: torch.Tensor,
    topk_sims: torch.Tensor,
    max_sims: torch.Tensor,
    tokenizer,
    n_examples: int = 5,
    top_display: int = 5,
) -> dict:
    """Select representative features from each alignment category."""
    examples = {}
    categories = {
        "strong": max_sims >= 0.8,
        "medium": (max_sims >= 0.5) & (max_sims < 0.8),
        "weak": (max_sims >= 0.3) & (max_sims < 0.5),
        "none": max_sims < 0.3,
    }

    for cat_name, mask in categories.items():
        indices = mask.nonzero(as_tuple=True)[0]
        if len(indices) == 0:
            examples[cat_name] = []
            continue

        # Pick features with highest max_sim within category
        cat_sims = max_sims[indices]
        _, sorted_idx = cat_sims.sort(descending=True)
        selected = indices[sorted_idx[:n_examples]]

        cat_examples = []
        for feat_id in selected:
            feat_id = feat_id.item()
            tokens_with_sim = []
            for j in range(min(top_display, topk_indices.size(1))):
                token_id = topk_indices[feat_id, j].item()
                sim_val = topk_sims[feat_id, j].item()
                token_str = tokenizer.decode([token_id])
                tokens_with_sim.append(
                    {"token": token_str, "token_id": token_id, "sim": round(sim_val, 4)}
                )
            cat_examples.append(
                {
                    "feature_id": feat_id,
                    "max_sim": round(max_sims[feat_id].item(), 4),
                    "top_tokens": tokens_with_sim,
                }
            )
        examples[cat_name] = cat_examples

    return examples


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load SAE
    print("Loading SAE model...")
    model = load_sae(args)

    # Load vocab embeddings
    if args.embedding_override:
        print(f"Loading custom embedding from {args.embedding_override}...")
        emb_weight = torch.load(
            args.embedding_override, map_location="cpu", weights_only=True
        )
        emb = torch.nn.Embedding.from_pretrained(emb_weight, freeze=True)
    else:
        print("Loading vocab embeddings...")
        bb_cfg = BlackBoxModelConfig(name="gpt2", dir=Path(args.blackbox_model_dir))
        emb = load_embedding_layer(bb_cfg)

    # Get weights
    decoder_weight = model.decoder.weight.data  # (dim_input, dim_sparse)
    embedding_weight = emb.weight.data  # (vocab_size, dim_input)

    print(f"Decoder weight shape: {decoder_weight.shape}")
    print(f"Embedding weight shape: {embedding_weight.shape}")

    # Compute alignment
    print("Computing cosine similarities...")
    result = compute_alignment(
        decoder_weight, embedding_weight, top_k=args.top_k, device=device
    )

    max_sims = result["max_sims"]
    topk_sims = result["topk_sims"]
    topk_indices = result["topk_indices"]

    # Statistics
    stats = compute_statistics(max_sims)
    print("\n=== Max Cosine Similarity Statistics ===")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    # Categories
    categories = compute_alignment_categories(max_sims)
    print("\n=== Alignment Categories ===")
    for cat, info in categories.items():
        print(f"  {cat}: {info['count']} ({info['pct']:.1f}%)")

    # Histogram
    save_histogram(max_sims, output_dir)

    # Token examples
    print("\nLoading tokenizer for token decoding...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    examples = format_examples(topk_indices, topk_sims, max_sims, tokenizer)

    print("\n=== Example Features ===")
    for cat_name, feats in examples.items():
        print(f"\n--- {cat_name.upper()} alignment ---")
        for feat in feats:
            tokens_str = ", ".join(
                f"'{t['token']}' ({t['sim']:.3f})" for t in feat["top_tokens"]
            )
            print(f"  Feature {feat['feature_id']}: max_sim={feat['max_sim']:.4f}")
            print(f"    Top tokens: {tokens_str}")

    # Save results
    output = {
        "statistics": stats,
        "categories": {k: v for k, v in categories.items()},
        "examples": examples,
        "config": {
            "model_path": args.model_path,
            "top_k": args.top_k,
            "dim_input": args.dim_input,
            "dim_sparse": args.dim_sparse,
        },
    }

    results_path = output_dir / "alignment_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")

    # Save raw max_sims tensor
    torch.save(max_sims, output_dir / "max_sims.pt")
    print(f"Raw max_sims tensor saved to {output_dir / 'max_sims.pt'}")


if __name__ == "__main__":
    main()
