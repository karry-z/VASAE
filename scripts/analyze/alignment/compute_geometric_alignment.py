"""Compute geometric alignment for SAE checkpoints and save max_sims.

Lightweight alternative to analyze_alignment_quality.py when only geometric
alignment (cosine similarity between decoder features and token embeddings)
is needed. Supports arbitrary checkpoint paths via --sae-paths.

Usage:
    uv run python scripts/analyze/alignment/compute_geometric_alignment.py \
        --model-name meta-llama/Llama-3.1-8B \
        --sae-paths 0:/path/to/L0_checkpoint 15:/path/to/L15_checkpoint \
        --output-dir exp/F002_AlignmentAnalysis/llama_5e-3 \
        --device cuda
"""

import argparse
import json
from pathlib import Path

import torch

from shared_utils.log import get_logger
from vasae.analysis.alignment import compute_geometric_alignment
from vasae.analysis.sae_loader import get_decoder_features, load_sae_for_analysis
from vasae.models.factory import get_embedding, load_model

log = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Compute geometric alignment for SAE checkpoints")
    p.add_argument("--model-name", type=str, required=True)
    p.add_argument(
        "--sae-paths",
        nargs="+",
        required=True,
        help="Layer:path pairs, e.g. 0:/path/to/checkpoint 15:/path/to/checkpoint",
    )
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def parse_sae_paths(sae_paths: list[str]) -> dict[int, Path]:
    result = {}
    for entry in sae_paths:
        layer_str, path = entry.split(":", 1)
        result[int(layer_str)] = Path(path)
    return result


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    checkpoints = parse_sae_paths(args.sae_paths)

    log.info("Loading %s...", args.model_name)
    lm_model, tokenizer = load_model(args.model_name, device=device)
    W_E = get_embedding(lm_model).weight.data
    vocab_size = W_E.shape[0]
    log.info("vocab_size=%d, embed_dim=%d", vocab_size, W_E.shape[1])

    for layer_idx in sorted(checkpoints):
        path = checkpoints[layer_idx]
        log.info("=== Layer %d: %s ===", layer_idx, path)

        sae = load_sae_for_analysis(path, device=device)
        features = get_decoder_features(sae)
        n_features = features.shape[0]
        log.info("  n_features=%d", n_features)

        geo = compute_geometric_alignment(features, W_E, top_k=1, device=device)
        max_sims = geo.max_sims
        top1_tokens = geo.topk_indices[:, 0]

        aligned_mask = max_sims >= 0.8
        n_aligned = aligned_mask.sum().item()
        unique_tokens = top1_tokens[aligned_mask].unique().numel() if n_aligned > 0 else 0

        log.info(
            "  max_sim: mean=%.4f, median=%.4f, max=%.4f",
            max_sims.mean(), max_sims.median(), max_sims.max(),
        )
        log.info(
            "  aligned (s>=0.8): %d/%d (%.2f%%)",
            n_aligned, n_features, n_aligned / n_features * 100,
        )

        # Save in same format as analyze_alignment_quality.py
        layer_dir = output_dir / f"L{layer_idx}"
        layer_dir.mkdir(exist_ok=True)

        torch.save({"max_sims": max_sims}, layer_dir / "max_sims.pt")

        result = {
            "layer_idx": layer_idx,
            "n_features": n_features,
            "n_aligned": n_aligned,
            "geometric": {
                "aligned_pct": round(n_aligned / n_features * 100, 2),
                "unique_tokens_covered": unique_tokens,
                "coverage_pct": round(unique_tokens / vocab_size * 100, 2),
                "max_sim_mean": round(max_sims.mean().item(), 4),
                "max_sim_median": round(max_sims.median().item(), 4),
            },
        }
        with open(layer_dir / "results.json", "w") as f:
            json.dump(result, f, indent=2)

        log.info("  Saved to %s", layer_dir)

        del sae, features, geo
        torch.cuda.empty_cache()

    log.info("Done.")


if __name__ == "__main__":
    main()
