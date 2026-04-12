"""Feature alignment quality analysis for F002.

For each SAE layer checkpoint, compute:
  1. Geometric alignment: cosine similarity between decoder features and token embeddings
  2. Input/output correlation: Pearson correlation between feature activations
     and semantic similarity to aligned token embeddings

Then categorize features into dual / input_related / output_related / non_functional.

Usage (single layer, via Slurm array):
    uv run python scripts/analyze/alignment/analyze_alignment_quality.py \
        --results-dir /scratch/b5bq/pu22650.b5bq/VASAE_out/001_F_Benchmarking \
        --model-name gpt2 \
        --variant soft \
        --layer-idx 6 \
        --output-dir exp/F002_AlignmentAnalysis/gpt2 \
        --device cuda

Usage (explicit SAE paths, e.g. Llama ablation checkpoints):
    uv run python scripts/analyze/alignment/analyze_alignment_quality.py \
        --model-name meta-llama/Llama-3.1-8B \
        --sae-paths 0:/path/to/L0 15:/path/to/L15 \
        --layer-idx 0 \
        --output-dir exp/F002_AlignmentAnalysis/llama_5e-3 \
        --device cuda
"""

import argparse
import json
import os
import re
from pathlib import Path

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

import torch
import torch.nn.functional as F
from nnsight import NNsight

from shared_utils.log import get_logger
from vasae.analysis.alignment import compute_geometric_alignment
from vasae.analysis.sae_loader import get_decoder_features, load_sae_for_analysis
from vasae.engine.intervention import extract_activations
from vasae.models.factory import get_embedding, get_layers, load_model
from vasae.models.sae import SAEModel

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Feature alignment quality analysis (F002)"
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing SAE checkpoints (e.g. 001_F_Benchmarking)",
    )
    p.add_argument(
        "--sae-paths",
        type=str,
        nargs="*",
        default=None,
        help="Explicit layer:path pairs (e.g. 0:/path/to/L0 15:/path/to/L15)",
    )
    p.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="HuggingFace model name (e.g. gpt2, meta-llama/Llama-3.1-8B)",
    )
    p.add_argument(
        "--variant", type=str, default="soft", help="Checkpoint variant to analyze"
    )
    p.add_argument(
        "--baseline-variant",
        type=str,
        default="plain",
        help="Baseline variant for geometric comparison",
    )
    p.add_argument(
        "--layer-idx",
        type=int,
        default=None,
        help="Single layer to analyze (default: all layers)",
    )
    p.add_argument("--dataset", type=str, default="wikitext")
    p.add_argument("--dataset-config", type=str, default="wikitext-103-raw-v1")
    p.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Number of text samples for correlation analysis",
    )
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--top-m",
        type=int,
        default=50,
        help="M: number of top output tokens for output semantic vector",
    )
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def discover_checkpoints(results_dir: str, model_name: str, variant: str):
    """Find all SAE checkpoints matching the model and variant pattern."""
    results_path = Path(results_dir)
    model_tag = "gpt2" if "gpt2" in model_name else "llama"
    pattern = re.compile(rf"001F_{model_tag}_L(\d+)_{variant}$")

    checkpoints = {}
    for d in sorted(results_path.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if m:
            layer = int(m.group(1))
            checkpoints[layer] = d
    return checkpoints


def parse_sae_paths(sae_paths: list[str]) -> dict[int, Path]:
    """Parse layer:path pairs into a dict."""
    checkpoints = {}
    for entry in sae_paths:
        layer_str, path_str = entry.split(":", 1)
        checkpoints[int(layer_str)] = Path(path_str)
    return checkpoints


# ---------------------------------------------------------------------------
# Online Pearson correlation accumulator
# ---------------------------------------------------------------------------


class PearsonAccumulator:
    """Accumulate statistics for online Pearson correlation computation.

    For each of d_a aligned features, tracks:
      sum_z, sum_u, sum_zu, sum_z2, sum_u2, count
    across all (batch, position) samples.
    """

    def __init__(self, n_features: int):
        self.sum_z = torch.zeros(n_features, dtype=torch.float64)
        self.sum_u = torch.zeros(n_features, dtype=torch.float64)
        self.sum_zu = torch.zeros(n_features, dtype=torch.float64)
        self.sum_z2 = torch.zeros(n_features, dtype=torch.float64)
        self.sum_u2 = torch.zeros(n_features, dtype=torch.float64)
        self.count = 0

    def update(self, z: torch.Tensor, u: torch.Tensor):
        """Update with a batch of (N, d_a) tensors."""
        z_d = z.double()
        u_d = u.double()
        self.sum_z += z_d.sum(dim=0)
        self.sum_u += u_d.sum(dim=0)
        self.sum_zu += (z_d * u_d).sum(dim=0)
        self.sum_z2 += (z_d * z_d).sum(dim=0)
        self.sum_u2 += (u_d * u_d).sum(dim=0)
        self.count += z.shape[0]

    def compute(self) -> torch.Tensor:
        """Compute Pearson correlation for each feature. Returns (d_a,)."""
        n = self.count
        if n < 2:
            return torch.zeros_like(self.sum_z)
        num = n * self.sum_zu - self.sum_z * self.sum_u
        den = torch.sqrt(
            (n * self.sum_z2 - self.sum_z**2) * (n * self.sum_u2 - self.sum_u**2)
        )
        rho = num / den.clamp(min=1e-12)
        return rho.float()


# ---------------------------------------------------------------------------
# Correlation-based input/output relevance
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_correlation(
    sae: SAEModel,
    lm_model,
    nn_model: NNsight,
    tokenizer,
    layer_idx: int,
    dataset,
    W_E: torch.Tensor,
    aligned_indices: list[int],
    aligned_token_ids: torch.Tensor,
    n_samples: int,
    batch_size: int,
    max_length: int,
    top_m: int,
    device: torch.device,
):
    """Compute input and output correlation for aligned features.

    For each (batch, valid_position) sample:
      - Input similarity: cos(W_E[input_token], E) where E = W_E[aligned_token_ids]
      - Output similarity: cos(weighted_output_embedding, E)
      - Feature activation: Z[:, :, aligned_indices]
    Then compute Pearson correlation between activations and similarities.

    Returns:
        dict with rho_in, rho_out (d_a,), alive_mask (n_features,), total_positions
    """
    n_features = sae.config.dim_sparse
    n_aligned = len(aligned_indices)
    aligned_tensor = torch.tensor(aligned_indices, dtype=torch.long, device=device)

    # Precompute aligned feature token embeddings: E^(ℓ) ∈ (d_a, d)
    E_aligned = F.normalize(W_E[aligned_token_ids].float(), dim=1)  # (d_a, d)

    acc_in = PearsonAccumulator(n_aligned)
    acc_out = PearsonAccumulator(n_aligned)
    feature_ever_active = torch.zeros(n_features, dtype=torch.bool)
    total_positions = 0

    n_batches = (n_samples + batch_size - 1) // batch_size

    for batch_idx, batch_start in enumerate(range(0, n_samples, batch_size)):
        if batch_idx % 50 == 0:
            log.info("  Correlation: batch %d/%d", batch_idx, n_batches)

        batch_end = min(batch_start + batch_size, n_samples)
        batch_texts = [dataset[i]["text"] for i in range(batch_start, batch_end)]
        batch_texts = [t for t in batch_texts if t.strip()]
        if not batch_texts:
            continue

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        input_ids = enc["input_ids"]  # (B, S)
        attn_mask = enc["attention_mask"]  # (B, S)
        B, S = input_ids.shape

        # --- Forward pass: get layer activations and final logits ---
        h = extract_activations(nn_model, input_ids, layer_idx)
        h = h.detach()
        if isinstance(h, tuple):
            h = h[0]
        if h.dim() == 2:
            h = h.reshape(B, S, -1)

        with torch.no_grad():
            model_out = lm_model(input_ids=input_ids, attention_mask=attn_mask)
            logits = model_out.logits.detach()  # (B, S, V)

        # --- SAE encode ---
        D = h.shape[-1]
        _, z = sae.encode(h.reshape(-1, D).float())  # (B*S, n_features)
        z = z.reshape(B, S, -1)  # (B, S, n_features)

        # Track alive features
        mask = attn_mask.bool()  # (B, S)
        mask_flat = mask.reshape(-1)
        feature_ever_active |= (z.reshape(-1, n_features)[mask_flat] > 0).any(dim=0).cpu()
        total_positions += mask.sum().item()

        # --- Extract aligned feature activations: (B, S, d_a) ---
        z_aligned = z[:, :, aligned_tensor]  # (B, S, d_a)

        # --- Input similarity: U_in = cos(W_E[input_ids], E_aligned) ---
        # H_p^(0) = W_E[input_ids]  (B, S, d)
        h_input = F.normalize(W_E[input_ids].float(), dim=-1)  # (B, S, d)
        # U_in[b, p, f] = cos(h_input[b, p], E_aligned[f])
        u_in = torch.einsum("bsd,fd->bsf", h_input, E_aligned)  # (B, S, d_a)

        # --- Output similarity: U_out = cos(H̄_p, E_aligned) ---
        # Top-M tokens from output distribution
        probs = torch.softmax(logits.float(), dim=-1)  # (B, S, V)
        topm_probs, topm_ids = probs.topk(top_m, dim=-1)  # (B, S, M)
        # Renormalize
        topm_probs = topm_probs / topm_probs.sum(dim=-1, keepdim=True)  # (B, S, M)
        # Weighted output embedding: H̄_p = Σ_m P̃_m * W_E[V_m]
        topm_embs = W_E[topm_ids].float()  # (B, S, M, d)
        h_output = (topm_probs.unsqueeze(-1) * topm_embs).sum(dim=-2)  # (B, S, d)
        h_output = F.normalize(h_output, dim=-1)  # (B, S, d)
        u_out = torch.einsum("bsd,fd->bsf", h_output, E_aligned)  # (B, S, d_a)

        # --- Accumulate only valid positions ---
        # Flatten (B, S) → (N,) keeping only mask=True
        valid = mask.reshape(-1)  # (B*S,)
        z_flat = z_aligned.reshape(-1, n_aligned)[valid].cpu()  # (N_valid, d_a)
        u_in_flat = u_in.reshape(-1, n_aligned)[valid].cpu()  # (N_valid, d_a)
        u_out_flat = u_out.reshape(-1, n_aligned)[valid].cpu()  # (N_valid, d_a)

        acc_in.update(z_flat, u_in_flat)
        acc_out.update(z_flat, u_out_flat)

        del h, z, z_aligned, logits, probs, topm_embs, h_input, h_output, u_in, u_out, enc
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rho_in = acc_in.compute()  # (d_a,)
    rho_out = acc_out.compute()  # (d_a,)

    return {
        "rho_in": rho_in,
        "rho_out": rho_out,
        "alive_mask": feature_ever_active,
        "total_positions": total_positions,
    }


# ---------------------------------------------------------------------------
# Categorization
# ---------------------------------------------------------------------------


def categorize_features(
    aligned_indices: list[int],
    rho_in: torch.Tensor,
    rho_out: torch.Tensor,
    threshold: float = 0.1,
):
    """Categorize aligned features based on correlation thresholds.

    Args:
        aligned_indices: list of feature indices that are geometrically aligned
        rho_in: (d_a,) input correlation values
        rho_out: (d_a,) output correlation values
        threshold: correlation threshold for "related" classification
    """
    categories = {}
    for ai, fi in enumerate(aligned_indices):
        is_input = rho_in[ai].item() >= threshold
        is_output = rho_out[ai].item() >= threshold

        if is_input and is_output:
            categories[fi] = "dual"
        elif is_input:
            categories[fi] = "input_related"
        elif is_output:
            categories[fi] = "output_related"
        else:
            categories[fi] = "non_functional"

    return categories


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


def build_examples(
    categories: dict,
    geo_result,
    rho_in: torch.Tensor,
    rho_out: torch.Tensor,
    aligned_indices: list[int],
    tokenizer,
    layer_idx: int,
):
    """Pick 2-3 representative features per category for case study."""
    from collections import defaultdict

    # Map feature_id -> aligned index
    fi_to_ai = {fi: ai for ai, fi in enumerate(aligned_indices)}

    by_cat = defaultdict(list)
    for fi, cat in categories.items():
        by_cat[cat].append(fi)

    examples = []
    for cat_name in ["dual", "input_related", "output_related", "non_functional"]:
        feats = by_cat.get(cat_name, [])
        if not feats:
            continue
        # Sort by max(rho_in, rho_out) descending
        feats_sorted = sorted(
            feats,
            key=lambda fi: max(rho_in[fi_to_ai[fi]].item(), rho_out[fi_to_ai[fi]].item()),
            reverse=True,
        )
        for fi in feats_sorted[:3]:
            ai = fi_to_ai[fi]
            t_i = geo_result.topk_indices[fi, 0].item()
            entry = {
                "layer": layer_idx,
                "feature_id": fi,
                "category": cat_name,
                "geo_max_sim": round(geo_result.max_sims[fi].item(), 4),
                "aligned_token": tokenizer.decode([t_i]),
                "aligned_token_id": t_i,
                "rho_in": round(rho_in[ai].item(), 4),
                "rho_out": round(rho_out[ai].item(), 4),
            }
            examples.append(entry)

    return examples


# ---------------------------------------------------------------------------
# Main per-layer analysis
# ---------------------------------------------------------------------------


def analyze_layer(
    layer_idx: int,
    sae_path: Path,
    baseline_sae_path: Path | None,
    lm_model,
    nn_model: NNsight,
    tokenizer,
    W_E,
    dataset,
    args,
    device,
):
    """Run full analysis for a single layer."""
    log.info("=== Layer %d ===", layer_idx)

    sae = load_sae_for_analysis(sae_path, device=device)
    n_features = sae.config.dim_sparse
    log.info("  SAE: n_features=%d, k=%s", n_features, sae.config.k)

    # Step 1: Geometric alignment
    log.info("  Step 1: Geometric alignment")
    geo = compute_geometric_alignment(
        get_decoder_features(sae), W_E, top_k=1, device=device
    )
    max_sims = geo.max_sims
    top1_tokens = geo.topk_indices[:, 0]

    aligned_mask = max_sims >= 0.8
    aligned_indices = aligned_mask.nonzero(as_tuple=True)[0].tolist()
    n_aligned = len(aligned_indices)
    unique_tokens = top1_tokens[aligned_mask].unique().numel() if n_aligned > 0 else 0
    vocab_size = W_E.shape[0]

    log.info(
        "  Aligned: %d/%d (%.1f%%), unique tokens: %d/%d",
        n_aligned,
        n_features,
        n_aligned / n_features * 100,
        unique_tokens,
        vocab_size,
    )

    # Baseline geometric alignment
    geo_baseline = None
    if baseline_sae_path is not None and baseline_sae_path.exists():
        log.info("  Computing baseline geometric alignment")
        baseline_sae = load_sae_for_analysis(baseline_sae_path, device=device)
        geo_baseline = compute_geometric_alignment(
            get_decoder_features(baseline_sae), W_E, top_k=1, device=device
        )
        bl_aligned = (geo_baseline.max_sims >= 0.8).sum().item()
        log.info(
            "  Baseline aligned: %d/%d (%.1f%%)",
            bl_aligned,
            n_features,
            bl_aligned / n_features * 100,
        )
        del baseline_sae

    # Step 2: Input/output correlation
    if n_aligned == 0:
        log.info("  No aligned features — skipping correlation analysis")
        rho_in = torch.zeros(0)
        rho_out = torch.zeros(0)
        alive_mask = torch.zeros(n_features, dtype=torch.bool)
        total_positions = 0
    else:
        aligned_token_ids = top1_tokens[aligned_indices]
        log.info(
            "  Step 2: Correlation analysis (%d aligned features, %d samples)",
            n_aligned,
            args.n_samples,
        )
        corr_result = compute_correlation(
            sae,
            lm_model,
            nn_model,
            tokenizer,
            layer_idx,
            dataset,
            W_E,
            aligned_indices,
            aligned_token_ids,
            args.n_samples,
            args.batch_size,
            args.max_length,
            args.top_m,
            device,
        )
        rho_in = corr_result["rho_in"]
        rho_out = corr_result["rho_out"]
        alive_mask = corr_result["alive_mask"]
        total_positions = corr_result["total_positions"]

    n_alive = alive_mask.sum().item()
    alive_aligned = [fi for fi in aligned_indices if alive_mask[fi]]
    n_alive_aligned = len(alive_aligned)
    log.info(
        "  Alive: %d/%d, Alive+Aligned: %d/%d",
        n_alive,
        n_features,
        n_alive_aligned,
        n_aligned,
    )

    # Correlation stats for alive+aligned
    if n_alive_aligned > 0 and n_aligned > 0:
        # Map alive+aligned to their indices in rho arrays
        fi_to_ai = {fi: ai for ai, fi in enumerate(aligned_indices)}
        alive_aligned_ai = [fi_to_ai[fi] for fi in alive_aligned]
        rho_in_aa = rho_in[alive_aligned_ai]
        rho_out_aa = rho_out[alive_aligned_ai]
        log.info(
            "  Input corr: mean=%.4f, median=%.4f",
            rho_in_aa.mean().item(),
            rho_in_aa.median().item(),
        )
        log.info(
            "  Output corr: mean=%.4f, median=%.4f",
            rho_out_aa.mean().item(),
            rho_out_aa.median().item(),
        )
    else:
        rho_in_aa = torch.zeros(0)
        rho_out_aa = torch.zeros(0)

    # Step 3: Categorize alive+aligned features
    if n_alive_aligned > 0:
        categories = categorize_features(aligned_indices, rho_in, rho_out)
        # Filter to alive+aligned only
        categories = {fi: categories[fi] for fi in alive_aligned if fi in categories}
    else:
        categories = {}

    cat_counts = {}
    for cat in ["dual", "input_related", "output_related", "non_functional"]:
        cat_counts[cat] = sum(1 for c in categories.values() if c == cat)
    log.info("  Categories (of %d alive+aligned): %s", n_alive_aligned, json.dumps(cat_counts))

    # Build examples
    examples = build_examples(
        categories, geo, rho_in, rho_out, aligned_indices, tokenizer, layer_idx
    )

    # Build result dict
    rho_in_values = [round(rho_in[fi_to_ai[fi]].item(), 4) for fi in alive_aligned] if n_alive_aligned > 0 else []
    rho_out_values = [round(rho_out[fi_to_ai[fi]].item(), 4) for fi in alive_aligned] if n_alive_aligned > 0 else []

    result = {
        "layer_idx": layer_idx,
        "n_features": n_features,
        "n_alive": n_alive,
        "n_aligned": n_aligned,
        "n_alive_aligned": n_alive_aligned,
        "geometric": {
            "aligned_pct": round(n_aligned / n_features * 100, 2),
            "unique_tokens_covered": unique_tokens,
            "coverage_pct": round(unique_tokens / vocab_size * 100, 2),
            "max_sim_mean": round(max_sims.mean().item(), 4),
            "max_sim_median": round(max_sims.median().item(), 4),
        },
        "input_correlation": {
            "mean_rho": round(rho_in_aa.mean().item(), 4) if n_alive_aligned > 0 else 0.0,
            "median_rho": round(rho_in_aa.median().item(), 4) if n_alive_aligned > 0 else 0.0,
            "rho_values": rho_in_values,
        },
        "output_correlation": {
            "mean_rho": round(rho_out_aa.mean().item(), 4) if n_alive_aligned > 0 else 0.0,
            "median_rho": round(rho_out_aa.median().item(), 4) if n_alive_aligned > 0 else 0.0,
            "rho_values": rho_out_values,
        },
        "n_categorized": len(categories),
        "categories": cat_counts,
        "examples": examples,
        "total_positions": total_positions,
    }

    if geo_baseline is not None:
        bl_sims = geo_baseline.max_sims
        bl_aligned_n = (bl_sims >= 0.8).sum().item()
        result["geometric_baseline"] = {
            "aligned_pct": round(bl_aligned_n / n_features * 100, 2),
            "max_sim_mean": round(bl_sims.mean().item(), 4),
            "max_sim_median": round(bl_sims.median().item(), 4),
        }

    return result, max_sims, geo_baseline.max_sims if geo_baseline else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Discover or parse checkpoints
    if args.sae_paths:
        checkpoints = parse_sae_paths(args.sae_paths)
        baseline_checkpoints = {}
    elif args.results_dir:
        checkpoints = discover_checkpoints(
            args.results_dir, args.model_name, args.variant
        )
        baseline_checkpoints = discover_checkpoints(
            args.results_dir, args.model_name, args.baseline_variant
        )
    else:
        log.error("Either --results-dir or --sae-paths must be provided")
        return

    if not checkpoints:
        log.error("No checkpoints found")
        return

    if args.layer_idx is not None:
        if args.layer_idx not in checkpoints:
            log.error(
                "Layer %d not found. Available: %s", args.layer_idx, sorted(checkpoints)
            )
            return
        checkpoints = {args.layer_idx: checkpoints[args.layer_idx]}

    log.info("Found %d checkpoints: layers %s", len(checkpoints), sorted(checkpoints))

    # Load LM
    log.info("Loading %s...", args.model_name)
    lm_model, tokenizer = load_model(args.model_name, device=device)
    nn_model = NNsight(lm_model)
    W_E = get_embedding(lm_model).weight.data
    log.info("  vocab_size=%d", W_E.shape[0])

    # Load dataset
    log.info("Loading dataset: %s/%s", args.dataset, args.dataset_config)
    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.dataset_config, split="train")

    # Analyze each layer
    for layer_idx in sorted(checkpoints):
        sae_path = checkpoints[layer_idx]
        baseline_path = baseline_checkpoints.get(layer_idx)

        result, max_sims, baseline_max_sims = analyze_layer(
            layer_idx,
            sae_path,
            baseline_path,
            lm_model,
            nn_model,
            tokenizer,
            W_E,
            ds,
            args,
            device,
        )

        # Save results
        layer_dir = output_dir / f"L{layer_idx}"
        layer_dir.mkdir(exist_ok=True)

        with open(layer_dir / "results.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        pt_data = {"max_sims": max_sims}
        if baseline_max_sims is not None:
            pt_data["baseline_max_sims"] = baseline_max_sims
        torch.save(pt_data, layer_dir / "max_sims.pt")

        log.info("  Saved results to %s", layer_dir)

    log.info("Done.")


if __name__ == "__main__":
    main()
