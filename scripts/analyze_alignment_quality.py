"""Feature alignment quality analysis for 002_F.

For each SAE layer checkpoint, compute:
  1. Geometric alignment: cosine similarity between decoder features and token embeddings
  2. Input detection: Precision@K — fraction of top-K activations with aligned token in context
  3. Output control: zero-ablation top-1/top-5 match rate

Then categorize features into dual / input_detector / output_controller / non_functional.

Usage (single layer, via Slurm array):
    uv run python scripts/analyze_alignment_quality.py \
        --results-dir /scratch/b5bq/pu22650.b5bq/VASAE_out/001_F_Benchmarking \
        --model-name gpt2 \
        --variant soft \
        --layer-idx 6 \
        --output-dir exp/002_F_AlignmentAnalysis/gpt2 \
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
from vasae.models.sae import SAEModel
from vasae.models.factory import load_model, get_embedding, get_layers
from vasae.engine.intervention import _get_layer_proxy

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Feature alignment quality analysis (002_F)")
    p.add_argument("--results-dir", type=str, required=True,
                   help="Directory containing SAE checkpoints (e.g. 001_F_Benchmarking)")
    p.add_argument("--model-name", type=str, required=True,
                   help="HuggingFace model name (e.g. gpt2, meta-llama/Llama-3.1-8B)")
    p.add_argument("--variant", type=str, default="soft",
                   help="Checkpoint variant to analyze")
    p.add_argument("--baseline-variant", type=str, default="plain",
                   help="Baseline variant for geometric comparison")
    p.add_argument("--layer-idx", type=int, default=None,
                   help="Single layer to analyze (default: all layers)")
    p.add_argument("--dataset", type=str, default="wikitext")
    p.add_argument("--dataset-config", type=str, default="wikitext-103-raw-v1")
    p.add_argument("--n-samples", type=int, default=5000,
                   help="Number of text samples for input detection")
    p.add_argument("--n-causal-samples", type=int, default=500,
                   help="Number of samples for output control (ablation)")
    p.add_argument("--n-causal-features", type=int, default=600,
                   help="Max number of aligned features to ablate (0=all)")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--top-k-positions", type=int, default=100,
                   help="K: number of top activation positions for input detection")
    p.add_argument("--ctx-window", type=int, default=32,
                   help="w: context window size for input detection")
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


# ---------------------------------------------------------------------------
# Step 1: Geometric alignment
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_geometric(sae: SAEModel, W_E: torch.Tensor, device: torch.device):
    """Compute max cosine similarity and top-1 aligned token for all features."""
    D = sae.decoder.weight.data.T.to(device).float()  # (n_features, dim_input)
    E = W_E.to(device).float()
    n_features = D.shape[0]

    D_norm = F.normalize(D, dim=1)
    E_norm = F.normalize(E, dim=1)

    max_sims = torch.zeros(n_features, device=device)
    top1_tokens = torch.zeros(n_features, dtype=torch.long, device=device)

    chunk = 512
    for i in range(0, n_features, chunk):
        end = min(i + chunk, n_features)
        sim = D_norm[i:end] @ E_norm.T  # (chunk, vocab)
        ms, t1 = sim.max(dim=1)
        max_sims[i:end] = ms
        top1_tokens[i:end] = t1

    return {
        "max_sims": max_sims.cpu(),
        "top1_tokens": top1_tokens.cpu(),
    }


# ---------------------------------------------------------------------------
# Step 2: Input detection (Precision@K)
# ---------------------------------------------------------------------------

def _extract_acts(nn_model: NNsight, input_ids: torch.Tensor, layer_idx: int):
    """Extract (B, S, D) activations from a layer."""
    B, S = input_ids.shape
    with nn_model.trace(input_ids):
        layer = _get_layer_proxy(nn_model, layer_idx)
        h = layer.output.save()
    h = h.detach()
    # GPT-2 layers return a tuple (hidden_states, ...) while some return tensor directly
    if isinstance(h, tuple):
        h = h[0]
    if h.dim() == 2:
        h = h.reshape(B, S, -1)
    return h


def _build_context_windows(input_ids: torch.Tensor, w: int):
    """Build context windows for each position.

    For position j in sequence of length S, context = input_ids[max(0, j-w):j+1].
    Returns a (B*S, w+1) tensor, padded with -1 for positions near the start.
    """
    B, S = input_ids.shape
    # ctx[:, j, offset] = input_ids[:, j - offset] if j - offset >= 0, else -1
    ctx = torch.full((B, S, w + 1), -1, dtype=torch.long, device=input_ids.device)
    for offset in range(w + 1):
        if offset == 0:
            ctx[:, :, 0] = input_ids
        elif offset <= S:
            ctx[:, offset:, offset] = input_ids[:, :-offset]

    return ctx.reshape(B * S, w + 1)  # (B*S, w+1)


@torch.no_grad()
def compute_input_detection(sae: SAEModel, nn_model: NNsight, tokenizer,
                             layer_idx: int, dataset, geo_top1_tokens: torch.Tensor,
                             aligned_indices: list[int],
                             n_samples: int, batch_size: int, max_length: int,
                             top_k_pos: int, ctx_window: int,
                             device: torch.device):
    """Compute Precision@K for input detection.

    For each aligned feature, find its top-K strongest activation positions,
    then check what fraction have t(i) in the context window.
    """
    n_features = sae.config.dim_sparse
    n_aligned = len(aligned_indices)
    aligned_tensor = torch.tensor(aligned_indices, dtype=torch.long)

    # Online top-K tracking for aligned features only
    # topk_vals: (n_aligned, K), topk_ctx: (n_aligned, K, w+1)
    K = top_k_pos
    W = ctx_window + 1
    topk_vals = torch.full((n_aligned, K), -float("inf"))
    topk_ctx = torch.full((n_aligned, K, W), -1, dtype=torch.long)

    # Also track alive status for all features
    feature_ever_active = torch.zeros(n_features, dtype=torch.bool)

    n_batches = (n_samples + batch_size - 1) // batch_size
    total_positions = 0

    for batch_idx, batch_start in enumerate(range(0, n_samples, batch_size)):
        if batch_idx % 50 == 0:
            log.info("  Input detection: batch %d/%d", batch_idx, n_batches)

        batch_end = min(batch_start + batch_size, n_samples)
        batch_texts = [dataset[i]["text"] for i in range(batch_start, batch_end)]
        batch_texts = [t for t in batch_texts if t.strip()]
        if not batch_texts:
            continue

        enc = tokenizer(batch_texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(device)
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]

        h = _extract_acts(nn_model, input_ids, layer_idx)
        B, S, _ = h.shape
        h_flat = h.reshape(-1, h.shape[-1]).float()  # (B*S, D)
        _, z = sae.encode(h_flat)  # (B*S, n_features)

        mask_flat = attn_mask.reshape(-1).bool()  # (B*S,)
        # Track alive features
        feature_ever_active |= (z[mask_flat] > 0).any(dim=0).cpu()
        z[~mask_flat] = -float("inf")  # exclude padded positions from top-K
        total_positions += mask_flat.sum().item()

        # Build context windows for this batch
        ctx_batch = _build_context_windows(input_ids, ctx_window)  # (B*S, w+1)
        ctx_batch_cpu = ctx_batch.cpu()
        BS = B * S

        # Extract aligned feature activations: (n_aligned, B*S)
        z_aligned = z.cpu()[:, aligned_tensor].T  # (n_aligned, B*S)

        # Batch top-K update: combine old top-K vals with new batch vals
        # combined: (n_aligned, K + B*S)
        combined = torch.cat([topk_vals, z_aligned], dim=1)
        new_vals, new_idx = combined.topk(K, dim=1, sorted=False)
        topk_vals = new_vals

        # Update context windows: new_idx values < K reference old topk_ctx,
        # values >= K reference ctx_batch_cpu at position (new_idx - K).
        # Process in small chunks to bound memory for the gather operation.
        feat_chunk = 64
        for ai_start in range(0, n_aligned, feat_chunk):
            ai_end = min(ai_start + feat_chunk, n_aligned)
            chunk_size = ai_end - ai_start
            chunk_idx = new_idx[ai_start:ai_end]  # (chunk, K)

            # Build source: (chunk, K+B*S, W)
            old_ctx = topk_ctx[ai_start:ai_end]  # (chunk, K, W)
            batch_ctx_exp = ctx_batch_cpu.unsqueeze(0).expand(chunk_size, -1, -1)
            all_ctx = torch.cat([old_ctx, batch_ctx_exp], dim=1)

            idx_exp = chunk_idx.unsqueeze(-1).expand(-1, -1, W)
            topk_ctx[ai_start:ai_end] = all_ctx.gather(1, idx_exp)

        del h, z, z_aligned, enc, ctx_batch, ctx_batch_cpu
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Compute P_i for aligned features
    P = torch.zeros(n_features)
    for ai, fi in enumerate(aligned_indices):
        t_i = geo_top1_tokens[fi].item()
        contexts = topk_ctx[ai]  # (K, W)
        hits = (contexts == t_i).any(dim=1).sum().item()
        P[fi] = hits / K

    return {
        "P": P,
        "alive_mask": feature_ever_active,
        "total_positions": total_positions,
    }


# ---------------------------------------------------------------------------
# Step 3: Output control (zero ablation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_output_control(sae: SAEModel, lm_model, nn_model: NNsight,
                            tokenizer, layer_idx: int, dataset,
                            aligned_features: list[int],
                            geo_top1_tokens: torch.Tensor,
                            n_samples: int, batch_size: int,
                            max_length: int, device: torch.device):
    """Compute output control via zero ablation.

    For each aligned feature, ablate z_i * d_i from the residual stream and
    check if t(i) is the token with the largest logit decrease.
    """
    vocab_size = lm_model.config.vocab_size
    layers = get_layers(lm_model)
    target_layer = layers[layer_idx]

    # Accumulate signed logit delta per feature
    delta_accum = {fi: torch.zeros(vocab_size, dtype=torch.float64)
                   for fi in aligned_features}
    delta_count = {fi: 0 for fi in aligned_features}

    n_batches = (n_samples + batch_size - 1) // batch_size

    for batch_idx, batch_start in enumerate(range(0, n_samples, batch_size)):
        if batch_idx % 5 == 0:
            log.info("  Output control: batch %d/%d", batch_idx, n_batches)

        batch_end = min(batch_start + batch_size, n_samples)
        batch_texts = [dataset[i]["text"] for i in range(batch_start, batch_end)]
        batch_texts = [t for t in batch_texts if t.strip()]
        if not batch_texts:
            continue

        enc = tokenizer(batch_texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(device)
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]

        # Get clean activations and SAE encoding
        h_clean = _extract_acts(nn_model, input_ids, layer_idx)
        B, S, D = h_clean.shape
        _, z = sae.encode(h_clean.reshape(-1, D).float())
        z = z.reshape(B, S, -1)

        # Get clean logits
        clean_out = lm_model(input_ids=input_ids, attention_mask=attn_mask)
        clean_logits = clean_out.logits.detach()

        mask = attn_mask.bool()
        n_valid = mask.sum().item()
        if n_valid == 0:
            continue

        # Ablate each aligned feature
        for fi in aligned_features:
            z_fi = z[:, :, fi]  # (B, S)
            if (z_fi.abs() < 1e-8).all():
                continue

            d_fi = sae.decoder.weight.data[:, fi].to(device)  # (dim_input,)
            delta = z_fi.unsqueeze(-1) * d_fi.unsqueeze(0).unsqueeze(0)  # (B, S, D)

            def hook_fn(module, input, output, _delta=delta):
                if isinstance(output, tuple):
                    h = output[0]
                    return (h - _delta.to(h.dtype),) + output[1:]
                return output - _delta.to(output.dtype)

            handle = target_layer.register_forward_hook(hook_fn)
            try:
                ablated_out = lm_model(input_ids=input_ids, attention_mask=attn_mask)
                ablated_logits = ablated_out.logits.detach()
            finally:
                handle.remove()

            # Signed delta: positive means clean logit was higher (feature promoted this token)
            signed_delta = clean_logits - ablated_logits  # (B, S, V)
            delta_masked = signed_delta * mask.unsqueeze(-1).float()
            mean_delta = delta_masked.sum(dim=(0, 1)) / n_valid
            delta_accum[fi] += mean_delta.cpu().double()
            delta_count[fi] += 1

            del signed_delta, delta_masked, mean_delta, ablated_logits, delta

        del h_clean, z, clean_logits, enc
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Compute top-1 and top-5 match for each feature
    results = {}
    for fi in aligned_features:
        if delta_count[fi] == 0:
            results[fi] = {
                "top1_match": False, "top5_match": False,
                "top5_tokens": [], "delta_at_ti": 0.0,
            }
            continue
        mean_delta = delta_accum[fi] / delta_count[fi]
        top5_vals, top5_idx = mean_delta.topk(5)
        top5_tokens = top5_idx.tolist()
        t_i = geo_top1_tokens[fi].item()
        results[fi] = {
            "top1_match": top5_tokens[0] == t_i,
            "top5_match": t_i in top5_tokens,
            "top5_tokens": top5_tokens,
            "top5_vals": [round(v, 4) for v in top5_vals.tolist()],
            "delta_at_ti": round(mean_delta[t_i].item(), 4),
        }

    return results


# ---------------------------------------------------------------------------
# Categorization
# ---------------------------------------------------------------------------

def categorize_features(aligned_indices: list[int], P: torch.Tensor,
                         output_results: dict):
    """Categorize aligned features into 4 types based on input detection and output control."""
    categories = {}
    for fi in aligned_indices:
        is_input = P[fi].item() >= 0.5
        is_output = output_results.get(fi, {}).get("top1_match", False)

        if is_input and is_output:
            categories[fi] = "dual"
        elif is_input:
            categories[fi] = "input_detector"
        elif is_output:
            categories[fi] = "output_controller"
        else:
            categories[fi] = "non_functional"

    return categories


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

def build_examples(categories: dict, geo_result: dict, P: torch.Tensor,
                    output_results: dict, tokenizer, layer_idx: int):
    """Pick 2-3 representative features per category for case study."""
    from collections import defaultdict

    by_cat = defaultdict(list)
    for fi, cat in categories.items():
        by_cat[cat].append(fi)

    examples = []
    for cat_name in ["dual", "input_detector", "output_controller", "non_functional"]:
        feats = by_cat.get(cat_name, [])
        if not feats:
            continue
        # Sort by geo_max_sim descending
        feats_sorted = sorted(
            feats, key=lambda fi: geo_result["max_sims"][fi].item(), reverse=True
        )
        for fi in feats_sorted[:3]:
            t_i = geo_result["top1_tokens"][fi].item()
            entry = {
                "layer": layer_idx,
                "feature_id": fi,
                "category": cat_name,
                "geo_max_sim": round(geo_result["max_sims"][fi].item(), 4),
                "aligned_token": tokenizer.decode([t_i]),
                "aligned_token_id": t_i,
                "P_i": round(P[fi].item(), 4),
            }
            oc = output_results.get(fi, {})
            entry["top1_match"] = oc.get("top1_match", False)
            entry["top5_match"] = oc.get("top5_match", False)
            if oc.get("top5_tokens"):
                entry["ablation_top5"] = [
                    tokenizer.decode([tid]) for tid in oc["top5_tokens"]
                ]
            examples.append(entry)

    return examples


# ---------------------------------------------------------------------------
# Main per-layer analysis
# ---------------------------------------------------------------------------

def analyze_layer(layer_idx: int, sae_path: Path, baseline_sae_path: Path | None,
                  lm_model, nn_model: NNsight, tokenizer, W_E,
                  dataset, args, device):
    """Run full analysis for a single layer."""
    log.info("=== Layer %d ===", layer_idx)

    # Load SAE (drop unused lowrank params to save GPU memory)
    sae = SAEModel.from_pretrained(str(sae_path)).eval()
    if not sae.config.use_lowrank:
        del sae.decoder_lowrank, sae.learnable_lowrank_coeff
    sae = sae.to(device)
    n_features = sae.config.dim_sparse
    log.info("  SAE: n_features=%d, k=%s", n_features, sae.config.k)

    # Step 1: Geometric alignment
    log.info("  Step 1: Geometric alignment")
    geo = compute_geometric(sae, W_E, device)
    max_sims = geo["max_sims"]
    top1_tokens = geo["top1_tokens"]

    aligned_mask = max_sims >= 0.8
    aligned_indices = aligned_mask.nonzero(as_tuple=True)[0].tolist()
    n_aligned = len(aligned_indices)
    unique_tokens = top1_tokens[aligned_mask].unique().numel() if n_aligned > 0 else 0
    vocab_size = W_E.shape[0]

    log.info("  Aligned: %d/%d (%.1f%%), unique tokens: %d/%d",
             n_aligned, n_features, n_aligned / n_features * 100,
             unique_tokens, vocab_size)

    # Baseline geometric alignment
    geo_baseline = None
    if baseline_sae_path is not None and baseline_sae_path.exists():
        log.info("  Computing baseline geometric alignment")
        baseline_sae = SAEModel.from_pretrained(str(baseline_sae_path)).eval()
        # Drop unused lowrank params before moving to GPU (they are huge for large dim_sparse)
        del baseline_sae.decoder_lowrank, baseline_sae.learnable_lowrank_coeff
        baseline_sae = baseline_sae.to(device)
        geo_baseline = compute_geometric(baseline_sae, W_E, device)
        bl_aligned = (geo_baseline["max_sims"] >= 0.8).sum().item()
        log.info("  Baseline aligned: %d/%d (%.1f%%)",
                 bl_aligned, n_features, bl_aligned / n_features * 100)
        del baseline_sae

    # Step 2: Input detection
    log.info("  Step 2: Input detection (Precision@%d, w=%d)",
             args.top_k_positions, args.ctx_window)
    input_result = compute_input_detection(
        sae, nn_model, tokenizer, layer_idx, dataset,
        top1_tokens, aligned_indices,
        args.n_samples, args.batch_size, args.max_length,
        args.top_k_positions, args.ctx_window, device
    )
    P = input_result["P"]
    alive_mask = input_result["alive_mask"]
    n_alive = alive_mask.sum().item()

    # Only consider alive AND aligned features for functional analysis
    alive_aligned = [fi for fi in aligned_indices if alive_mask[fi]]
    n_alive_aligned = len(alive_aligned)
    log.info("  Alive: %d/%d, Alive+Aligned: %d/%d",
             n_alive, n_features, n_alive_aligned, n_aligned)

    detection_rate = 0.0
    if n_alive_aligned > 0:
        n_detected = sum(1 for fi in alive_aligned if P[fi].item() >= 0.5)
        detection_rate = n_detected / n_alive_aligned * 100
    log.info("  Input detection rate: %.1f%% (%d/%d alive+aligned)",
             detection_rate, n_detected if n_alive_aligned > 0 else 0, n_alive_aligned)

    # Step 3: Output control (sample from alive+aligned features)
    causal_features = alive_aligned
    if args.n_causal_features > 0 and len(alive_aligned) > args.n_causal_features:
        perm = torch.randperm(len(alive_aligned))[:args.n_causal_features]
        causal_features = [alive_aligned[i] for i in sorted(perm.tolist())]
        log.info("  Sampled %d/%d alive+aligned features for ablation",
                 len(causal_features), n_alive_aligned)

    log.info("  Step 3: Output control (ablation, %d features × %d samples)",
             len(causal_features), args.n_causal_samples)
    output_results = compute_output_control(
        sae, lm_model, nn_model, tokenizer, layer_idx, dataset,
        causal_features, top1_tokens,
        args.n_causal_samples, args.batch_size, args.max_length, device
    )

    n_ablated = len(causal_features)
    top1_matches = sum(1 for fi in causal_features
                       if output_results.get(fi, {}).get("top1_match", False))
    top5_matches = sum(1 for fi in causal_features
                       if output_results.get(fi, {}).get("top5_match", False))
    top1_rate = top1_matches / max(n_ablated, 1) * 100
    top5_rate = top5_matches / max(n_ablated, 1) * 100
    log.info("  Output control: top-1 %.1f%%, top-5 %.1f%% (of %d ablated)",
             top1_rate, top5_rate, n_ablated)

    # Step 4: Categorize (only features that were tested for output control)
    # Features not in causal_features have no output control data, so we only
    # categorize those that were fully tested (both input detection + output control).
    categories = categorize_features(causal_features, P, output_results)
    cat_counts = {}
    for cat in ["dual", "input_detector", "output_controller", "non_functional"]:
        cat_counts[cat] = sum(1 for c in categories.values() if c == cat)
    n_categorized = len(causal_features)
    log.info("  Categories (of %d tested): %s", n_categorized, json.dumps(cat_counts))

    # Build examples
    examples = build_examples(categories, geo, P, output_results, tokenizer, layer_idx)

    # Collect P values for alive+aligned features (for distribution plot)
    P_values = [round(P[fi].item(), 4) for fi in alive_aligned]

    # Build result dict
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
        "input_detection": {
            "detection_rate": round(detection_rate, 2),
            "P_values": P_values,
        },
        "output_control": {
            "n_ablated": n_ablated,
            "top1_match_rate": round(top1_rate, 2),
            "top5_match_rate": round(top5_rate, 2),
        },
        "n_categorized": n_categorized,
        "categories": cat_counts,
        "examples": examples,
        "total_positions": input_result["total_positions"],
    }

    if geo_baseline is not None:
        bl_sims = geo_baseline["max_sims"]
        bl_aligned_n = (bl_sims >= 0.8).sum().item()
        result["geometric_baseline"] = {
            "aligned_pct": round(bl_aligned_n / n_features * 100, 2),
            "max_sim_mean": round(bl_sims.mean().item(), 4),
            "max_sim_median": round(bl_sims.median().item(), 4),
        }

    return result, max_sims, geo_baseline["max_sims"] if geo_baseline else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Discover checkpoints
    checkpoints = discover_checkpoints(args.results_dir, args.model_name, args.variant)
    if not checkpoints:
        log.error("No checkpoints found in %s for model=%s variant=%s",
                  args.results_dir, args.model_name, args.variant)
        return

    baseline_checkpoints = discover_checkpoints(
        args.results_dir, args.model_name, args.baseline_variant
    )

    if args.layer_idx is not None:
        if args.layer_idx not in checkpoints:
            log.error("Layer %d not found. Available: %s",
                      args.layer_idx, sorted(checkpoints))
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
            layer_idx, sae_path, baseline_path,
            lm_model, nn_model, tokenizer, W_E, ds, args, device
        )

        # Save results
        layer_dir = output_dir / f"L{layer_idx}"
        layer_dir.mkdir(exist_ok=True)

        # JSON: summary + examples (no huge arrays)
        with open(layer_dir / "results.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # PT: full max_sims arrays for histogram plotting
        pt_data = {"max_sims": max_sims}
        if baseline_max_sims is not None:
            pt_data["baseline_max_sims"] = baseline_max_sims
        torch.save(pt_data, layer_dir / "max_sims.pt")

        log.info("  Saved results to %s", layer_dir)

    log.info("Done.")


if __name__ == "__main__":
    main()
