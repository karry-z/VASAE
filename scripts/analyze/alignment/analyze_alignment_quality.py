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

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Feature alignment quality analysis (F002)")
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

    Per-feature count: each feature's effective sample size is tracked
    separately via `count_per_feat`, which enables masked/active-only
    accumulation (e.g. only positions where z > 0).
    """

    def __init__(self, n_features: int):
        self.sum_z = torch.zeros(n_features, dtype=torch.float64)
        self.sum_u = torch.zeros(n_features, dtype=torch.float64)
        self.sum_zu = torch.zeros(n_features, dtype=torch.float64)
        self.sum_z2 = torch.zeros(n_features, dtype=torch.float64)
        self.sum_u2 = torch.zeros(n_features, dtype=torch.float64)
        self.count_per_feat = torch.zeros(n_features, dtype=torch.float64)

    def update(self, z: torch.Tensor, u: torch.Tensor, mask: torch.Tensor):
        """Update with (N, d_a) tensors; mask (N, d_a) is 0/1 double."""
        mask_d = mask.double()
        z_m = z.double() * mask_d
        u_m = u.double() * mask_d
        self.sum_z += z_m.sum(dim=0)
        self.sum_u += u_m.sum(dim=0)
        self.sum_zu += (z_m * u_m).sum(dim=0)
        self.sum_z2 += (z_m * z_m).sum(dim=0)
        self.sum_u2 += (u_m * u_m).sum(dim=0)
        self.count_per_feat += mask_d.sum(dim=0)

    def compute(self) -> torch.Tensor:
        """Compute Pearson correlation per feature. Returns (d_a,)."""
        n = self.count_per_feat
        num = n * self.sum_zu - self.sum_z * self.sum_u
        den = torch.sqrt(
            (n * self.sum_z2 - self.sum_z**2) * (n * self.sum_u2 - self.sum_u**2)
        )
        rho = num / den.clamp(min=1e-12)
        rho = torch.where(n >= 2, rho, torch.zeros_like(rho))
        return rho.float()


# ---------------------------------------------------------------------------
# Firing card collector: per-feature top-K firing positions with context window
# ---------------------------------------------------------------------------


class FiringCardCollector:
    """For each feature, maintain the top-K firing events by z value.

    Each event stores a small context window around the firing position
    so downstream code can decode and display it. No z>0 filtering here —
    non-positive z records are simply dominated by any active record and
    evicted as the heap fills.
    """

    def __init__(self, n_features: int, top_k: int = 10, context_window: int = 5):
        self.n_features = n_features
        self.top_k = top_k
        self.context_window = context_window
        # One min-heap per feature; each entry is
        # (z_float, tiebreaker, context_token_ids_list, pos_in_context_int)
        self._heaps: list[list] = [[] for _ in range(n_features)]
        self._counter = 0

    @torch.no_grad()
    def update(
        self,
        z_flat: torch.Tensor,
        valid_orig_idx: torch.Tensor,
        input_ids_cpu: torch.Tensor,
    ):
        """Update per-feature top-K heaps with this batch's firings.

        z_flat:          (N_valid, d_a) on CPU
        valid_orig_idx:  (N_valid,) indices into (B*S) space, CPU long
        input_ids_cpu:   (B, S) CPU long
        """
        import heapq

        if z_flat.shape[0] == 0:
            return
        k_local = min(self.top_k, z_flat.shape[0])
        # For each feature, batch top-K with indices into N_valid rows.
        batch_top_z, batch_top_row = torch.topk(
            z_flat.t(), k=k_local, dim=1
        )  # (d_a, K)

        # When using demeaned z, values can be negative; skip features whose
        # best value in this batch is -inf (all-padding edge case).
        active_f = torch.isfinite(batch_top_z[:, 0]).nonzero(as_tuple=True)[0].tolist()
        if not active_f:
            return

        S = input_ids_cpu.shape[1]
        W = self.context_window
        K = self.top_k
        valid_orig_idx_list = valid_orig_idx.tolist()
        batch_top_z_list = batch_top_z.tolist()
        batch_top_row_list = batch_top_row.tolist()

        for f_idx in active_f:
            heap = self._heaps[f_idx]
            z_row = batch_top_z_list[f_idx]
            row_row = batch_top_row_list[f_idx]
            for k_i in range(k_local):
                z_val = z_row[k_i]
                if z_val == float("-inf"):
                    break  # remaining are also -inf (topk is descending)
                # If heap is full and this z is not larger than current min, skip fast.
                if len(heap) == K and z_val <= heap[0][0]:
                    break
                orig = valid_orig_idx_list[row_row[k_i]]
                b = orig // S
                p = orig % S
                lo = max(0, p - W)
                hi = min(S, p + W + 1)
                context_ids = input_ids_cpu[b, lo:hi].tolist()
                pos_in_ctx = p - lo
                self._counter += 1
                entry = (z_val, self._counter, context_ids, pos_in_ctx)
                if len(heap) < K:
                    heapq.heappush(heap, entry)
                else:
                    heapq.heappushpop(heap, entry)

    def finalize(self, tokenizer) -> list[list[dict]]:
        """Decode heap entries to per-feature lists of firing records, sorted desc by z."""
        out = []
        for heap in self._heaps:
            entries = sorted(heap, key=lambda e: -e[0])
            records = []
            for z_val, _tb, ctx_ids, pos in entries:
                tokens = [tokenizer.decode([tid]) for tid in ctx_ids]
                records.append(
                    {
                        "z": round(float(z_val), 4),
                        "context_tokens": tokens,
                        "fire_pos": pos,
                    }
                )
            out.append(records)
        return out


# ---------------------------------------------------------------------------
# Correlation-based input/output relevance
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_correlation(
    sae: SAEModel,
    lm_model: torch.nn.Module,
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
    cards = FiringCardCollector(n_aligned, top_k=10, context_window=5)
    feature_ever_active = torch.zeros(n_features, dtype=torch.bool)
    total_positions = 0

    n_batches = (n_samples + batch_size - 1) // batch_size

    for batch_idx, batch_start in enumerate(range(0, n_samples, batch_size)):
        if batch_idx % 50 == 0:
            logger.info("  Correlation: batch %d/%d", batch_idx, n_batches)

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
        feature_ever_active |= (
            (z.reshape(-1, n_features)[mask_flat] > 0).any(dim=0).cpu()
        )
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

        # --- Per-sentence demean over valid positions (sentence fixed-effect) ---
        valid_f = mask.float().unsqueeze(-1)  # (B, S, 1)
        n_per_sent = valid_f.sum(dim=1).clamp(min=1)  # (B, 1)
        z_sm = (z_aligned * valid_f).sum(dim=1) / n_per_sent  # (B, d_a)
        uin_sm = (u_in * valid_f).sum(dim=1) / n_per_sent
        uout_sm = (u_out * valid_f).sum(dim=1) / n_per_sent
        z_dm = z_aligned - z_sm.unsqueeze(1)
        uin_dm = u_in - uin_sm.unsqueeze(1)
        uout_dm = u_out - uout_sm.unsqueeze(1)

        # --- Accumulate only valid positions ---
        # Flatten (B, S) → (N,) keeping only mask=True
        valid = mask.reshape(-1)  # (B*S,)
        z_flat = z_aligned.reshape(-1, n_aligned)[valid].cpu()  # raw, for active mask
        z_dm_flat = z_dm.reshape(-1, n_aligned)[valid].cpu()
        uin_dm_flat = uin_dm.reshape(-1, n_aligned)[valid].cpu()
        uout_dm_flat = uout_dm.reshape(-1, n_aligned)[valid].cpu()

        # Active mask: only positions where the feature actually fires (z > 0)
        active_mask = (z_flat > 0).double()  # (N_valid, d_a)

        acc_in.update(z_dm_flat, uin_dm_flat, mask=active_mask)
        acc_out.update(z_dm_flat, uout_dm_flat, mask=active_mask)

        # Per-feature top-K firing records (for qualitative inspection).
        # Use demeaned z so that always-on features are suppressed and
        # position-specific activations stand out (same logic as casestudy
        # relative mode).  Mask out positions where the raw z == 0 (feature
        # did not fire) so they never enter the top-K heap.
        z_dm_for_cards = z_dm_flat.clone()
        z_dm_for_cards[z_flat <= 0] = float("-inf")
        valid_orig_idx = valid.nonzero(as_tuple=True)[0].cpu()
        cards.update(z_dm_for_cards, valid_orig_idx, input_ids.cpu())

        del (
            h,
            z,
            z_aligned,
            logits,
            probs,
            topm_embs,
            h_input,
            h_output,
            u_in,
            u_out,
            enc,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rho_in = acc_in.compute()  # (d_a,)
    rho_out = acc_out.compute()  # (d_a,)
    firing_cards = cards.finalize(tokenizer)

    return {
        "rho_in": rho_in,
        "rho_out": rho_out,
        "n_active_per_feat": acc_in.count_per_feat.clone(),
        "firing_cards": firing_cards,
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
            key=lambda fi: max(
                rho_in[fi_to_ai[fi]].item(), rho_out[fi_to_ai[fi]].item()
            ),
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
    lm_model: torch.nn.Module,
    nn_model: NNsight,
    tokenizer,
    W_E,
    dataset,
    args,
    device,
):
    """Run full analysis for a single layer."""
    logger.info("=== Layer %d ===", layer_idx)

    sae = load_sae_for_analysis(sae_path, device=device)
    n_features = sae.config.dim_sparse
    logger.info("  SAE: n_features=%d, k=%s", n_features, sae.config.k)

    # Step 1: Geometric alignment
    logger.info("  Step 1: Geometric alignment")
    geo = compute_geometric_alignment(
        get_decoder_features(sae), W_E, top_k=1, device=device
    )
    max_sims = geo.max_sims
    top1_tokens = geo.topk_indices[:, 0]

    aligned_mask = max_sims >= 0.8  # mask
    aligned_mask_indices = aligned_mask.nonzero(as_tuple=True)[
        0
    ].tolist()  # still the mask
    n_aligned = len(aligned_mask_indices)
    unique_tokens = top1_tokens[aligned_mask].unique().numel() if n_aligned > 0 else 0
    vocab_size = W_E.shape[0]

    logger.info(
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
        logger.info("  Computing baseline geometric alignment")
        baseline_sae = load_sae_for_analysis(baseline_sae_path, device=device)
        geo_baseline = compute_geometric_alignment(
            get_decoder_features(baseline_sae), W_E, top_k=1, device=device
        )
        bl_aligned = (geo_baseline.max_sims >= 0.8).sum().item()
        logger.info(
            "  Baseline aligned: %d/%d (%.1f%%)",
            bl_aligned,
            n_features,
            bl_aligned / n_features * 100,
        )
        del baseline_sae

    # Step 2: Input/output correlation
    if n_aligned == 0:
        logger.info("  No aligned features — skipping correlation analysis")
        rho_in = torch.zeros(0)
        rho_out = torch.zeros(0)
        n_active_per_feat = torch.zeros(0, dtype=torch.float64)
        firing_cards_raw: list[list[dict]] = []
        alive_mask = torch.zeros(n_features, dtype=torch.bool)
        total_positions = 0
    else:
        aligned_token_ids = top1_tokens[aligned_mask]
        logger.info(
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
            aligned_mask_indices,
            aligned_token_ids,
            args.n_samples,
            args.batch_size,
            args.max_length,
            args.top_m,
            device,
        )
        rho_in = corr_result["rho_in"]
        rho_out = corr_result["rho_out"]
        n_active_per_feat = corr_result["n_active_per_feat"]
        firing_cards_raw = corr_result["firing_cards"]
        alive_mask = corr_result["alive_mask"]
        total_positions = corr_result["total_positions"]

    n_alive = alive_mask.sum().item()
    alive_aligned = [fi for fi in aligned_mask_indices if alive_mask[fi]]
    n_alive_aligned = len(alive_aligned)
    logger.info(
        "  Alive: %d/%d, Alive+Aligned: %d/%d",
        n_alive,
        n_features,
        n_alive_aligned,
        n_aligned,
    )

    # Correlation stats for alive+aligned
    if n_alive_aligned > 0 and n_aligned > 0:
        # Map alive+aligned to their indices in rho arrays
        fi_to_ai = {fi: ai for ai, fi in enumerate(aligned_mask_indices)}
        alive_aligned_ai = [fi_to_ai[fi] for fi in alive_aligned]
        rho_in_aa = rho_in[alive_aligned_ai]
        rho_out_aa = rho_out[alive_aligned_ai]
        n_active_aa = n_active_per_feat[alive_aligned_ai]
        n_active_median = int(n_active_aa.median().item())
        logger.info(
            "  Input corr: mean=%.4f, median=%.4f",
            rho_in_aa.mean().item(),
            rho_in_aa.median().item(),
        )
        logger.info(
            "  Output corr: mean=%.4f, median=%.4f",
            rho_out_aa.mean().item(),
            rho_out_aa.median().item(),
        )
        logger.info(
            "  n_active_per_feat (alive+aligned): min=%d, median=%d, max=%d",
            int(n_active_aa.min().item()),
            n_active_median,
            int(n_active_aa.max().item()),
        )
    else:
        rho_in_aa = torch.zeros(0)
        rho_out_aa = torch.zeros(0)
        n_active_median = 0

    # Step 3: Categorize alive+aligned features
    if n_alive_aligned > 0:
        categories = categorize_features(aligned_mask_indices, rho_in, rho_out)
        # Filter to alive+aligned only
        categories = {fi: categories[fi] for fi in alive_aligned if fi in categories}
    else:
        categories = {}

    cat_counts = {}
    for cat in ["dual", "input_related", "output_related", "non_functional"]:
        cat_counts[cat] = sum(1 for c in categories.values() if c == cat)
    logger.info(
        "  Categories (of %d alive+aligned): %s",
        n_alive_aligned,
        json.dumps(cat_counts),
    )

    # Build examples
    examples = build_examples(
        categories, geo, rho_in, rho_out, aligned_mask_indices, tokenizer, layer_idx
    )

    # Build result dict
    rho_in_values = (
        [round(rho_in[fi_to_ai[fi]].item(), 4) for fi in alive_aligned]
        if n_alive_aligned > 0
        else []
    )
    rho_out_values = (
        [round(rho_out[fi_to_ai[fi]].item(), 4) for fi in alive_aligned]
        if n_alive_aligned > 0
        else []
    )

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
            "mean_rho": (
                round(rho_in_aa.mean().item(), 4) if n_alive_aligned > 0 else 0.0
            ),
            "median_rho": (
                round(rho_in_aa.median().item(), 4) if n_alive_aligned > 0 else 0.0
            ),
            "n_active_median": n_active_median,
            "rho_values": rho_in_values,
        },
        "output_correlation": {
            "mean_rho": (
                round(rho_out_aa.mean().item(), 4) if n_alive_aligned > 0 else 0.0
            ),
            "median_rho": (
                round(rho_out_aa.median().item(), 4) if n_alive_aligned > 0 else 0.0
            ),
            "n_active_median": n_active_median,
            "rho_values": rho_out_values,
        },
        "n_categorized": len(categories),
        "categories": cat_counts,
        "examples": examples,
        "total_positions": total_positions,
    }

    # Per-feature firing cards for alive+aligned features (qualitative inspection).
    if n_alive_aligned > 0 and firing_cards_raw:
        firing_cards_out = []
        for fi in alive_aligned:
            ai = fi_to_ai[fi]
            t_i = geo.topk_indices[fi, 0].item()
            firing_cards_out.append(
                {
                    "feature_id": fi,
                    "aligned_token": tokenizer.decode([t_i]),
                    "aligned_token_id": t_i,
                    "geo_max_sim": round(max_sims[fi].item(), 4),
                    "n_active": int(n_active_per_feat[ai].item()),
                    "rho_in": round(rho_in[ai].item(), 4),
                    "rho_out": round(rho_out[ai].item(), 4),
                    "category": categories.get(fi, "non_functional"),
                    "top_firings": firing_cards_raw[ai],
                }
            )
        result["firing_cards"] = firing_cards_out

    if geo_baseline is not None:
        bl_sims = geo_baseline.max_sims
        bl_aligned_n = (bl_sims >= 0.8).sum().item()
        result["geometric_baseline"] = {
            "aligned_pct": round(bl_aligned_n / n_features * 100, 2),
            "max_sim_mean": round(bl_sims.mean().item(), 4),
            "max_sim_median": round(bl_sims.median().item(), 4),
        }

    return result, max_sims, geo_baseline.max_sims if geo_baseline else None


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
        logger.error("Either --results-dir or --sae-paths must be provided")
        return

    if not checkpoints:
        logger.error("No checkpoints found")
        return

    if args.layer_idx is not None:
        if args.layer_idx not in checkpoints:
            logger.error(
                "Layer %d not found. Available: %s", args.layer_idx, sorted(checkpoints)
            )
            return
        checkpoints = {args.layer_idx: checkpoints[args.layer_idx]}

    logger.info(
        "Found %d checkpoints: layers %s", len(checkpoints), sorted(checkpoints)
    )

    # Load LM
    logger.info("Loading %s...", args.model_name)
    lm_model, tokenizer = load_model(args.model_name, device=device)
    nn_model = NNsight(lm_model)
    W_E = get_embedding(lm_model).weight.data
    logger.info("  vocab_size=%d", W_E.shape[0])

    # Load dataset
    logger.info("Loading dataset: %s/%s", args.dataset, args.dataset_config)
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

        logger.info("  Saved results to %s", layer_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
