"""Input-Output decomposition analysis for SAE features.

For each feature, compute:
  - t_geo:    token whose embedding is closest to decoder column (geometric alignment)
  - t_logit:  token most promoted by the feature's logit attribution (W_U @ d_i)
  - t_input:  token whose presence in context maximally activates the feature (empirical)
  - t_causal: token whose probability changes most when the feature is ablated (causal)

Measures consistency across these four views to understand whether
geometric alignment corresponds to functional alignment.

Usage:
    python scripts/analyze_feature_io_decomposition.py \
        --sae-path /path/to/010_soft_gpt2_L6_k32_a1e-4 \
        --layer-idx 6 \
        --output-dir exp/011_p_IODecomposition/L6_k32
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

import torch
import torch.nn.functional as F
from nnsight import NNsight

from vasae.models.sae import SAEModel
from vasae.models.factory import load_model, get_embedding, get_lm_head, get_layers
from vasae.engine.intervention import _get_layer_proxy


def extract_acts(nn_model: NNsight, input_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """Extract full (B, S, D) activations from a layer."""
    with nn_model.trace(input_ids):
        layer = _get_layer_proxy(nn_model, layer_idx)
        h = layer.output.save()
    return h.detach()


def parse_args():
    p = argparse.ArgumentParser(description="Feature I/O decomposition analysis")
    p.add_argument("--sae-path", type=str, required=True,
                   help="Path to trained SAE (HF format directory)")
    p.add_argument("--model-name", type=str, default="gpt2")
    p.add_argument("--layer-idx", type=int, required=True)
    p.add_argument("--dataset", type=str, default="wikitext")
    p.add_argument("--dataset-config", type=str, default="wikitext-103-raw-v1")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--n-samples", type=int, default=500,
                   help="Number of text samples for input/causal analysis")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--top-k", type=int, default=5,
                   help="Top-k tokens to record per feature per view")
    p.add_argument("--n-causal-samples", type=int, default=100,
                   help="Number of samples for causal (ablation) analysis (slower)")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


@torch.no_grad()
def compute_geometric_and_logit_alignment(sae: SAEModel, W_E: torch.Tensor,
                                           W_U: torch.Tensor, top_k: int,
                                           device: torch.device):
    """Compute t_geo and t_logit for all features (cheap, no data needed)."""
    D = sae.decoder.weight.data.T.to(device)  # (n_features, dim_input)
    E = W_E.to(device)
    U = W_U.to(device)

    n_features = D.shape[0]

    D_norm = F.normalize(D, dim=1)
    E_norm = F.normalize(E, dim=1)

    geo_topk_sims = torch.zeros(n_features, top_k, device=device)
    geo_topk_tokens = torch.zeros(n_features, top_k, dtype=torch.long, device=device)
    geo_max_sims = torch.zeros(n_features, device=device)

    chunk = 512
    for i in range(0, n_features, chunk):
        sim = D_norm[i:i+chunk] @ E_norm.T
        topk_s, topk_i = sim.topk(top_k, dim=1)
        geo_topk_sims[i:i+chunk] = topk_s
        geo_topk_tokens[i:i+chunk] = topk_i
        geo_max_sims[i:i+chunk] = sim.max(dim=1)[0]

    logit_topk_vals = torch.zeros(n_features, top_k, device=device)
    logit_topk_tokens = torch.zeros(n_features, top_k, dtype=torch.long, device=device)

    for i in range(0, n_features, chunk):
        logit_attr = D[i:i+chunk] @ U.T
        topk_v, topk_i = logit_attr.topk(top_k, dim=1)
        logit_topk_vals[i:i+chunk] = topk_v
        logit_topk_tokens[i:i+chunk] = topk_i

    return {
        "geo_topk_tokens": geo_topk_tokens.cpu(),
        "geo_topk_sims": geo_topk_sims.cpu(),
        "geo_max_sims": geo_max_sims.cpu(),
        "logit_topk_tokens": logit_topk_tokens.cpu(),
        "logit_topk_vals": logit_topk_vals.cpu(),
    }


@torch.no_grad()
def compute_input_alignment(sae: SAEModel, lm_model, tokenizer, layer_idx: int,
                             dataset, n_samples: int, batch_size: int,
                             device: torch.device):
    """Compute t_input: which input tokens maximally activate each feature."""
    n_features = sae.config.dim_sparse
    vocab_size = tokenizer.vocab_size

    feature_token_sum = torch.zeros(n_features, vocab_size, dtype=torch.float32)
    feature_token_count = torch.zeros(n_features, vocab_size, dtype=torch.int32)
    feature_total_act = torch.zeros(n_features, dtype=torch.float64)
    feature_total_count = torch.zeros(n_features, dtype=torch.int64)
    total_positions = 0

    nn_model = NNsight(lm_model)
    n_batches = (n_samples + batch_size - 1) // batch_size

    for batch_idx, batch_start in enumerate(range(0, n_samples, batch_size)):
        if batch_idx % 10 == 0:
            print(f"  Input alignment: batch {batch_idx}/{n_batches}")

        batch_end = min(batch_start + batch_size, n_samples)
        batch_texts = [dataset[i]["text"] for i in range(batch_start, batch_end)]
        batch_texts = [t for t in batch_texts if t.strip()]
        if not batch_texts:
            continue

        enc = tokenizer(batch_texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=128).to(device)
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]

        h = extract_acts(nn_model, input_ids, layer_idx)

        B, S, _ = h.shape
        h_flat = h.reshape(-1, h.shape[-1])
        _, z = sae.encode(h_flat)
        z = z.reshape(B, S, -1)

        mask = attn_mask.bool()
        z_cpu = z.cpu()
        ids_cpu = input_ids.cpu()
        mask_cpu = mask.cpu()

        for b in range(B):
            for s in range(S):
                if not mask_cpu[b, s]:
                    continue
                total_positions += 1
                token_id = ids_cpu[b, s].item()
                z_pos = z_cpu[b, s]
                active = z_pos.nonzero(as_tuple=True)[0]
                for fi in active:
                    fi = fi.item()
                    val = z_pos[fi].item()
                    feature_token_sum[fi, token_id] += val
                    feature_token_count[fi, token_id] += 1
                    feature_total_act[fi] += val
                    feature_total_count[fi] += 1

        del h, z, z_cpu, enc
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "feature_token_sum": feature_token_sum,
        "feature_token_count": feature_token_count,
        "feature_total_act": feature_total_act,
        "feature_total_count": feature_total_count,
        "total_positions": total_positions,
    }


@torch.no_grad()
def compute_causal_alignment(sae: SAEModel, lm_model, tokenizer, layer_idx: int,
                              dataset, n_samples: int, batch_size: int,
                              device: torch.device, top_features: list[int]):
    """Compute t_causal via hook-based activation patching (no nnsight for patching).

    For each batch:
      1. Extract clean activations + encode SAE
      2. Get clean logits (normal forward)
      3. For each feature, use a forward hook to subtract the feature's
         contribution at the target layer, then read logits
    """
    nn_model = NNsight(lm_model)
    vocab_size = tokenizer.vocab_size
    causal_accum = {fi: torch.zeros(vocab_size, dtype=torch.float64) for fi in top_features}
    causal_count = {fi: 0 for fi in top_features}
    n_batches = (n_samples + batch_size - 1) // batch_size

    layers = get_layers(lm_model)
    target_layer = layers[layer_idx]

    for batch_idx, batch_start in enumerate(range(0, n_samples, batch_size)):
        if batch_idx % 5 == 0:
            print(f"  Causal alignment: batch {batch_idx}/{n_batches}")

        batch_end = min(batch_start + batch_size, n_samples)
        batch_texts = [dataset[i]["text"] for i in range(batch_start, batch_end)]
        batch_texts = [t for t in batch_texts if t.strip()]
        if not batch_texts:
            continue

        enc = tokenizer(batch_texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=128).to(device)
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]

        # Get clean activations at target layer
        h_clean = extract_acts(nn_model, input_ids, layer_idx)  # (B, S, D)

        # Encode through SAE
        B, S, D = h_clean.shape
        h_flat = h_clean.reshape(-1, D)
        _, z = sae.encode(h_flat)
        z = z.reshape(B, S, -1)  # (B, S, n_features)

        # Get clean logits (normal forward, no hooks)
        clean_out = lm_model(input_ids=input_ids, attention_mask=attn_mask)
        clean_logits = clean_out.logits.detach()  # (B, S, vocab)

        mask = attn_mask.bool()
        n_valid = mask.sum().item()
        if n_valid == 0:
            continue

        # For each feature, do a hooked forward pass to ablate it
        for fi in top_features:
            z_fi = z[:, :, fi]  # (B, S)
            if (z_fi.abs() < 1e-8).all():
                continue

            d_fi = sae.decoder.weight.data[:, fi].to(device)  # (dim_input,)
            # Precompute the delta to subtract: z_fi[b,s] * d_fi
            delta = z_fi.unsqueeze(-1) * d_fi.unsqueeze(0).unsqueeze(0)  # (B, S, D)

            # Use a forward hook to ablate
            def hook_fn(module, input, output, _delta=delta):
                # GPT-2 block output is a tuple: (hidden_states, presents, ...)
                if isinstance(output, tuple):
                    h = output[0]
                    h_patched = h - _delta
                    return (h_patched,) + output[1:]
                else:
                    return output - _delta

            handle = target_layer.register_forward_hook(hook_fn)
            try:
                ablated_out = lm_model(input_ids=input_ids, attention_mask=attn_mask)
                ablated_logits = ablated_out.logits.detach()
            finally:
                handle.remove()

            # Mean absolute logit change over valid positions
            abs_delta = (clean_logits - ablated_logits).abs()
            delta_masked = abs_delta * mask.unsqueeze(-1).float()
            mean_delta = delta_masked.sum(dim=(0, 1)) / n_valid  # (vocab,)
            causal_accum[fi] += mean_delta.cpu().double()
            causal_count[fi] += 1

            del abs_delta, delta_masked, mean_delta, ablated_logits, delta

        del h_clean, z, clean_logits, enc
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return causal_accum, causal_count


def analyze_consistency(geo_tokens, logit_tokens, input_tokens, causal_tokens,
                        n_features, top_k):
    """Compute pairwise consistency between views."""
    stats = {
        "geo_eq_logit_top1": 0,
        "geo_eq_input_top1": 0,
        "geo_eq_causal_top1": 0,
        "logit_eq_input_top1": 0,
        "logit_eq_causal_top1": 0,
        "input_eq_causal_top1": 0,
        "all_agree_top1": 0,
        "geo_input_topk_overlap": [],
        "geo_causal_topk_overlap": [],
        "n_analyzed": 0,
    }

    feature_ids = sorted(set(geo_tokens.keys()) & set(input_tokens.keys()))
    has_causal = bool(causal_tokens)

    for fi in feature_ids:
        g = geo_tokens[fi]
        l = logit_tokens[fi]
        inp = input_tokens[fi]
        cau = causal_tokens.get(fi)

        if g is None or inp is None:
            continue
        stats["n_analyzed"] += 1

        g0, l0, i0 = g[0], l[0], inp[0]
        if g0 == l0:
            stats["geo_eq_logit_top1"] += 1
        if g0 == i0:
            stats["geo_eq_input_top1"] += 1
        if l0 == i0:
            stats["logit_eq_input_top1"] += 1

        g_set = set(g[:top_k])
        i_set = set(inp[:top_k])
        stats["geo_input_topk_overlap"].append(len(g_set & i_set) / top_k)

        if has_causal and cau is not None:
            c0 = cau[0]
            if g0 == c0:
                stats["geo_eq_causal_top1"] += 1
            if l0 == c0:
                stats["logit_eq_causal_top1"] += 1
            if i0 == c0:
                stats["input_eq_causal_top1"] += 1
            if g0 == l0 == i0 == c0:
                stats["all_agree_top1"] += 1
            c_set = set(cau[:top_k])
            stats["geo_causal_topk_overlap"].append(len(g_set & c_set) / top_k)

    n = max(stats["n_analyzed"], 1)
    summary = {
        "n_features_analyzed": stats["n_analyzed"],
        "geo_eq_logit_top1_pct": stats["geo_eq_logit_top1"] / n * 100,
        "geo_eq_input_top1_pct": stats["geo_eq_input_top1"] / n * 100,
        "logit_eq_input_top1_pct": stats["logit_eq_input_top1"] / n * 100,
        "geo_input_topk_overlap_mean": (sum(stats["geo_input_topk_overlap"]) /
                                         len(stats["geo_input_topk_overlap"])
                                         if stats["geo_input_topk_overlap"] else 0),
    }
    if has_causal:
        summary["geo_eq_causal_top1_pct"] = stats["geo_eq_causal_top1"] / n * 100
        summary["logit_eq_causal_top1_pct"] = stats["logit_eq_causal_top1"] / n * 100
        summary["input_eq_causal_top1_pct"] = stats["input_eq_causal_top1"] / n * 100
        summary["all_agree_top1_pct"] = stats["all_agree_top1"] / n * 100
        summary["geo_causal_topk_overlap_mean"] = (
            sum(stats["geo_causal_topk_overlap"]) /
            len(stats["geo_causal_topk_overlap"])
            if stats["geo_causal_topk_overlap"] else 0
        )

    return summary


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load SAE
    print("Loading SAE...")
    sae = SAEModel.from_pretrained(args.sae_path).to(device).eval()
    n_features = sae.config.dim_sparse
    print(f"  n_features={n_features}, k={sae.config.k}, dim_input={sae.config.dim_input}")

    # Load LM
    print(f"Loading {args.model_name}...")
    lm_model, tokenizer = load_model(args.model_name, device=device)
    W_E = get_embedding(lm_model).weight.data  # (vocab, dim)
    W_U = get_lm_head(lm_model).weight.data    # (vocab, dim)
    vocab_size = W_E.shape[0]
    print(f"  vocab_size={vocab_size}, W_E tied to W_U: {torch.equal(W_E, W_U)}")

    # ========== Step 1: Geometric & Logit alignment (cheap) ==========
    print("\n=== Step 1: Geometric & Logit alignment ===")
    geo_logit = compute_geometric_and_logit_alignment(
        sae, W_E, W_U, top_k=args.top_k, device=device
    )

    geo_top1 = geo_logit["geo_topk_tokens"][:, 0]
    logit_top1 = geo_logit["logit_topk_tokens"][:, 0]
    geo_logit_match = (geo_top1 == logit_top1).float().mean().item()
    print(f"  geo_top1 == logit_top1: {geo_logit_match*100:.1f}%")
    print(f"  Mean geo max_sim: {geo_logit['geo_max_sims'].mean():.4f}")

    # ========== Step 2: Input alignment (need data) ==========
    print("\n=== Step 2: Input alignment (empirical) ===")
    from datasets import load_dataset
    ds = load_dataset(args.dataset, args.dataset_config, split="train")

    input_result = compute_input_alignment(
        sae, lm_model, tokenizer, args.layer_idx,
        ds, args.n_samples, args.batch_size, device
    )

    fts = input_result["feature_token_sum"]
    input_topk_vals, input_topk_tokens = fts.topk(args.top_k, dim=1)

    alive_features = (input_result["feature_total_count"] > 0).sum().item()
    dead_features = n_features - alive_features
    print(f"  Alive features: {alive_features}/{n_features} ({alive_features/n_features*100:.1f}%)")
    print(f"  Dead features: {dead_features}")
    print(f"  Total positions processed: {input_result['total_positions']}")

    # ========== Step 3: Causal alignment (expensive, subset) ==========
    print("\n=== Step 3: Causal alignment (ablation) ===")
    alive_mask = input_result["feature_total_count"] > 0
    alive_indices = alive_mask.nonzero(as_tuple=True)[0]

    alive_geo_sims = geo_logit["geo_max_sims"][alive_indices]
    n_causal = min(200, len(alive_indices))
    _, top_alive_idx = alive_geo_sims.topk(n_causal)
    causal_features = alive_indices[top_alive_idx].tolist()
    print(f"  Running causal analysis on {len(causal_features)} features...")

    causal_accum, causal_count = compute_causal_alignment(
        sae, lm_model, tokenizer, args.layer_idx,
        ds, args.n_causal_samples, args.batch_size, device, causal_features
    )

    causal_topk_tokens = {}
    for fi in causal_features:
        if causal_count[fi] > 0:
            mean_delta = causal_accum[fi] / causal_count[fi]
            _, topk_i = mean_delta.topk(args.top_k)
            causal_topk_tokens[fi] = topk_i.tolist()

    # ========== Step 4: Consistency analysis ==========
    print("\n=== Step 4: Consistency analysis ===")

    geo_tokens = {}
    logit_tokens = {}
    input_tokens = {}

    for fi in range(n_features):
        if not alive_mask[fi]:
            continue
        geo_tokens[fi] = geo_logit["geo_topk_tokens"][fi].tolist()
        logit_tokens[fi] = geo_logit["logit_topk_tokens"][fi].tolist()
        input_tokens[fi] = input_topk_tokens[fi].tolist()

    consistency = analyze_consistency(
        geo_tokens, logit_tokens, input_tokens, causal_topk_tokens,
        n_features, args.top_k
    )

    print("\n--- Consistency Results ---")
    for k, v in consistency.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}{'%' if 'pct' in k else ''}")
        else:
            print(f"  {k}: {v}")

    # ========== Step 5: Feature categorization ==========
    print("\n=== Step 5: Feature categorization ===")
    categories = defaultdict(list)
    for fi in range(n_features):
        if not alive_mask[fi]:
            categories["dead"].append(fi)
            continue

        g0 = geo_logit["geo_topk_tokens"][fi, 0].item()
        i0 = input_topk_tokens[fi, 0].item()
        c0 = causal_topk_tokens.get(fi, [None])[0]

        geo_sim = geo_logit["geo_max_sims"][fi].item()

        if geo_sim < 0.3:
            categories["unaligned"].append(fi)
        elif g0 == i0 and (c0 is None or g0 == c0):
            categories["token_feature"].append(fi)
        elif c0 is not None and g0 == c0 and g0 != i0:
            categories["output_feature"].append(fi)
        elif g0 == i0 and c0 is not None and g0 != c0:
            categories["input_feature"].append(fi)
        else:
            categories["context_feature"].append(fi)

    cat_summary = {cat: len(feats) for cat, feats in categories.items()}
    total_alive = sum(v for k, v in cat_summary.items() if k != "dead")
    print(f"  Total alive: {total_alive}")
    for cat in ["token_feature", "output_feature", "input_feature",
                "context_feature", "unaligned", "dead"]:
        count = cat_summary.get(cat, 0)
        base = total_alive if cat != "dead" else n_features
        print(f"  {cat}: {count} ({count/max(base,1)*100:.1f}%)")

    # ========== Save results ==========
    print("\n=== Saving results ===")

    def tok(token_id):
        return tokenizer.decode([token_id])

    examples = []
    example_indices = alive_indices[top_alive_idx[:50]].tolist()
    for fi in example_indices:
        entry = {
            "feature_id": fi,
            "geo_max_sim": round(geo_logit["geo_max_sims"][fi].item(), 4),
            "geo_top_tokens": [
                {"token": tok(t), "token_id": t, "sim": round(s, 4)}
                for t, s in zip(geo_logit["geo_topk_tokens"][fi].tolist(),
                               geo_logit["geo_topk_sims"][fi].tolist())
            ],
            "logit_top_tokens": [
                {"token": tok(t), "token_id": t, "val": round(v, 4)}
                for t, v in zip(geo_logit["logit_topk_tokens"][fi].tolist(),
                               geo_logit["logit_topk_vals"][fi].tolist())
            ],
            "input_top_tokens": [
                {"token": tok(t), "token_id": t, "act_sum": round(v, 4)}
                for t, v in zip(input_topk_tokens[fi].tolist(),
                               input_topk_vals[fi].tolist())
            ],
            "total_activations": int(input_result["feature_total_count"][fi].item()),
        }
        if fi in causal_topk_tokens:
            entry["causal_top_tokens"] = [
                {"token": tok(t), "token_id": t}
                for t in causal_topk_tokens[fi]
            ]
        examples.append(entry)

    results = {
        "config": {
            "sae_path": args.sae_path,
            "model_name": args.model_name,
            "layer_idx": args.layer_idx,
            "n_samples": args.n_samples,
            "n_causal_samples": args.n_causal_samples,
            "n_features": n_features,
            "top_k": args.top_k,
        },
        "weight_tying": {
            "W_E_eq_W_U": bool(torch.equal(W_E, W_U)),
            "geo_eq_logit_top1_pct": geo_logit_match * 100,
        },
        "feature_stats": {
            "alive": alive_features,
            "dead": dead_features,
            "alive_pct": alive_features / n_features * 100,
        },
        "consistency": consistency,
        "categories": cat_summary,
        "examples": examples,
    }

    results_path = output_dir / "io_decomposition_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to {results_path}")

    torch.save({
        "geo_max_sims": geo_logit["geo_max_sims"],
        "geo_topk_tokens": geo_logit["geo_topk_tokens"],
        "logit_topk_tokens": geo_logit["logit_topk_tokens"],
        "input_topk_tokens": input_topk_tokens,
        "feature_total_count": input_result["feature_total_count"],
    }, output_dir / "io_tensors.pt")
    print(f"  Tensors saved to {output_dir / 'io_tensors.pt'}")


if __name__ == "__main__":
    main()
