"""Weight-only t_geo analysis across all 12 GPT-2 layers.

Covers:
  - Analysis 1: Alignment margin + permutation baseline
  - Analysis 2: Hub token + embedding geometry bias
  - Analysis 5: Layer-wise t_geo token properties

No data pass needed — only loads SAE decoder weights and GPT-2 embeddings.

Usage:
    python scripts/analyze_tgeo_weight_only.py \
        --model-name gpt2 \
        --sae-dir /scratch/.../010_soft_align \
        --output-dir exp/012_p_TgeoMeaning/weight_only
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

import torch
import torch.nn.functional as F

from vasae.models.sae import SAEModel
from vasae.models.factory import load_model, get_embedding


def parse_args():
    p = argparse.ArgumentParser(description="Weight-only t_geo analysis (all layers)")
    p.add_argument("--model-name", type=str, default="gpt2")
    p.add_argument("--sae-dir", type=str, required=True,
                   help="Base directory containing per-layer SAE dirs")
    p.add_argument("--sae-pattern", type=str, default="010_soft_gpt2_L{layer}_k32_a1e-4",
                   help="Pattern for SAE subdirectory names")
    p.add_argument("--layers", type=str, default="0-11",
                   help="Layer range, e.g. '0-11'")
    p.add_argument("--knn-k", type=int, default=10,
                   help="k for kNN isolation in embedding space")
    p.add_argument("--n-null", type=int, default=5,
                   help="Number of null baseline samples")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def parse_layer_range(s: str) -> list[int]:
    parts = s.split("-")
    return list(range(int(parts[0]), int(parts[1]) + 1))


@torch.no_grad()
def compute_margin_and_tgeo(D: torch.Tensor, E_norm: torch.Tensor, chunk: int = 512):
    """Compute t_geo, margin (sim_1st - sim_2nd), and max sim for all features.

    Args:
        D: (n_features, dim) decoder columns (unnormalized)
        E_norm: (vocab, dim) normalized embeddings
    Returns:
        tgeo_ids: (n_features,) top-1 token ids
        margins: (n_features,) sim_1st - sim_2nd
        max_sims: (n_features,) top-1 cosine similarity
        top2_sims: (n_features, 2) top-2 similarities
    """
    D_norm = F.normalize(D, dim=1)
    n = D.shape[0]
    tgeo_ids = torch.zeros(n, dtype=torch.long, device=D.device)
    margins = torch.zeros(n, device=D.device)
    max_sims = torch.zeros(n, device=D.device)
    top2_sims = torch.zeros(n, 2, device=D.device)

    for i in range(0, n, chunk):
        sim = D_norm[i:i+chunk] @ E_norm.T  # (chunk, vocab)
        top2_s, top2_i = sim.topk(2, dim=1)
        tgeo_ids[i:i+chunk] = top2_i[:, 0]
        max_sims[i:i+chunk] = top2_s[:, 0]
        margins[i:i+chunk] = top2_s[:, 0] - top2_s[:, 1]
        top2_sims[i:i+chunk] = top2_s

    return tgeo_ids, margins, max_sims, top2_sims


@torch.no_grad()
def compute_null_margins(D: torch.Tensor, E_norm: torch.Tensor,
                         n_null: int, chunk: int = 512):
    """Compute margin distributions under null baselines.

    (a) Random orthogonal rotation of D
    (b) Random unit vectors of same shape as D
    """
    device = D.device
    n_features, dim = D.shape
    rotated_margins_all = []
    random_margins_all = []

    for trial in range(n_null):
        # (a) Random orthogonal rotation
        Q, _ = torch.linalg.qr(torch.randn(dim, dim, device=device))
        D_rot = D @ Q  # rotated decoder
        _, margins_rot, _, _ = compute_margin_and_tgeo(D_rot, E_norm, chunk)
        rotated_margins_all.append(margins_rot.cpu())

        # (b) Random unit vectors
        D_rand = torch.randn_like(D)
        D_rand = F.normalize(D_rand, dim=1) * D.norm(dim=1, keepdim=True)
        _, margins_rand, _, _ = compute_margin_and_tgeo(D_rand, E_norm, chunk)
        random_margins_all.append(margins_rand.cpu())

    return torch.stack(rotated_margins_all), torch.stack(random_margins_all)


@torch.no_grad()
def compute_knn_isolation(E_norm: torch.Tensor, k: int, chunk: int = 1024):
    """Mean k-NN cosine similarity for each token embedding."""
    n = E_norm.shape[0]
    knn_sims = torch.zeros(n, device=E_norm.device)
    for i in range(0, n, chunk):
        sim = E_norm[i:i+chunk] @ E_norm.T
        topk_s, _ = sim.topk(k + 1, dim=1)
        knn_sims[i:i+chunk] = topk_s[:, 1:].mean(dim=1)
    return knn_sims


def get_token_properties(tokenizer, token_ids: list[int]):
    """Get properties of tokens: decoded string, length, is_subword."""
    props = []
    for tid in token_ids:
        tok_str = tokenizer.decode([tid])
        props.append({
            "token_id": tid,
            "token": tok_str,
            "length": len(tok_str),
            "starts_with_space": tok_str.startswith(" ") or tok_str.startswith("Ġ"),
            "is_punctuation": all(not c.isalnum() for c in tok_str.strip()),
        })
    return props


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    layers = parse_layer_range(args.layers)

    # Load GPT-2 embeddings
    print(f"Loading {args.model_name}...")
    lm_model, tokenizer = load_model(args.model_name, device=device)
    W_E = get_embedding(lm_model).weight.data.to(device)  # (vocab, dim)
    E_norm = F.normalize(W_E, dim=1)
    vocab_size, dim = W_E.shape
    emb_norms = W_E.norm(dim=1).cpu()
    print(f"  vocab_size={vocab_size}, dim={dim}")

    # Compute kNN isolation for all tokens (once)
    print("Computing kNN isolation for all tokens...")
    knn_sims = compute_knn_isolation(E_norm, args.knn_k).cpu()

    # Compute token frequency ranks from tokenizer (use byte-level heuristic)
    # We don't have corpus frequencies here, so use embedding norm as proxy
    # and note this in results

    # Per-layer analysis
    margin_stats = {}
    hub_stats = {}
    layer_comparison = {}
    all_tensors = {}

    for layer_idx in layers:
        sae_name = args.sae_pattern.format(layer=layer_idx)
        sae_path = Path(args.sae_dir) / sae_name
        print(f"\n=== Layer {layer_idx}: {sae_path} ===")

        if not sae_path.exists():
            print(f"  SKIP: {sae_path} does not exist")
            continue

        sae = SAEModel.from_pretrained(str(sae_path)).to(device).eval()
        D = sae.decoder.weight.data.T.to(device)  # (n_features, dim)
        n_features = D.shape[0]
        print(f"  n_features={n_features}")

        # --- Analysis 1: Margin + null baseline ---
        print("  Computing margins...")
        tgeo_ids, margins, max_sims, top2_sims = compute_margin_and_tgeo(D, E_norm)
        tgeo_ids_cpu = tgeo_ids.cpu()
        margins_cpu = margins.cpu()
        max_sims_cpu = max_sims.cpu()

        print("  Computing null baselines...")
        null_rot_margins, null_rand_margins = compute_null_margins(
            D, E_norm, args.n_null
        )

        margin_stats[layer_idx] = {
            "median_margin": margins_cpu.median().item(),
            "mean_margin": margins_cpu.mean().item(),
            "std_margin": margins_cpu.std().item(),
            "pct_margin_gt_005": (margins_cpu > 0.05).float().mean().item() * 100,
            "pct_margin_gt_01": (margins_cpu > 0.1).float().mean().item() * 100,
            "pct_margin_gt_02": (margins_cpu > 0.2).float().mean().item() * 100,
            "mean_max_sim": max_sims_cpu.mean().item(),
            "median_max_sim": max_sims_cpu.median().item(),
            "null_rotated_median_margin": null_rot_margins.median().item(),
            "null_rotated_mean_margin": null_rot_margins.mean().item(),
            "null_random_median_margin": null_rand_margins.median().item(),
            "null_random_mean_margin": null_rand_margins.mean().item(),
        }
        print(f"    Real median margin: {margin_stats[layer_idx]['median_margin']:.4f}")
        print(f"    Null (rot) median:  {margin_stats[layer_idx]['null_rotated_median_margin']:.4f}")
        print(f"    Null (rand) median: {margin_stats[layer_idx]['null_random_median_margin']:.4f}")

        # --- Analysis 2: Hub tokens + embedding geometry ---
        print("  Computing hub statistics...")
        tgeo_list = tgeo_ids_cpu.tolist()
        hub_counter = Counter(tgeo_list)

        # How many unique tokens are selected as t_geo
        unique_tgeo = len(hub_counter)
        coverage = unique_tgeo / vocab_size * 100

        # Hub count distribution: how many features select each token
        hub_counts = list(hub_counter.values())
        max_hub = max(hub_counts)

        # Token properties of t_geo set vs full vocab
        tgeo_set = list(hub_counter.keys())
        tgeo_norms = emb_norms[tgeo_set]
        tgeo_knn = knn_sims[tgeo_set]
        all_norms = emb_norms
        all_knn = knn_sims

        hub_stats[layer_idx] = {
            "unique_tgeo_tokens": unique_tgeo,
            "coverage_pct": coverage,
            "max_hub_count": max_hub,
            "mean_hub_count": n_features / unique_tgeo if unique_tgeo > 0 else 0,
            "top10_hubs": [
                {"token": tokenizer.decode([tid]), "token_id": tid, "count": cnt}
                for tid, cnt in hub_counter.most_common(10)
            ],
            "tgeo_mean_emb_norm": tgeo_norms.mean().item(),
            "tgeo_median_emb_norm": tgeo_norms.median().item(),
            "all_mean_emb_norm": all_norms.mean().item(),
            "all_median_emb_norm": all_norms.median().item(),
            "tgeo_mean_knn_sim": tgeo_knn.mean().item(),
            "tgeo_median_knn_sim": tgeo_knn.median().item(),
            "all_mean_knn_sim": all_knn.mean().item(),
            "all_median_knn_sim": all_knn.median().item(),
        }
        print(f"    Coverage: {coverage:.1f}% ({unique_tgeo}/{vocab_size})")
        print(f"    Max hub: {max_hub} features → '{tokenizer.decode([hub_counter.most_common(1)[0][0]])}'")

        # --- Analysis 5: Token properties by layer ---
        print("  Computing token properties...")
        tgeo_tokens_decoded = [tokenizer.decode([tid]) for tid in tgeo_set]
        lengths = [len(t) for t in tgeo_tokens_decoded]
        starts_space = [t.startswith(" ") or t.startswith("Ġ") for t in tgeo_tokens_decoded]

        layer_comparison[layer_idx] = {
            "tgeo_mean_token_length": sum(lengths) / len(lengths) if lengths else 0,
            "tgeo_pct_starts_space": sum(starts_space) / len(starts_space) * 100 if starts_space else 0,
            "tgeo_unique_set_size": unique_tgeo,
        }

        # Save tensors
        all_tensors[f"L{layer_idx}"] = {
            "tgeo_ids": tgeo_ids_cpu,
            "margins": margins_cpu,
            "max_sims": max_sims_cpu,
            "null_rot_margins": null_rot_margins,
            "null_rand_margins": null_rand_margins,
        }

        del sae, D
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- Layer-wise Jaccard overlap ---
    print("\n=== Computing layer-wise Jaccard overlap ===")
    jaccard = {}
    for i in layers:
        key_i = f"L{i}"
        if key_i not in all_tensors:
            continue
        set_i = set(all_tensors[key_i]["tgeo_ids"].tolist())
        for j in layers:
            if j <= i:
                continue
            key_j = f"L{j}"
            if key_j not in all_tensors:
                continue
            set_j = set(all_tensors[key_j]["tgeo_ids"].tolist())
            inter = len(set_i & set_j)
            union = len(set_i | set_j)
            jac = inter / union if union > 0 else 0
            jaccard[f"L{i}_L{j}"] = round(jac, 4)
            print(f"  Jaccard(L{i}, L{j}) = {jac:.4f}")

    layer_comparison["jaccard_overlap"] = jaccard

    # --- Save all results ---
    print("\n=== Saving results ===")

    # Convert keys to strings for JSON
    def int_keys_to_str(d):
        return {str(k): v for k, v in d.items()}

    with open(output_dir / "margin_stats.json", "w") as f:
        json.dump(int_keys_to_str(margin_stats), f, indent=2, ensure_ascii=False)

    with open(output_dir / "hub_stats.json", "w") as f:
        json.dump(int_keys_to_str(hub_stats), f, indent=2, ensure_ascii=False)

    with open(output_dir / "layer_comparison.json", "w") as f:
        json.dump(int_keys_to_str(layer_comparison), f, indent=2, ensure_ascii=False)

    torch.save(all_tensors, output_dir / "tensors.pt")

    print(f"  Saved to {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
