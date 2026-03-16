"""Data-dependent t_geo analysis for a single layer.

Covers:
  - Analysis 3: Mean activation direction (does d_i ≈ mean(h | z_i > 0)?)
  - Analysis 4: Context position analysis (is t_geo = next-token, prev-token, etc.?)

Requires GPU + data pass through GPT-2.

Usage:
    python scripts/analyze_tgeo_data.py \
        --sae-path /scratch/.../010_soft_gpt2_L6_k32_a1e-4 \
        --layer-idx 6 \
        --n-samples 500 \
        --output-dir exp/012_p_TgeoMeaning/data/L6_k32
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
from vasae.models.factory import load_model, get_embedding
from vasae.engine.intervention import _get_layer_proxy


def extract_acts(nn_model: NNsight, input_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
    with nn_model.trace(input_ids):
        layer = _get_layer_proxy(nn_model, layer_idx)
        h = layer.output.save()
    return h.detach()


def parse_args():
    p = argparse.ArgumentParser(description="Data-dependent t_geo analysis")
    p.add_argument("--sae-path", type=str, required=True)
    p.add_argument("--model-name", type=str, default="gpt2")
    p.add_argument("--layer-idx", type=int, required=True)
    p.add_argument("--dataset", type=str, default="wikitext")
    p.add_argument("--dataset-config", type=str, default="wikitext-103-raw-v1")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--context-window", type=int, default=5,
                   help="Window size for context position analysis (each side)")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


@torch.no_grad()
def run_data_analysis(sae, lm_model, tokenizer, layer_idx, dataset,
                      n_samples, batch_size, context_window, device):
    """Single data pass collecting mean activation direction + context tokens."""
    n_features = sae.config.dim_sparse
    dim = sae.config.dim_input
    nn_model = NNsight(lm_model)

    W_E = get_embedding(lm_model).weight.data.to(device)
    E_norm = F.normalize(W_E, dim=1)
    vocab_size = W_E.shape[0]

    # Analysis 3: accumulators for mean activation direction
    feat_h_sum = torch.zeros(n_features, dim, dtype=torch.float64, device="cpu")
    feat_act_count = torch.zeros(n_features, dtype=torch.int64)

    # Analysis 4: context position token counts
    # For each feature, count how often t_geo matches token at each offset
    window_range = list(range(-context_window, context_window + 1))
    # feat_position_counts[fi][offset] = Counter of token_ids
    # To save memory, only track t_geo match counts per offset
    feat_tgeo_pos_match = torch.zeros(n_features, len(window_range), dtype=torch.int64)
    feat_total_activations = torch.zeros(n_features, dtype=torch.int64)

    # Also compute random baseline: for non-active features at each position
    random_tgeo_pos_match = torch.zeros(n_features, len(window_range), dtype=torch.int64)
    random_total = torch.zeros(n_features, dtype=torch.int64)

    # Pre-compute t_geo for all features
    D = sae.decoder.weight.data.T.to(device)
    D_norm = F.normalize(D, dim=1)
    chunk_size = 512
    tgeo_ids = torch.zeros(n_features, dtype=torch.long, device=device)
    for i in range(0, n_features, chunk_size):
        sim = D_norm[i:i+chunk_size] @ E_norm.T
        tgeo_ids[i:i+chunk_size] = sim.argmax(dim=1)
    tgeo_ids_cpu = tgeo_ids.cpu()

    n_batches = (n_samples + batch_size - 1) // batch_size
    total_positions = 0

    for batch_idx, batch_start in enumerate(range(0, n_samples, batch_size)):
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{n_batches}")

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
        B, S, D_dim = h.shape
        h_flat = h.reshape(-1, D_dim)
        _, z = sae.encode(h_flat)
        z = z.reshape(B, S, -1)

        mask = attn_mask.bool()
        z_cpu = z.cpu()
        ids_cpu = input_ids.cpu()
        mask_cpu = mask.cpu()
        h_cpu = h.cpu().double()

        for b in range(B):
            seq_len = mask_cpu[b].sum().item()
            for s in range(seq_len):
                total_positions += 1
                z_pos = z_cpu[b, s]
                active = z_pos.nonzero(as_tuple=True)[0]

                # Collect active and inactive features for this position
                active_set = set(active.tolist())

                for fi_t in active:
                    fi = fi_t.item()

                    # Analysis 3: accumulate h for mean direction
                    feat_h_sum[fi] += h_cpu[b, s]
                    feat_act_count[fi] += 1
                    feat_total_activations[fi] += 1

                    # Analysis 4: check context positions
                    tgeo_token = tgeo_ids_cpu[fi].item()
                    for w_idx, offset in enumerate(window_range):
                        pos = s + offset
                        if 0 <= pos < seq_len:
                            if ids_cpu[b, pos].item() == tgeo_token:
                                feat_tgeo_pos_match[fi, w_idx] += 1

                # Random baseline: sample a few inactive features
                if len(active_set) < n_features:
                    # Sample up to 5 inactive features for baseline
                    inactive_candidates = []
                    for _ in range(min(5, n_features - len(active_set))):
                        ri = torch.randint(0, n_features, (1,)).item()
                        if ri not in active_set:
                            inactive_candidates.append(ri)
                    for fi in inactive_candidates:
                        random_total[fi] += 1
                        tgeo_token = tgeo_ids_cpu[fi].item()
                        for w_idx, offset in enumerate(window_range):
                            pos = s + offset
                            if 0 <= pos < seq_len:
                                if ids_cpu[b, pos].item() == tgeo_token:
                                    random_tgeo_pos_match[fi, w_idx] += 1

        del h, z, z_cpu, enc, h_cpu
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "feat_h_sum": feat_h_sum,
        "feat_act_count": feat_act_count,
        "feat_total_activations": feat_total_activations,
        "feat_tgeo_pos_match": feat_tgeo_pos_match,
        "random_tgeo_pos_match": random_tgeo_pos_match,
        "random_total": random_total,
        "tgeo_ids": tgeo_ids_cpu,
        "total_positions": total_positions,
        "window_range": window_range,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load SAE
    print("Loading SAE...")
    sae = SAEModel.from_pretrained(args.sae_path).to(device).eval()
    n_features = sae.config.dim_sparse
    dim = sae.config.dim_input
    print(f"  n_features={n_features}, dim={dim}, k={sae.config.k}")

    # Load LM
    print(f"Loading {args.model_name}...")
    lm_model, tokenizer = load_model(args.model_name, device=device)
    W_E = get_embedding(lm_model).weight.data.to(device)
    E_norm = F.normalize(W_E, dim=1)

    # Load dataset
    print("Loading dataset...")
    from datasets import load_dataset
    ds = load_dataset(args.dataset, args.dataset_config, split="train")

    # Run data analysis
    print("\n=== Running data analysis ===")
    result = run_data_analysis(
        sae, lm_model, tokenizer, args.layer_idx, ds,
        args.n_samples, args.batch_size, args.context_window, device
    )

    # --- Analysis 3: Mean activation direction ---
    print("\n=== Analysis 3: Mean activation direction ===")
    D = sae.decoder.weight.data.T.to(device)  # (n_features, dim)
    tgeo_ids = result["tgeo_ids"]
    feat_h_sum = result["feat_h_sum"]
    feat_act_count = result["feat_act_count"]
    window_range = result["window_range"]

    alive_mask = feat_act_count > 0
    alive_indices = alive_mask.nonzero(as_tuple=True)[0]
    n_alive = len(alive_indices)
    print(f"  Alive features: {n_alive}/{n_features}")

    # Compute mean activation direction and its nearest token
    cos_d_mu = torch.zeros(n_features)
    t_mean_ids = torch.zeros(n_features, dtype=torch.long)
    tgeo_eq_tmean = 0

    for fi in alive_indices:
        fi = fi.item()
        mu = feat_h_sum[fi] / feat_act_count[fi]  # (dim,)
        mu_dev = mu.to(device).float()
        d_i = D[fi]

        # cos(d_i, mu_i)
        cos_val = F.cosine_similarity(d_i.unsqueeze(0), mu_dev.unsqueeze(0)).item()
        cos_d_mu[fi] = cos_val

        # t_mean = argmax_t cos(mu_i, e_t)
        mu_norm = F.normalize(mu_dev.unsqueeze(0), dim=1)
        sim_to_vocab = (mu_norm @ E_norm.T).squeeze(0)
        t_mean = sim_to_vocab.argmax().item()
        t_mean_ids[fi] = t_mean

        if t_mean == tgeo_ids[fi].item():
            tgeo_eq_tmean += 1

    tgeo_eq_tmean_pct = tgeo_eq_tmean / max(n_alive, 1) * 100
    alive_cos = cos_d_mu[alive_indices]
    print(f"  t_geo == t_mean: {tgeo_eq_tmean}/{n_alive} ({tgeo_eq_tmean_pct:.1f}%)")
    print(f"  cos(d_i, mu_i): mean={alive_cos.mean():.4f}, median={alive_cos.median():.4f}")

    # --- Analysis 4: Context position analysis ---
    print("\n=== Analysis 4: Context position analysis ===")
    feat_tgeo_pos_match = result["feat_tgeo_pos_match"]
    feat_total_activations = result["feat_total_activations"]
    random_tgeo_pos_match = result["random_tgeo_pos_match"]
    random_total = result["random_total"]

    # Per-position match rates (averaged over alive features)
    position_match_rates = {}
    random_position_match_rates = {}

    for w_idx, offset in enumerate(window_range):
        # Real: for alive features, what fraction of activations have t_geo at this offset?
        alive_rates = []
        for fi in alive_indices:
            fi = fi.item()
            total = feat_total_activations[fi].item()
            if total > 0:
                rate = feat_tgeo_pos_match[fi, w_idx].item() / total
                alive_rates.append(rate)
        mean_rate = sum(alive_rates) / len(alive_rates) if alive_rates else 0
        position_match_rates[f"offset_{offset}"] = mean_rate * 100

        # Random baseline
        rand_rates = []
        for fi in range(n_features):
            total = random_total[fi].item()
            if total > 10:  # need enough samples
                rate = random_tgeo_pos_match[fi, w_idx].item() / total
                rand_rates.append(rate)
        rand_mean = sum(rand_rates) / len(rand_rates) if rand_rates else 0
        random_position_match_rates[f"offset_{offset}"] = rand_mean * 100

    print("  Position match rates (t_geo == token at offset):")
    for offset in window_range:
        key = f"offset_{offset}"
        label = {-1: "prev", 0: "current", 1: "next"}.get(offset, str(offset))
        real = position_match_rates[key]
        rand = random_position_match_rates[key]
        print(f"    {label:>8s} (offset={offset:+d}): {real:.3f}% (random: {rand:.3f}%)")

    # --- Save results ---
    print("\n=== Saving results ===")
    results = {
        "config": {
            "sae_path": args.sae_path,
            "model_name": args.model_name,
            "layer_idx": args.layer_idx,
            "n_samples": args.n_samples,
            "context_window": args.context_window,
        },
        "feature_stats": {
            "n_features": n_features,
            "n_alive": n_alive,
            "alive_pct": n_alive / n_features * 100,
            "total_positions": result["total_positions"],
        },
        "analysis_3_mean_direction": {
            "tgeo_eq_tmean_count": tgeo_eq_tmean,
            "tgeo_eq_tmean_pct": tgeo_eq_tmean_pct,
            "cos_d_mu_mean": alive_cos.mean().item(),
            "cos_d_mu_median": alive_cos.median().item(),
            "cos_d_mu_std": alive_cos.std().item(),
            "cos_d_mu_q25": alive_cos.quantile(0.25).item(),
            "cos_d_mu_q75": alive_cos.quantile(0.75).item(),
        },
        "analysis_4_context_position": {
            "position_match_rates_pct": position_match_rates,
            "random_baseline_match_rates_pct": random_position_match_rates,
            "window_offsets": window_range,
        },
    }

    with open(output_dir / "tgeo_data_analysis.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    torch.save({
        "cos_d_mu": cos_d_mu,
        "t_mean_ids": t_mean_ids,
        "tgeo_ids": tgeo_ids,
        "feat_act_count": feat_act_count,
        "feat_tgeo_pos_match": feat_tgeo_pos_match,
        "feat_total_activations": feat_total_activations,
    }, output_dir / "tgeo_data_tensors.pt")

    print(f"  Saved to {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
