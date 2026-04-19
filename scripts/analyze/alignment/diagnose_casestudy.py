"""Diagnose case study: what does argmax(z_relative) actually select at L6?

For each position, show:
  - input token
  - top-1 feature id and its aligned token (relative mode)
  - whether this feature actually fired (z > 0) at this position
  - the z_relative value
  - how many positions this feature fired at in the sentence
"""

import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

import torch
import torch.nn.functional as F

from vasae.analysis.alignment import compute_geometric_alignment
from vasae.analysis.sae_loader import get_decoder_features, load_sae_for_analysis
from vasae.models.factory import get_embedding, load_model

CASES = [
    "Pete Townsend, the legendary guitarist of The Who, walked down the street",
    "Nicole Kidman accepted the award and thanked her family and friends",
    "The cafe is located on Baker Street, just around the corner from the avenue",
]

LAYER = 6
SAE_DIR = "/scratch/b5bq/pu22650.b5bq/VASAE_out/001_F_Benchmarking/001F_gpt2_L6_soft"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    model, tokenizer = load_model("gpt2", device=DEVICE)
    W_E = get_embedding(model).weight.data

    sae = load_sae_for_analysis(SAE_DIR, device=DEVICE)
    geo = compute_geometric_alignment(
        get_decoder_features(sae), W_E, top_k=1, device=DEVICE
    )
    feat_to_tok = geo.topk_indices[:, 0]  # (n_features,)

    for text in CASES:
        toks = tokenizer(text, return_tensors="pt").to(DEVICE)
        input_ids = toks["input_ids"]  # (1, S)
        S = input_ids.shape[1]

        out = model(**toks, output_hidden_states=True)
        h = out.hidden_states[LAYER + 1]  # +1 because index 0 is embedding

        target_dtype = next(sae.parameters()).dtype
        _, z = sae.encode(h.to(target_dtype))
        z_cpu = z.detach().float().cpu()[0]  # (S, F)

        # Relative mode
        feat_mean = z_cpu.mean(dim=0, keepdim=True)  # (1, F)
        z_relative = z_cpu - feat_mean  # (S, F)

        # Full argmax (unrestricted)
        top_feat_ids = z_relative.argmax(dim=-1)  # (S,)

        # Argmax restricted to aligned features (max_sim >= 0.8)
        aligned_mask = (geo.max_sims >= 0.8).cpu()  # (F,)
        z_rel_aligned = z_relative.clone()
        z_rel_aligned[:, ~aligned_mask] = float("-inf")
        top_feat_ids_aligned = z_rel_aligned.argmax(dim=-1)  # (S,)

        # Count aligned feature firings per position
        fired_mask = z_cpu > 0  # (S, F)
        n_fire_total = fired_mask.sum(dim=-1)  # (S,)
        n_fire_aligned = (fired_mask & aligned_mask.unsqueeze(0)).sum(dim=-1)  # (S,)
        print(
            f"Total active per pos: min={n_fire_total.min().item()} "
            f"max={n_fire_total.max().item()} "
            f"mean={n_fire_total.float().mean().item():.1f}"
        )
        print(
            f"Aligned active per pos: min={n_fire_aligned.min().item()} "
            f"max={n_fire_aligned.max().item()} "
            f"mean={n_fire_aligned.float().mean().item():.1f}"
        )

        print(f"\n{'='*80}")
        print(f"TEXT: {text}")
        print(f"{'='*80}")
        print(
            f"{'pos':>3} {'input_tok':<12} "
            f"| {'fid_full':>8} {'alg_full':<12} {'sim':>5} "
            f"| {'fid_algn':>8} {'alg_algn':<12} {'sim':>5} {'fired':>5}"
        )
        print("-" * 100)

        for p in range(S):
            inp_tok = tokenizer.decode([input_ids[0, p].item()])

            fid_f = top_feat_ids[p].item()
            alg_f = tokenizer.decode([feat_to_tok[fid_f].item()])
            sim_f = geo.max_sims[fid_f].item()

            fid_a = top_feat_ids_aligned[p].item()
            alg_a = tokenizer.decode([feat_to_tok[fid_a].item()])
            sim_a = geo.max_sims[fid_a].item()
            z_raw_a = z_cpu[p, fid_a].item()
            fired_a = z_raw_a > 0

            print(
                f"{p:3d} {repr(inp_tok):<12} "
                f"| {fid_f:8d} {repr(alg_f):<12} {sim_f:5.2f} "
                f"| {fid_a:8d} {repr(alg_a):<12} {sim_a:5.2f} {'YES' if fired_a else 'no':>5}"
            )


if __name__ == "__main__":
    main()
