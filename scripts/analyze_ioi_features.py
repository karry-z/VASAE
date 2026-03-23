"""
Analyze IOI feature sweep results to find features with semantically meaningful
vocab alignment for the Indirect Object Identification (IOI) task.

Outputs a table: layer, feature_id, mean_recovery, n_active_prompts, top-5 vocab tokens
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


SWEEP_DIR = Path("/scratch/b5bq/pu22650.b5bq/VASAE_out/ioi_feature_sweep")
SAE_BASE_DIR = Path("/scratch/b5bq/pu22650.b5bq/VASAE_out/010_soft_align")
N_LAYERS = 12
TOP_N = 100
MIN_ACTIVE = 3
TOP_VOCAB = 5


def load_gpt2_embeddings():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print("Loading GPT-2 model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    embeddings = model.transformer.wte.weight.detach()  # (50257, 768)
    print(f"  Embeddings shape: {embeddings.shape}")
    return embeddings, tokenizer


def load_layer_results(layer_idx: int) -> dict:
    path = SWEEP_DIR / f"layer_{layer_idx}.json"
    with open(path) as f:
        return json.load(f)


def aggregate_features(data: dict) -> dict[int, dict]:
    """
    Collect recovery values across all prompts for each feature.
    Returns: {feature_id: {"recoveries": [...], "sae_path": str}}
    """
    sae_path = data["sae_path"]
    # Remap from old lus path to local scratch
    sae_path = sae_path.replace(
        "/lus/lfs1aip2/scratch/b5bq/pu22650.b5bq",
        "/scratch/b5bq/pu22650.b5bq",
    )

    feature_data: dict[int, dict] = defaultdict(lambda: {"recoveries": [], "sae_path": sae_path})

    for example in data["examples"]:
        for feat in example["features"]:
            fid = feat["feature_id"]
            rec = feat["recovery"]
            feature_data[fid]["recoveries"].append(rec)

    return feature_data


def get_top_vocab_tokens(decoder_dir: torch.Tensor, embeddings: torch.Tensor, tokenizer, top_k: int = 5) -> list[str]:
    """
    Compute cosine similarity between decoder direction and all embedding rows,
    return top-k decoded tokens.
    """
    # decoder_dir: (dim,)   embeddings: (vocab, dim)
    dec_norm = F.normalize(decoder_dir.unsqueeze(0), dim=-1)  # (1, dim)
    emb_norm = F.normalize(embeddings, dim=-1)                # (vocab, dim)
    cos_sim = (emb_norm @ dec_norm.T).squeeze(-1)             # (vocab,)
    top_ids = cos_sim.topk(top_k).indices.tolist()
    tokens = [repr(tokenizer.decode([tid])) for tid in top_ids]
    return tokens


def load_sae_decoder(sae_path: str) -> torch.Tensor:
    from vasae.models.sae import SAEModel
    sae = SAEModel.from_pretrained(sae_path)
    sae.eval()
    # decoder.weight shape: (dim_input, dim_sparse)
    return sae.decoder.weight.detach()


def main():
    embeddings, tokenizer = load_gpt2_embeddings()

    # Accumulate per-layer feature stats
    # key: (layer_idx, feature_id)
    all_features: dict[tuple, dict] = {}

    for layer_idx in range(N_LAYERS):
        print(f"Loading layer {layer_idx} results...")
        data = load_layer_results(layer_idx)
        feat_data = aggregate_features(data)

        for fid, info in feat_data.items():
            key = (layer_idx, fid)
            all_features[key] = {
                "layer_idx": layer_idx,
                "feature_id": fid,
                "recoveries": info["recoveries"],
                "sae_path": info["sae_path"],
            }

    print(f"\nTotal unique (layer, feature) pairs: {len(all_features)}")

    # Filter by min active prompts and compute mean recovery
    candidates = []
    for key, info in all_features.items():
        recs = info["recoveries"]
        if len(recs) >= MIN_ACTIVE:
            mean_rec = sum(recs) / len(recs)
            candidates.append({
                **info,
                "mean_recovery": mean_rec,
                "n_active": len(recs),
            })

    print(f"Candidates with >= {MIN_ACTIVE} active prompts: {len(candidates)}")

    # Sort by mean recovery descending, take top N
    candidates.sort(key=lambda x: x["mean_recovery"], reverse=True)
    top_candidates = candidates[:TOP_N]

    print(f"\nTop {TOP_N} features by mean recovery:")
    print(f"  Range: {top_candidates[0]['mean_recovery']:.4f} to {top_candidates[-1]['mean_recovery']:.4f}")

    # Load SAE decoders per layer (cache to avoid repeated loading)
    decoder_cache: dict[str, torch.Tensor] = {}

    print("\nComputing vocab alignments...\n")

    # Print header
    header = f"{'Layer':>5}  {'FeatID':>6}  {'MeanRec':>8}  {'NActive':>7}  Top-5 Vocab Tokens"
    print(header)
    print("-" * len(header))

    ioi_relevant_keywords = {
        # Names / proper nouns
        "mary", "john", "tom", "alice", "bob", "she", "he", "her", "him",
        "his", "they", "them", "their", "who", "whose", "name",
        # Relationships / IOI-relevant
        "gave", "give", "told", "tell", "said", "asked", "sent", "showed",
        "to", "friend", "sister", "brother", "mother", "father",
        # Pronouns / people
        "person", "man", "woman", "girl", "boy",
    }

    ioi_highlighted = []

    for cand in top_candidates:
        layer_idx = cand["layer_idx"]
        fid = cand["feature_id"]
        sae_path = cand["sae_path"]
        mean_rec = cand["mean_recovery"]
        n_active = cand["n_active"]

        # Load decoder for this SAE (cached per path)
        if sae_path not in decoder_cache:
            print(f"  Loading SAE from {sae_path}...")
            decoder_cache[sae_path] = load_sae_decoder(sae_path)

        decoder_weight = decoder_cache[sae_path]  # (dim_input, dim_sparse)

        if fid >= decoder_weight.shape[1]:
            print(f"  WARNING: feature {fid} out of range for decoder shape {decoder_weight.shape}")
            continue

        decoder_dir = decoder_weight[:, fid]  # (dim_input,)
        top_tokens = get_top_vocab_tokens(decoder_dir, embeddings, tokenizer, TOP_VOCAB)

        tokens_str = ", ".join(top_tokens)
        row = f"{layer_idx:>5}  {fid:>6}  {mean_rec:>8.4f}  {n_active:>7}  {tokens_str}"
        print(row)

        # Check if any top token is IOI-relevant
        is_relevant = any(
            kw in tok.lower()
            for tok in top_tokens
            for kw in ioi_relevant_keywords
        )
        if is_relevant:
            ioi_highlighted.append((row, top_tokens))

    print("\n" + "=" * 80)
    print(f"FEATURES WITH IOI-RELEVANT VOCAB ALIGNMENT ({len(ioi_highlighted)} found):")
    print("=" * 80)
    print(header)
    print("-" * len(header))
    for row, _ in ioi_highlighted:
        print(row)

    print("\nDone.")


if __name__ == "__main__":
    main()
