"""Conditional t_geo analysis: split features by t_geo token type.

Classifies each token as content_word / function_word / punctuation / subword_fragment,
then recomputes all key metrics per group to see if content-word features
show stronger alignment signals.

Runs on CPU using saved tensors from 012a/012b/012c. No GPU needed.

Usage:
    python scripts/analyze_tgeo_by_token_type.py \
        --model-name gpt2 \
        --weight-dir exp/012_p_TgeoMeaning/weight_only \
        --data-dir exp/012_p_TgeoMeaning/data \
        --io-dir exp/012_p_TgeoMeaning/io_full \
        --output-dir exp/012_p_TgeoMeaning/by_token_type
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


# ─── Token Classification ───────────────────────────────────────────────

FUNCTION_WORDS = {
    # determiners
    "the", "a", "an", "this", "that", "these", "those", "my", "your", "his",
    "her", "its", "our", "their", "some", "any", "no", "every", "each", "all",
    # prepositions
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "over", "up", "down", "out", "off", "against",
    # conjunctions
    "and", "or", "but", "nor", "so", "yet", "both", "either", "neither",
    # pronouns
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "us", "them",
    "who", "whom", "which", "what", "that", "whose",
    # aux / copula
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "can", "could", "must",
    # other function
    "not", "no", "if", "then", "than", "as", "also", "just", "only",
    "very", "too", "even", "still", "already", "much", "more", "most",
    "own", "other", "such", "when", "where", "how", "why", "because",
    "while", "although", "though", "since", "until", "unless",
    "there", "here",
    # contractions (GPT-2 BPE pieces)
    "'s", "'t", "'re", "'ve", "'ll", "'d", "'m",
    "n't",
}


def classify_token(token_str: str) -> str:
    """Classify a decoded GPT-2 token string into a category.

    Categories:
      - punctuation: only punctuation/whitespace chars
      - subword_fragment: does not start with space (Ġ) and is alphabetic
                          (= continuation piece, not a word boundary)
      - function_word: common function words
      - content_word: everything else that starts a word
    """
    stripped = token_str.strip()

    # Empty / whitespace only
    if not stripped:
        return "punctuation"

    # Pure punctuation / symbols / digits
    if all(not c.isalpha() for c in stripped):
        return "punctuation"

    # Subword fragment: alphabetic but doesn't start with space (Ġ in raw)
    # In decoded form, GPT-2 word-initial tokens start with " " (space)
    is_word_start = token_str.startswith(" ") or token_str.startswith("Ġ")

    if not is_word_start:
        # Could be a standalone single-char or a BPE fragment
        # If it's very short and lowercase, likely a fragment
        if len(stripped) <= 3 and stripped.isalpha() and stripped.islower():
            return "subword_fragment"
        # Longer fragments without leading space are still fragments
        if stripped.isalpha():
            return "subword_fragment"
        return "punctuation"

    # Word-initial token: check if function word
    word = stripped.lower()
    if word in FUNCTION_WORDS:
        return "function_word"

    return "content_word"


def build_token_type_map(tokenizer) -> dict[int, str]:
    """Classify all tokens in the vocabulary."""
    type_map = {}
    for token_id in range(tokenizer.vocab_size):
        token_str = tokenizer.decode([token_id])
        type_map[token_id] = classify_token(token_str)
    return type_map


# ─── Analysis ────────────────────────────────────────────────────────────

def analyze_margin_by_type(tensors_012a: dict, token_type_map: dict,
                           layers: list[int]):
    """Split margin statistics by t_geo token type."""
    results = {}
    for layer in layers:
        key = f"L{layer}"
        if key not in tensors_012a:
            continue
        t = tensors_012a[key]
        tgeo_ids = t["tgeo_ids"]
        margins = t["margins"]
        max_sims = t["max_sims"]

        groups = defaultdict(list)
        sim_groups = defaultdict(list)
        for i in range(len(tgeo_ids)):
            tid = tgeo_ids[i].item()
            ttype = token_type_map.get(tid, "unknown")
            groups[ttype].append(margins[i].item())
            sim_groups[ttype].append(max_sims[i].item())

        layer_result = {}
        for ttype in ["content_word", "function_word", "punctuation", "subword_fragment"]:
            m = groups.get(ttype, [])
            s = sim_groups.get(ttype, [])
            if m:
                mt = torch.tensor(m)
                st = torch.tensor(s)
                layer_result[ttype] = {
                    "count": len(m),
                    "pct": len(m) / len(tgeo_ids) * 100,
                    "margin_median": mt.median().item(),
                    "margin_mean": mt.mean().item(),
                    "margin_gt_01_pct": (mt > 0.1).float().mean().item() * 100,
                    "margin_gt_02_pct": (mt > 0.2).float().mean().item() * 100,
                    "max_sim_median": st.median().item(),
                    "max_sim_mean": st.mean().item(),
                }
            else:
                layer_result[ttype] = {"count": 0}
        results[layer] = layer_result
    return results


def analyze_data_by_type(data_tensors: dict, token_type_map: dict):
    """Split 012b data-dependent metrics by t_geo token type."""
    tgeo_ids = data_tensors["tgeo_ids"]
    cos_d_mu = data_tensors["cos_d_mu"]
    t_mean_ids = data_tensors["t_mean_ids"]
    feat_act_count = data_tensors["feat_act_count"]
    feat_tgeo_pos_match = data_tensors["feat_tgeo_pos_match"]
    feat_total_activations = data_tensors["feat_total_activations"]

    n_features = len(tgeo_ids)
    alive_mask = feat_act_count > 0

    groups = defaultdict(lambda: {
        "cos_d_mu": [], "tgeo_eq_tmean": 0, "count": 0,
        "pos_match_sums": None, "pos_count": 0,
    })

    n_offsets = feat_tgeo_pos_match.shape[1]

    for fi in range(n_features):
        if not alive_mask[fi]:
            continue
        tid = tgeo_ids[fi].item()
        ttype = token_type_map.get(tid, "unknown")
        g = groups[ttype]
        g["count"] += 1
        g["cos_d_mu"].append(cos_d_mu[fi].item())
        if tgeo_ids[fi].item() == t_mean_ids[fi].item():
            g["tgeo_eq_tmean"] += 1

        total_act = feat_total_activations[fi].item()
        if total_act > 0:
            if g["pos_match_sums"] is None:
                g["pos_match_sums"] = torch.zeros(n_offsets)
            for w in range(n_offsets):
                g["pos_match_sums"][w] += feat_tgeo_pos_match[fi, w].item() / total_act
            g["pos_count"] += 1

    result = {}
    for ttype in ["content_word", "function_word", "punctuation", "subword_fragment"]:
        g = groups[ttype]
        if g["count"] == 0:
            result[ttype] = {"count": 0}
            continue
        cos_t = torch.tensor(g["cos_d_mu"])
        entry = {
            "count": g["count"],
            "cos_d_mu_mean": cos_t.mean().item(),
            "cos_d_mu_median": cos_t.median().item(),
            "tgeo_eq_tmean_pct": g["tgeo_eq_tmean"] / g["count"] * 100,
        }
        if g["pos_count"] > 0:
            avg_pos = g["pos_match_sums"] / g["pos_count"] * 100
            entry["pos_match_offset_0_pct"] = avg_pos[5].item()  # offset 0 = index 5
            entry["pos_match_offset_1_pct"] = avg_pos[6].item()  # offset +1 = index 6
        result[ttype] = entry
    return result


def analyze_io_by_type(io_tensors: dict, io_results: dict,
                       token_type_map: dict):
    """Split 012c IO consistency by t_geo token type."""
    geo_topk = io_tensors["geo_topk_tokens"]   # (n_features, 5)
    input_topk = io_tensors["input_topk_tokens"]  # (n_features, 5)
    feat_count = io_tensors["feature_total_count"]
    geo_max_sims = io_tensors["geo_max_sims"]

    n_features = geo_topk.shape[0]
    alive_mask = feat_count > 0

    # Load causal info from results JSON
    causal_map = {}
    for ex in io_results.get("examples", []):
        fi = ex["feature_id"]
        if "causal_top_tokens" in ex and ex["causal_top_tokens"]:
            causal_map[fi] = ex["causal_top_tokens"][0]["token_id"]

    groups = defaultdict(lambda: {
        "count": 0, "geo_eq_input": 0, "geo_eq_causal": 0,
        "has_causal": 0, "input_eq_causal": 0,
        "max_sims": [],
    })

    for fi in range(n_features):
        if not alive_mask[fi]:
            continue
        tid = geo_topk[fi, 0].item()
        ttype = token_type_map.get(tid, "unknown")
        g = groups[ttype]
        g["count"] += 1
        g["max_sims"].append(geo_max_sims[fi].item())

        g0 = geo_topk[fi, 0].item()
        i0 = input_topk[fi, 0].item()
        if g0 == i0:
            g["geo_eq_input"] += 1

        if fi in causal_map:
            c0 = causal_map[fi]
            g["has_causal"] += 1
            if g0 == c0:
                g["geo_eq_causal"] += 1
            if i0 == c0:
                g["input_eq_causal"] += 1

    result = {}
    for ttype in ["content_word", "function_word", "punctuation", "subword_fragment"]:
        g = groups[ttype]
        n = max(g["count"], 1)
        nc = max(g["has_causal"], 1)
        entry = {
            "count": g["count"],
            "geo_eq_input_pct": g["geo_eq_input"] / n * 100,
        }
        if g["has_causal"] > 0:
            entry["geo_eq_causal_pct"] = g["geo_eq_causal"] / nc * 100
            entry["input_eq_causal_pct"] = g["input_eq_causal"] / nc * 100
            entry["n_with_causal"] = g["has_causal"]
        if g["max_sims"]:
            st = torch.tensor(g["max_sims"])
            entry["max_sim_mean"] = st.mean().item()
            entry["max_sim_median"] = st.median().item()
        result[ttype] = entry
    return result


# ─── Token type distribution overview ────────────────────────────────────

def token_type_overview(token_type_map: dict, tokenizer):
    """Show distribution of token types across full vocab."""
    counts = Counter(token_type_map.values())
    total = len(token_type_map)
    overview = {}
    for ttype in ["content_word", "function_word", "punctuation", "subword_fragment"]:
        c = counts.get(ttype, 0)
        overview[ttype] = {"count": c, "pct": c / total * 100}
    # Sample tokens
    for ttype in ["content_word", "function_word", "punctuation", "subword_fragment"]:
        examples = [tokenizer.decode([tid]) for tid, tt in
                    list(token_type_map.items())[:5000] if tt == ttype][:10]
        overview[ttype]["examples"] = examples
    return overview


# ─── Main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="gpt2")
    p.add_argument("--weight-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--io-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--layers", default="0-11")
    p.add_argument("--data-layers", default="2,6,11")
    return p.parse_args()


def parse_layers(s):
    if "," in s:
        return [int(x) for x in s.split(",")]
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_layers(args.layers)
    data_layers = parse_layers(args.data_layers)
    weight_dir = Path(args.weight_dir)
    data_dir = Path(args.data_dir)
    io_dir = Path(args.io_dir)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Classifying tokens...")
    token_type_map = build_token_type_map(tokenizer)
    overview = token_type_overview(token_type_map, tokenizer)
    print(f"  Token type distribution:")
    for ttype, info in overview.items():
        print(f"    {ttype}: {info['count']} ({info['pct']:.1f}%) — e.g. {info['examples'][:5]}")

    # ── 012a: Margin by type ──
    print("\n=== 012a: Margin by token type ===")
    tensors_012a = torch.load(weight_dir / "tensors.pt", map_location="cpu",
                              weights_only=True)
    margin_by_type = analyze_margin_by_type(tensors_012a, token_type_map, layers)

    for layer in layers:
        if layer not in margin_by_type:
            continue
        print(f"\n  Layer {layer}:")
        for ttype in ["content_word", "function_word", "punctuation", "subword_fragment"]:
            d = margin_by_type[layer].get(ttype, {})
            if d.get("count", 0) == 0:
                continue
            print(f"    {ttype:20s}: n={d['count']:4d} ({d['pct']:5.1f}%)  "
                  f"margin_med={d['margin_median']:.3f}  "
                  f"max_sim_med={d['max_sim_median']:.3f}  "
                  f"margin>0.2={d['margin_gt_02_pct']:.1f}%")

    # ── 012b: Data metrics by type ──
    print("\n=== 012b: Data-dependent metrics by token type ===")
    data_by_type = {}
    for layer in data_layers:
        tensor_path = data_dir / f"L{layer}_k32" / "tgeo_data_tensors.pt"
        if not tensor_path.exists():
            continue
        dt = torch.load(tensor_path, map_location="cpu", weights_only=True)
        result = analyze_data_by_type(dt, token_type_map)
        data_by_type[layer] = result
        print(f"\n  Layer {layer}:")
        for ttype in ["content_word", "function_word", "punctuation", "subword_fragment"]:
            d = result.get(ttype, {})
            if d.get("count", 0) == 0:
                continue
            cos_str = f"cos(d,mu)={d.get('cos_d_mu_mean', 0):.3f}"
            tmean_str = f"t_geo=t_mean={d.get('tgeo_eq_tmean_pct', 0):.1f}%"
            pos0_str = f"pos0={d.get('pos_match_offset_0_pct', 0):.2f}%"
            pos1_str = f"pos+1={d.get('pos_match_offset_1_pct', 0):.2f}%"
            print(f"    {ttype:20s}: n={d['count']:4d}  {cos_str}  {tmean_str}  {pos0_str}  {pos1_str}")

    # ── 012c: IO consistency by type ──
    print("\n=== 012c: IO consistency by token type ===")
    io_by_type = {}
    for layer in layers:
        tensor_path = io_dir / f"L{layer}_k32" / "io_tensors.pt"
        result_path = io_dir / f"L{layer}_k32" / "io_decomposition_results.json"
        if not tensor_path.exists() or not result_path.exists():
            continue
        it = torch.load(tensor_path, map_location="cpu", weights_only=True)
        with open(result_path) as f:
            ir = json.load(f)
        result = analyze_io_by_type(it, ir, token_type_map)
        io_by_type[layer] = result
        print(f"\n  Layer {layer}:")
        for ttype in ["content_word", "function_word", "punctuation", "subword_fragment"]:
            d = result.get(ttype, {})
            if d.get("count", 0) == 0:
                continue
            geo_input = f"geo=input={d.get('geo_eq_input_pct', 0):.1f}%"
            geo_causal = f"geo=causal={d.get('geo_eq_causal_pct', 0):.1f}%" if "geo_eq_causal_pct" in d else "geo=causal=N/A"
            sim_str = f"sim={d.get('max_sim_mean', 0):.3f}"
            print(f"    {ttype:20s}: n={d['count']:4d}  {sim_str}  {geo_input}  {geo_causal}")

    # ── Save ──
    all_results = {
        "token_type_overview": {k: {kk: vv for kk, vv in v.items() if kk != "examples"}
                                for k, v in overview.items()},
        "margin_by_type": {str(k): v for k, v in margin_by_type.items()},
        "data_by_type": {str(k): v for k, v in data_by_type.items()},
        "io_by_type": {str(k): v for k, v in io_by_type.items()},
    }

    with open(output_dir / "tgeo_by_token_type.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_dir / 'tgeo_by_token_type.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
