"""
Experiment: Explicitly decompose position/frequency/wordform from activations,
then test whether OMP@k reconstruction improves on the cleaned residual.

Approach:
  h = h_interpretable + h_clean
  h_interpretable = X @ W  (linear projection from [position, log_freq, is_word_start, ...])
  h_clean = h - h_interpretable

  If the non-sparse component IS these features, then OMP@k(h_clean, E) should be
  much better than OMP@k(h, E).

We also test progressively richer feature sets:
  F0: nothing (baseline)
  F1: position only (one-hot or scalar)
  F2: position + log_frequency
  F3: position + log_frequency + is_word_start
  F4: position + log_frequency + is_word_start + is_punctuation + token_strlen
  F5: learned linear probe (OLS) from all features → residual (upper bound)
"""

import argparse
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_missing_components import (
    get_device,
    get_sample_texts,
    get_token_embeddings,
    load_gpt2,
    omp_k_error,
    save_json,
    set_seed,
)
from analyze_nonvocab_subspace import compute_omp_residuals

logging.basicConfig(
    format="[%(levelname)s] %(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_token_features(tokenizer):
    """Pre-compute per-token-id features."""
    vocab_size = tokenizer.vocab_size

    function_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "must", "need",
        "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below",
        "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
        "neither", "each", "every", "all", "any", "few", "more", "most",
        "other", "some", "such", "no", "only", "own", "same", "than",
        "too", "very", "just", "because", "if", "when", "while", "although",
        "that", "which", "who", "whom", "whose", "what", "where", "how",
        "this", "these", "those", "it", "its", "he", "she", "they", "we",
        "me", "him", "her", "us", "them", "my", "your", "his", "our", "their",
        "i",
    }
    punct_chars = set(".,;:!?\"'()[]{}-/\\@#$%^&*~`<>|+=_")

    is_function = torch.zeros(vocab_size)
    is_punct = torch.zeros(vocab_size)
    is_word_start = torch.zeros(vocab_size)
    token_strlen = torch.zeros(vocab_size)

    for tid in range(vocab_size):
        tok_str = tokenizer.decode([tid])
        tok_clean = tok_str.strip().lower()
        is_function[tid] = 1.0 if tok_clean in function_words else 0.0
        is_punct[tid] = 1.0 if (len(tok_clean) > 0 and all(c in punct_chars for c in tok_clean)) else 0.0
        raw_token = tokenizer.convert_ids_to_tokens(tid)
        is_word_start[tid] = 1.0 if (raw_token and raw_token.startswith("Ġ")) else 0.0
        token_strlen[tid] = len(tok_str)

    return {
        "is_function_word": is_function,
        "is_punctuation": is_punct,
        "is_word_start": is_word_start,
        "token_str_length": token_strlen,
    }


def collect_activations_with_metadata(model, tokenizer, texts, layers, device, max_length=64):
    """Collect hidden states + position/token metadata."""
    all_hidden = {l: [] for l in layers}
    all_positions = []
    all_token_ids = []

    for text in texts:
        tokens = tokenizer(
            text, return_tensors="pt", max_length=max_length,
            truncation=True, padding="max_length",
        ).to(device)
        input_ids = tokens["input_ids"].squeeze(0)

        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)

        for l in layers:
            h = outputs.hidden_states[l + 1].squeeze(0).float().cpu()
            all_hidden[l].append(h)

        all_positions.append(torch.arange(max_length))
        all_token_ids.append(input_ids.cpu())

    positions_flat = torch.cat(all_positions)
    token_ids_flat = torch.cat(all_token_ids)

    hidden_flat = {}
    for l in layers:
        hidden_flat[l] = torch.cat(all_hidden[l], dim=0)  # [N*seq, d]

    return hidden_flat, positions_flat, token_ids_flat


def build_feature_matrices(positions, token_ids, token_features, log_freq, max_length=64):
    """Build progressively richer feature matrices.

    Returns dict of {feature_set_name: X [N, n_features]}.
    """
    N = positions.shape[0]

    # Scalar position (normalized to [0, 1])
    pos_scalar = positions.float() / (max_length - 1)  # [N]

    # Log frequency per token
    lf = log_freq[token_ids].float()  # [N]

    # Token-level features
    is_ws = token_features["is_word_start"][token_ids].float()
    is_pn = token_features["is_punctuation"][token_ids].float()
    t_len = token_features["token_str_length"][token_ids].float()
    is_fn = token_features["is_function_word"][token_ids].float()

    # Normalize continuous features to zero-mean unit-variance
    def normalize(x):
        m, s = x.mean(), x.std()
        return (x - m) / s.clamp(min=1e-8)

    pos_n = normalize(pos_scalar)
    lf_n = normalize(lf)
    tlen_n = normalize(t_len)

    # F1: position only
    F1 = pos_n.unsqueeze(1)  # [N, 1]

    # F2: position + log_frequency
    F2 = torch.stack([pos_n, lf_n], dim=1)  # [N, 2]

    # F3: + is_word_start
    F3 = torch.stack([pos_n, lf_n, is_ws], dim=1)  # [N, 3]

    # F4: + is_punctuation + token_strlen
    F4 = torch.stack([pos_n, lf_n, is_ws, is_pn, tlen_n], dim=1)  # [N, 5]

    # F5: + is_function_word (all features)
    F5 = torch.stack([pos_n, lf_n, is_ws, is_pn, tlen_n, is_fn], dim=1)  # [N, 6]

    # F_pos_onehot: one-hot position (64-dim) — to capture nonlinear position effects
    pos_onehot = torch.zeros(N, max_length)
    pos_onehot.scatter_(1, positions.unsqueeze(1).long(), 1.0)

    # F6: one-hot position + all scalar features
    F6 = torch.cat([pos_onehot, lf_n.unsqueeze(1), is_ws.unsqueeze(1),
                     is_pn.unsqueeze(1), tlen_n.unsqueeze(1), is_fn.unsqueeze(1)], dim=1)  # [N, 69]

    return {
        "F0_baseline": None,  # no features
        "F1_position": F1,
        "F2_pos+freq": F2,
        "F3_pos+freq+ws": F3,
        "F4_pos+freq+ws+pn+len": F4,
        "F5_all_scalar": F5,
        "F6_pos_onehot+all": F6,
    }


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------


def run_experiment(model, tokenizer, texts, layers, device, output_dir,
                   n_samples=512, k_omp=8, max_length=64):
    logger.info("Collecting activations with metadata...")
    E = get_token_embeddings(model).to(device)

    hidden_flat, positions, token_ids = collect_activations_with_metadata(
        model, tokenizer, texts, layers, device, max_length=max_length)

    # Compute token frequencies
    token_counts = torch.zeros(tokenizer.vocab_size)
    for tid in token_ids:
        token_counts[tid.item()] += 1
    token_counts = token_counts.clamp(min=1)
    log_freq = torch.log(token_counts / token_counts.sum())

    # Token features
    token_feats = build_token_features(tokenizer)

    # Build feature matrices
    feature_sets = build_feature_matrices(positions, token_ids, token_feats, log_freq, max_length)

    results = {}

    for l in layers:
        logger.info(f"=== Layer {l} ===")
        H_all = hidden_flat[l]
        N_total = H_all.shape[0]

        # Subsample
        idx = torch.randperm(N_total)[:min(n_samples, N_total)]
        H = H_all[idx].to(device)
        d = H.shape[1]

        h_norm_sq = H.pow(2).sum(dim=1).mean().item()

        layer_results = {}

        for feat_name, X_full in feature_sets.items():
            if X_full is None:
                # Baseline: OMP on raw activations
                err = omp_k_error(H, E, k=k_omp).item()
                layer_results[feat_name] = {
                    "omp_error": err,
                    "relative_error": err / h_norm_sq,
                    "n_features": 0,
                }
                logger.info(f"  {feat_name}: OMP@{k_omp} error={err:.1f}, relative={err/h_norm_sq:.4f}")
                continue

            X = X_full[idx].to(device)  # [n, n_feat]
            n_feat = X.shape[1]

            # Learn optimal linear mapping: H_hat = X @ W where W = (X^T X)^-1 X^T H
            # This is the OLS solution: minimize ||H - X @ W||^2
            # W = pinv(X) @ H
            XtX = X.T @ X  # [n_feat, n_feat]
            # Add small ridge for numerical stability
            XtX_reg = XtX + 1e-6 * torch.eye(n_feat, device=device)
            XtX_inv = torch.linalg.inv(XtX_reg)
            W = XtX_inv @ (X.T @ H)  # [n_feat, d]

            H_hat = X @ W  # [n, d] predicted (interpretable) component
            H_clean = H - H_hat  # [n, d] residual after removing interpretable features

            # Variance explained by the linear features
            var_explained = H_hat.pow(2).sum(dim=1).mean().item() / h_norm_sq

            # OMP on cleaned activations
            err_clean = omp_k_error(H_clean, E, k=k_omp).item()
            clean_norm_sq = H_clean.pow(2).sum(dim=1).mean().item()

            # Reconstruction: h ≈ H_hat + OMP_recon(H_clean)
            # Total error = OMP error on H_clean (since H_hat is exact)
            # Compare with baseline OMP error on H

            baseline_err = layer_results["F0_baseline"]["omp_error"]

            layer_results[feat_name] = {
                "omp_error": err_clean,
                "relative_error": err_clean / h_norm_sq,
                "clean_norm_sq": clean_norm_sq,
                "relative_error_of_clean": err_clean / max(clean_norm_sq, 1e-12),
                "variance_explained_by_features": var_explained,
                "error_reduction_vs_baseline": (baseline_err - err_clean) / baseline_err,
                "n_features": n_feat,
            }

            logger.info(
                f"  {feat_name} ({n_feat}d): "
                f"var_explained={var_explained:.4f}, "
                f"OMP@{k_omp} error={err_clean:.1f} "
                f"(reduction={((baseline_err - err_clean) / baseline_err)*100:.1f}%), "
                f"relative_of_clean={err_clean / max(clean_norm_sq, 1e-12):.4f}"
            )

        # Also test: OMP with more atoms (k=16, 32) on raw H as comparison
        for k_extra in [16, 32, 64]:
            err_extra = omp_k_error(H, E, k=k_extra).item()
            layer_results[f"F0_baseline_k{k_extra}"] = {
                "omp_error": err_extra,
                "relative_error": err_extra / h_norm_sq,
                "n_features": 0,
            }
            logger.info(f"  OMP@{k_extra} (raw): error={err_extra:.1f}, relative={err_extra/h_norm_sq:.4f}")

        results[f"layer_{l}"] = layer_results

    save_json(results, os.path.join(output_dir, "exp_decompose_reconstruct.json"))

    # ---- Plot ----
    feat_names = [k for k in list(results[f"layer_{layers[0]}"].keys())
                  if not k.startswith("F0_baseline_k")]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: Relative error per feature set per layer
    x = np.arange(len(layers))
    width = 0.8 / len(feat_names)
    colors = plt.cm.viridis(np.linspace(0, 1, len(feat_names)))
    for i, fn in enumerate(feat_names):
        vals = [results[f"layer_{l}"][fn]["relative_error"] for l in layers]
        axes[0].bar(x + i * width, vals, width, label=fn, color=colors[i])
    axes[0].set_xticks(x + width * len(feat_names) / 2)
    axes[0].set_xticklabels([str(l) for l in layers])
    axes[0].set_title("Relative OMP@8 error (lower = better)")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("||r||^2 / ||h||^2")
    axes[0].legend(fontsize=6, ncol=2)

    # Plot 2: Error reduction vs baseline
    for fn in feat_names:
        if fn == "F0_baseline":
            continue
        vals = [results[f"layer_{l}"][fn].get("error_reduction_vs_baseline", 0) * 100
                for l in layers]
        axes[1].plot([str(l) for l in layers], vals, "o-", label=fn, markersize=4)
    axes[1].set_title("OMP error reduction vs raw baseline (%)")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Error reduction (%)")
    axes[1].legend(fontsize=6)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Compare feature removal + OMP@8 vs raw OMP@k
    # "Is removing features + OMP@8 equivalent to just using more atoms?"
    for l in layers:
        res = results[f"layer_{l}"]
        # Feature-based results
        best_feat = min(
            [(fn, res[fn]["relative_error"]) for fn in feat_names if fn != "F0_baseline"],
            key=lambda x: x[1]
        )
        # k-scaling results
        k_vals = [8, 16, 32, 64]
        k_errs = [res["F0_baseline"]["relative_error"]]
        for k in [16, 32, 64]:
            k_key = f"F0_baseline_k{k}"
            if k_key in res:
                k_errs.append(res[k_key]["relative_error"])

        axes[2].plot(k_vals[:len(k_errs)], k_errs, "o-", label=f"L{l} OMP@k", alpha=0.5)
        axes[2].axhline(y=best_feat[1], linestyle="--", alpha=0.3)
        axes[2].plot([8], [best_feat[1]], "s", markersize=8,
                     label=f"L{l} {best_feat[0]}", alpha=0.7)

    axes[2].set_title("Feature removal + OMP@8 vs raw OMP@k")
    axes[2].set_xlabel("k (for raw OMP) or 8 (for feature-cleaned)")
    axes[2].set_ylabel("Relative error")
    axes[2].set_xscale("log", base=2)
    axes[2].legend(fontsize=5, ncol=2)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "exp_decompose_reconstruct.png"), dpi=150)
    plt.close(fig)

    # ---- Summary table ----
    logger.info("\n=== SUMMARY ===")
    logger.info(f"{'Layer':>5} | {'Baseline':>10} | {'Best feat':>10} | {'Reduction':>10} | {'Equiv k':>8}")
    logger.info("-" * 55)
    for l in layers:
        res = results[f"layer_{l}"]
        base = res["F0_baseline"]["relative_error"]
        best_fn, best_val = min(
            [(fn, res[fn]["relative_error"]) for fn in feat_names if fn != "F0_baseline"],
            key=lambda x: x[1]
        )
        # Find equivalent k
        equiv_k = ">64"
        for k in [16, 32, 64]:
            k_key = f"F0_baseline_k{k}"
            if k_key in res and res[k_key]["relative_error"] <= best_val:
                equiv_k = str(k)
                break
        reduction = (base - best_val) / base * 100
        logger.info(f"  L{l:>2} | {base:>10.4f} | {best_val:>10.4f} | {reduction:>9.1f}% | {equiv_k:>8}")

    logger.info("Done.")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=str, default="0,1,3,5,7,9,11")
    parser.add_argument("--n_samples", type=int, default=512)
    parser.add_argument("--n_texts", type=int, default=256)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="exp/nonvocab_subspace")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    layers = [int(x) for x in args.layers.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device()
    model, tokenizer = load_gpt2(device)
    texts = get_sample_texts(args.n_texts)
    logger.info(f"Device: {device}, {len(texts)} texts")

    run_experiment(model, tokenizer, texts, layers, device, args.output_dir,
                   n_samples=args.n_samples, k_omp=args.k)


if __name__ == "__main__":
    main()
