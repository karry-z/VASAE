"""
Identify WHAT the non-sparsely-representable component of GPT-2 residual streams IS.

Key insight: E [50257, 768] has rank 768, so span(E) = R^768. No direction is "outside" E.
The OMP residual is NOT orthogonal to E — it's the part of h that's DENSE in E
(requires many vocabulary vectors simultaneously, not sparse).

The question: what information in the residual stream requires dense vocabulary combinations?

Six experiments (A-F), selectable via --exp {A,B,C,D,E,F,all}.
Results (JSON + PNG) saved to --output_dir (default: exp/nonvocab_subspace/).
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_missing_components import (
    SubLayerHookCollector,
    collect_sublayer_activations,
    flatten_activations,
    get_device,
    get_pos_embeddings,
    get_sample_texts,
    get_token_embeddings,
    load_gpt2,
    omp_k_error,
    sample_flat,
    save_json,
    set_seed,
    streaming_topM_candidates,
)

logging.basicConfig(
    format="[%(levelname)s] %(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_omp_residuals(
    H: torch.Tensor,
    E: torch.Tensor,
    k: int = 8,
    M: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """OMP@k reconstruction and residuals.

    Returns:
        H_recon: [N, d] reconstructed activations
        R: [N, d] residuals (H - H_recon)
    """
    N, d = H.shape
    device = H.device
    E = E.to(device)

    cand_idx = streaming_topM_candidates(H, E, M=M)

    H_recon = torch.zeros_like(H)
    for n in range(N):
        h = H[n]
        D = E[cand_idx[n]]
        r = h.clone()
        selected = []
        for _ in range(k):
            corr = (D @ r).abs()
            j = int(torch.argmax(corr).item())
            if j in selected:
                break
            selected.append(j)
            A = D[selected].T  # [d, t]
            sol = torch.linalg.lstsq(A, h).solution
            r = h - A @ sol
        H_recon[n] = h - r

    R = H - H_recon
    return H_recon, R


@torch.no_grad()
def logit_lens_decode(
    vectors: torch.Tensor,
    W_U: torch.Tensor,
    tokenizer: GPT2TokenizerFast,
    top_n: int = 10,
) -> list[dict]:
    """Map vectors through unembedding to get top/bottom tokens."""
    vectors = vectors.to(W_U.device)
    logits = vectors @ W_U.T  # [k, V]

    results = []
    for i in range(vectors.shape[0]):
        vals, top_idx = torch.topk(logits[i], top_n)
        bot_vals, bot_idx = torch.topk(logits[i], top_n, largest=False)
        results.append({
            "top_tokens": [tokenizer.decode([idx.item()]) for idx in top_idx],
            "top_logits": vals.cpu().tolist(),
            "bottom_tokens": [tokenizer.decode([idx.item()]) for idx in bot_idx],
            "bottom_logits": bot_vals.cpu().tolist(),
        })
    return results


@torch.no_grad()
def extract_qk_matrices(model: GPT2LMHeadModel, layer: int) -> torch.Tensor:
    """Extract per-head QK matrices: W_QK^h = W_Q^h @ W_K^h^T [12, 768, 768].

    Determines what each head attends to: attn_logit = x_q @ W_QK @ x_k.
    """
    block = model.transformer.h[layer]
    n_heads = 12
    head_dim = 64
    d_model = 768

    c_attn_w = block.attn.c_attn.weight.detach().float()  # [768, 2304]
    W_Q = c_attn_w[:, :d_model]  # [768, 768]
    W_K = c_attn_w[:, d_model:2 * d_model]  # [768, 768]

    W_Q_heads = W_Q.reshape(d_model, n_heads, head_dim)  # [768, 12, 64]
    W_K_heads = W_K.reshape(d_model, n_heads, head_dim)  # [768, 12, 64]

    W_QK = torch.zeros(n_heads, d_model, d_model)
    for h in range(n_heads):
        # W_QK^h = W_Q^h @ W_K^h^T : [768, 64] @ [64, 768] = [768, 768]
        W_QK[h] = W_Q_heads[:, h, :] @ W_K_heads[:, h, :].T

    return W_QK


@torch.no_grad()
def extract_ov_matrices(model: GPT2LMHeadModel, layer: int) -> torch.Tensor:
    """Extract per-head OV matrices: W_OV^h = W_V^h @ W_O^h [12, 768, 768].

    Determines what each head writes: output = (x @ W_V) @ W_O.
    """
    block = model.transformer.h[layer]
    n_heads = 12
    head_dim = 64
    d_model = 768

    c_attn_w = block.attn.c_attn.weight.detach().float()  # [768, 2304]
    W_V = c_attn_w[:, 2 * d_model:]  # [768, 768]
    W_O = block.attn.c_proj.weight.detach().float()  # [768, 768]

    W_V_heads = W_V.reshape(d_model, n_heads, head_dim)
    W_O_heads = W_O.reshape(d_model, n_heads, head_dim).permute(1, 2, 0)

    W_OV = torch.zeros(n_heads, d_model, d_model)
    for h in range(n_heads):
        W_OV[h] = W_V_heads[:, h, :] @ W_O_heads[h]

    return W_OV


# ---------------------------------------------------------------------------
# Experiment A: Additive source decomposition of OMP residuals
# "Which sublayer's output contributes most to the non-sparse part?"
# ---------------------------------------------------------------------------


def expA_source_decomposition(
    model, tokenizer, texts, layers, device, output_dir,
    n_samples=512, k_omp=8, max_length=64,
):
    """For each layer l, h_l = wte + wpe + sum(attn_i + mlp_i for i in 0..l).
    OMP picks 8 atoms from E to approximate h. The residual r = h - OMP_recon.
    We decompose r by projecting each sublayer's ACTUAL output onto r.

    This directly answers: which sublayer is responsible for the non-sparse content?
    """
    logger.info("=== Exp A: Additive source decomposition of OMP residuals ===")
    E = get_token_embeddings(model).to(device)

    all_acts = collect_sublayer_activations(model, tokenizer, texts, layers, device)
    results = {}

    for l in layers:
        logger.info(f"  Layer {l}...")
        block_act = flatten_activations(all_acts[f"block.{l}"])

        # We need to subsample consistently across all sublayers
        N_total = block_act.shape[0]
        n = min(n_samples, N_total)
        idx = torch.randperm(N_total)[:n]

        H = block_act[idx].to(device)
        _, R = compute_omp_residuals(H, E, k=k_omp)

        # R_norm for each sample [n]
        r_norm_sq = R.pow(2).sum(dim=1)  # [n]
        total_r_norm = r_norm_sq.mean().item()

        # For each sublayer, compute: <sublayer_output, r> / ||r||^2
        # This gives "how much of r is explained by this sublayer's direction"
        contributions = {}

        for comp_name in ["wte", "wpe"]:
            if comp_name in all_acts:
                comp = flatten_activations(all_acts[comp_name])[idx].to(device)
                dot = (comp * R).sum(dim=1)  # [n] signed dot product
                frac = (dot / r_norm_sq.clamp(min=1e-12)).mean().item()
                contributions[comp_name] = {
                    "signed_fraction": frac,
                    "mean_dot_product": dot.mean().item(),
                    "comp_norm": comp.pow(2).sum(dim=1).mean().item(),
                }

        for ll in range(l + 1):
            for sub in ["attn", "mlp"]:
                comp_key = f"{sub}.{ll}"
                if comp_key in all_acts:
                    comp = flatten_activations(all_acts[comp_key])[idx].to(device)
                    dot = (comp * R).sum(dim=1)
                    frac = (dot / r_norm_sq.clamp(min=1e-12)).mean().item()
                    contributions[comp_key] = {
                        "signed_fraction": frac,
                        "mean_dot_product": dot.mean().item(),
                        "comp_norm": comp.pow(2).sum(dim=1).mean().item(),
                    }

        # Verify: sum of all fractions should ≈ 1 (since h = sum of components and r = h - recon)
        sum_frac = sum(v["signed_fraction"] for v in contributions.values())

        # Also compute OMP_recon's dot with r (should be ~0 since OMP minimizes ||r||)
        H_recon = H - R
        recon_dot_r = (H_recon * R).sum(dim=1).mean().item()

        results[f"layer_{l}"] = {
            "residual_norm": total_r_norm,
            "component_contributions": contributions,
            "sum_of_fractions": sum_frac,
            "recon_dot_residual": recon_dot_r,
        }

        # Log top contributors
        sorted_comps = sorted(contributions.items(),
                              key=lambda x: abs(x[1]["signed_fraction"]), reverse=True)
        logger.info(f"    Residual norm: {total_r_norm:.2f}")
        logger.info(f"    Sum of fractions: {sum_frac:.4f} (should ≈ 1)")
        for name, info in sorted_comps[:5]:
            logger.info(f"    {name}: fraction={info['signed_fraction']:.4f}")

    save_json(results, os.path.join(output_dir, "expA_source_decomposition.json"))

    # Plot: stacked bar chart of source contributions per layer
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for l in layers:
        comps = results[f"layer_{l}"]["component_contributions"]
        # Group: wte, wpe, attn (sum), mlp (sum) per source layer
        wte_frac = comps.get("wte", {}).get("signed_fraction", 0)
        wpe_frac = comps.get("wpe", {}).get("signed_fraction", 0)
        attn_fracs = {}
        mlp_fracs = {}
        for k, v in comps.items():
            if k.startswith("attn."):
                ll = int(k.split(".")[1])
                attn_fracs[ll] = v["signed_fraction"]
            elif k.startswith("mlp."):
                ll = int(k.split(".")[1])
                mlp_fracs[ll] = v["signed_fraction"]

        # Bar chart for this layer
        names = ["wte", "wpe"]
        vals = [wte_frac, wpe_frac]
        for ll in sorted(set(list(attn_fracs.keys()) + list(mlp_fracs.keys()))):
            names.append(f"attn.{ll}")
            vals.append(attn_fracs.get(ll, 0))
            names.append(f"mlp.{ll}")
            vals.append(mlp_fracs.get(ll, 0))

        x_pos = [str(l)] * len(names)

    # Simplified: for each layer, show grouped contributions
    all_comp_names = set()
    for l in layers:
        all_comp_names.update(results[f"layer_{l}"]["component_contributions"].keys())
    comp_list = sorted(all_comp_names)

    # Positive and negative contributions
    pos_data = {c: [] for c in comp_list}
    neg_data = {c: [] for c in comp_list}
    for l in layers:
        comps = results[f"layer_{l}"]["component_contributions"]
        for c in comp_list:
            f = comps.get(c, {}).get("signed_fraction", 0)
            pos_data[c].append(max(f, 0))
            neg_data[c].append(min(f, 0))

    x = np.arange(len(layers))
    # Only plot top contributors (abs > 0.05 for any layer)
    important_comps = [c for c in comp_list
                       if max(max(pos_data[c]), -min(neg_data[c])) > 0.02]

    bottom_pos = np.zeros(len(layers))
    bottom_neg = np.zeros(len(layers))
    colors = plt.cm.tab20(np.linspace(0, 1, len(important_comps)))
    for i, c in enumerate(important_comps):
        pv = np.array(pos_data[c])
        nv = np.array(neg_data[c])
        if pv.any():
            axes[0].bar(x, pv, bottom=bottom_pos, label=c, color=colors[i], width=0.6)
            bottom_pos += pv
        if nv.any():
            axes[0].bar(x, nv, bottom=bottom_neg, color=colors[i], width=0.6)
            bottom_neg += nv

    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(l) for l in layers])
    axes[0].set_title("Source contributions to OMP residual (signed fraction)")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Fraction of ||r||^2")
    axes[0].legend(fontsize=6, ncol=2, loc="upper left")
    axes[0].axhline(y=0, color="black", linewidth=0.5)

    # Right: residual norms
    r_norms = [results[f"layer_{l}"]["residual_norm"] for l in layers]
    axes[1].bar(x, r_norms)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(l) for l in layers])
    axes[1].set_title("OMP residual norm (mean squared)")
    axes[1].set_xlabel("Layer")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "expA_source_decomposition.png"), dpi=150)
    plt.close(fig)

    logger.info("Experiment A done.")
    return results


# ---------------------------------------------------------------------------
# Experiment B: Dense vocabulary structure of the residual
# "When we decode the residual through E, what pattern do the 50k coefficients form?"
# ---------------------------------------------------------------------------


def expB_dense_vocab_structure(
    model, tokenizer, texts, layers, device, output_dir,
    n_samples=512, k_omp=8, top_n=20,
):
    """The OMP residual r lives in span(E) but is dense. Project r onto all 50k
    vocab directions: c = E @ (E^T E)^-1 @ r (least-squares coefficients).

    Analyze: are coefficients uniform? Semantically clustered? What word categories
    dominate the dense component?
    """
    logger.info("=== Exp B: Dense vocabulary structure of OMP residuals ===")
    E = get_token_embeddings(model).to(device)  # [V, d]
    V, d = E.shape

    # Precompute pseudo-inverse projection: (E^T E)^-1 E^T
    EtE_inv = torch.linalg.inv(E.T @ E)  # [d, d]
    # For vector r: coefficients c = E @ EtE_inv @ r  [V]

    all_acts = collect_sublayer_activations(model, tokenizer, texts, layers, device)
    results = {}

    for l in layers:
        logger.info(f"  Layer {l}...")
        block_act = flatten_activations(all_acts[f"block.{l}"])
        H = sample_flat(block_act, n_samples).to(device)
        _, R = compute_omp_residuals(H, E, k=k_omp)

        # For each residual, compute dense coefficients
        # c_i = E @ EtE_inv @ r_i  [V]
        # Too large to store all. Compute statistics over samples.
        n_analyze = min(128, R.shape[0])

        # Accumulate: mean |c|, top tokens, coefficient distribution stats
        mean_abs_c = torch.zeros(V, device=device)
        mean_c = torch.zeros(V, device=device)
        top_pos_tokens_counter = {}  # token_id -> count of being in top-k positive
        top_neg_tokens_counter = {}

        gini_list = []
        entropy_list = []

        for i in range(n_analyze):
            c = E @ (EtE_inv @ R[i])  # [V]
            mean_abs_c += c.abs()
            mean_c += c

            c_abs = c.abs()
            # Gini coefficient
            sorted_c, _ = c_abs.sort()
            index = torch.arange(1, V + 1, device=device, dtype=torch.float32)
            gini = (2 * (index * sorted_c).sum() / (V * sorted_c.sum().clamp(min=1e-12)) - (V + 1) / V).item()
            gini_list.append(gini)

            # Entropy of |c| distribution (normalized)
            p = c_abs / c_abs.sum().clamp(min=1e-12)
            ent = -(p * (p + 1e-12).log()).sum().item()
            entropy_list.append(ent)

            # Top positive and negative coefficient tokens
            _, top_pos = torch.topk(c, top_n)
            _, top_neg = torch.topk(c, top_n, largest=False)
            for idx in top_pos.cpu().tolist():
                top_pos_tokens_counter[idx] = top_pos_tokens_counter.get(idx, 0) + 1
            for idx in top_neg.cpu().tolist():
                top_neg_tokens_counter[idx] = top_neg_tokens_counter.get(idx, 0) + 1

        mean_abs_c /= n_analyze
        mean_c /= n_analyze

        # Most consistently important tokens across samples
        _, consistent_top = torch.topk(mean_abs_c, top_n)
        consistent_tokens = [
            {"token": tokenizer.decode([idx.item()]),
             "token_id": idx.item(),
             "mean_abs_coeff": mean_abs_c[idx].item(),
             "mean_signed_coeff": mean_c[idx].item()}
            for idx in consistent_top
        ]

        # Most frequently appearing in per-sample top-k
        freq_pos = sorted(top_pos_tokens_counter.items(), key=lambda x: x[1], reverse=True)[:top_n]
        freq_neg = sorted(top_neg_tokens_counter.items(), key=lambda x: x[1], reverse=True)[:top_n]

        results[f"layer_{l}"] = {
            "gini_mean": float(np.mean(gini_list)),
            "gini_std": float(np.std(gini_list)),
            "entropy_mean": float(np.mean(entropy_list)),
            "entropy_max_possible": float(np.log(V)),
            "entropy_ratio": float(np.mean(entropy_list) / np.log(V)),
            "consistent_top_tokens": consistent_tokens,
            "frequent_positive_tokens": [
                {"token": tokenizer.decode([tid]), "token_id": tid, "frequency": cnt}
                for tid, cnt in freq_pos
            ],
            "frequent_negative_tokens": [
                {"token": tokenizer.decode([tid]), "token_id": tid, "frequency": cnt}
                for tid, cnt in freq_neg
            ],
        }

        logger.info(f"    Gini: {np.mean(gini_list):.4f}, Entropy ratio: {np.mean(entropy_list)/np.log(V):.4f}")
        logger.info(f"    Consistent top tokens: {[t['token'] for t in consistent_tokens[:10]]}")

    save_json(results, os.path.join(output_dir, "expB_dense_vocab_structure.json"))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    layer_labels = [str(l) for l in layers]
    ginis = [results[f"layer_{l}"]["gini_mean"] for l in layers]
    ent_ratios = [results[f"layer_{l}"]["entropy_ratio"] for l in layers]

    axes[0].bar(layer_labels, ginis)
    axes[0].set_title("Gini coefficient of dense residual coefficients")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Gini (1=sparse, 0=uniform)")

    axes[1].bar(layer_labels, ent_ratios)
    axes[1].set_title("Entropy ratio of |coefficients| (1=uniform)")
    axes[1].set_xlabel("Layer")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "expB_dense_vocab_structure.png"), dpi=150)
    plt.close(fig)

    logger.info("Experiment B done.")
    return results


# ---------------------------------------------------------------------------
# Experiment C: Downstream consumption — QK readout analysis
# "Which attention heads READ from the non-sparse subspace?"
# ---------------------------------------------------------------------------


def expC_downstream_readers(
    model, tokenizer, texts, layers, device, output_dir,
    n_samples=512, k_omp=8, n_pcs=16,
):
    """For each downstream head's QK circuit, measure how much attention depends
    on the non-sparse subspace of the residual stream.

    If a direction is an "attention routing signal," downstream W_QK will have
    high eigenvalues along that direction.
    """
    logger.info("=== Exp C: Downstream QK readout of non-sparse subspace ===")
    E = get_token_embeddings(model).to(device)

    all_acts = collect_sublayer_activations(model, tokenizer, texts, layers, device)
    results = {}

    for l in layers:
        logger.info(f"  Layer {l}...")
        block_act = flatten_activations(all_acts[f"block.{l}"])
        H = sample_flat(block_act, n_samples).to(device)

        # PCA of residuals to get the non-sparse subspace
        _, R = compute_omp_residuals(H, E, k=k_omp)
        Rc = R - R.mean(0, keepdim=True)
        _, S_r, Vh_r = torch.linalg.svd(Rc, full_matrices=False)
        residual_pcs = Vh_r[:n_pcs]  # [n_pcs, d]

        # Also get the "sparse subspace" = top PCA of OMP reconstruction
        H_recon = H - R
        Hrc = H_recon - H_recon.mean(0, keepdim=True)
        _, _, Vh_recon = torch.linalg.svd(Hrc, full_matrices=False)
        recon_pcs = Vh_recon[:n_pcs]  # [n_pcs, d]

        # For each downstream layer's heads, compute readout strength
        downstream_readout = {}
        for dl in range(l + 1, 12):
            W_QK = extract_qk_matrices(model, dl).to(device)  # [12, 768, 768]
            for h in range(12):
                # How much does this head's attention pattern depend on residual PCs?
                # Readout strength = ||W_QK^h @ pc||^2 for each pc
                # This measures: if the key has component along pc, how much does it
                # affect the attention logit?
                res_readout = []
                recon_readout = []
                for pc_idx in range(n_pcs):
                    # As query: how much does query along pc attend?
                    q_strength = (W_QK[h] @ residual_pcs[pc_idx]).pow(2).sum().item()
                    # As key: how much is this pc attended to?
                    k_strength = (W_QK[h].T @ residual_pcs[pc_idx]).pow(2).sum().item()
                    res_readout.append({"as_query": q_strength, "as_key": k_strength})

                    q_str_r = (W_QK[h] @ recon_pcs[pc_idx]).pow(2).sum().item()
                    k_str_r = (W_QK[h].T @ recon_pcs[pc_idx]).pow(2).sum().item()
                    recon_readout.append({"as_query": q_str_r, "as_key": k_str_r})

                # Aggregate: total readout from non-sparse vs sparse subspace
                total_res = sum(r["as_query"] + r["as_key"] for r in res_readout)
                total_recon = sum(r["as_query"] + r["as_key"] for r in recon_readout)

                downstream_readout[f"L{dl}_H{h}"] = {
                    "residual_total": total_res,
                    "recon_total": total_recon,
                    "ratio": total_res / max(total_recon, 1e-12),
                }

        # Find heads that disproportionately read from non-sparse subspace
        if downstream_readout:
            sorted_heads = sorted(downstream_readout.items(),
                                  key=lambda x: x[1]["ratio"], reverse=True)
            top_readers = sorted_heads[:10]
        else:
            top_readers = []

        results[f"layer_{l}"] = {
            "downstream_readout": downstream_readout,
            "top_nonsparse_readers": [
                {"head": h, "ratio": v["ratio"],
                 "residual_total": v["residual_total"],
                 "recon_total": v["recon_total"]}
                for h, v in top_readers
            ],
        }

        for h, v in top_readers[:5]:
            logger.info(f"    {h}: nonsparse/sparse ratio = {v['ratio']:.3f}")

    save_json(results, os.path.join(output_dir, "expC_downstream_readers.json"))

    # Plot: heatmap of readout ratio per (source_layer, downstream_head)
    for l in layers:
        dr = results[f"layer_{l}"]["downstream_readout"]
        if not dr:
            continue

        # Build matrix: rows = downstream layers, cols = heads
        dl_range = list(range(l + 1, 12))
        if not dl_range:
            continue
        matrix = np.zeros((len(dl_range), 12))
        for i, dl in enumerate(dl_range):
            for h in range(12):
                key = f"L{dl}_H{h}"
                if key in dr:
                    matrix[i, h] = dr[key]["ratio"]

        fig, ax = plt.subplots(figsize=(12, max(4, len(dl_range) * 0.5)))
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(12))
        ax.set_xticklabels([f"H{h}" for h in range(12)])
        ax.set_yticks(range(len(dl_range)))
        ax.set_yticklabels([f"L{dl}" for dl in dl_range])
        ax.set_title(f"Layer {l}: downstream QK readout ratio (nonsparse/sparse)")
        ax.set_xlabel("Head")
        ax.set_ylabel("Downstream layer")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"expC_readout_L{l}.png"), dpi=150)
        plt.close(fig)

    logger.info("Experiment C done.")
    return results


# ---------------------------------------------------------------------------
# Experiment D: Linguistic feature probing
# "What concrete linguistic features does the non-sparse component encode?"
# ---------------------------------------------------------------------------


def expD_linguistic_probes(
    model, tokenizer, texts, layers, device, output_dir,
    n_samples=512, k_omp=8, n_pcs=8, max_length=64,
):
    """Probe the OMP residual for concrete linguistic features:
    - Position in sequence
    - Token unigram log-frequency
    - Is function word (determiner, preposition, conjunction, etc.)
    - Is punctuation
    - Is capitalized (starts a sentence?)
    - Is subword continuation (starts with 'Ġ' in GPT-2 tokenizer)
    - Token string length (proxy for morphological complexity)
    """
    logger.info("=== Exp D: Linguistic feature probing of OMP residuals ===")
    E = get_token_embeddings(model).to(device)

    # Build token-level feature lookups
    logger.info("  Building token feature tables...")
    vocab_size = tokenizer.vocab_size

    # Function words (rough heuristic list for English)
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

    # Per-token features
    is_function = torch.zeros(vocab_size)
    is_punct = torch.zeros(vocab_size)
    is_word_start = torch.zeros(vocab_size)  # starts with Ġ (space) in GPT-2
    token_strlen = torch.zeros(vocab_size)

    for tid in range(vocab_size):
        tok_str = tokenizer.decode([tid])
        tok_clean = tok_str.strip().lower()

        is_function[tid] = 1.0 if tok_clean in function_words else 0.0
        is_punct[tid] = 1.0 if (len(tok_clean) > 0 and all(c in punct_chars for c in tok_clean)) else 0.0

        # GPT-2 uses Ġ (byte \xc4\xa0) for space prefix = word start
        raw_token = tokenizer.convert_ids_to_tokens(tid)
        is_word_start[tid] = 1.0 if (raw_token and raw_token.startswith("Ġ")) else 0.0
        token_strlen[tid] = len(tok_str)

    # Compute token log-frequencies from corpus
    logger.info("  Collecting activations with metadata...")
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

    # Token log-frequencies
    token_counts = torch.zeros(vocab_size)
    for tid in token_ids_flat:
        token_counts[tid.item()] += 1
    token_counts = token_counts.clamp(min=1)
    log_freq = torch.log(token_counts / token_counts.sum())

    # Feature vectors for each token position
    feat_position = positions_flat.float()
    feat_logfreq = log_freq[token_ids_flat].float()
    feat_is_function = is_function[token_ids_flat].float()
    feat_is_punct = is_punct[token_ids_flat].float()
    feat_is_word_start = is_word_start[token_ids_flat].float()
    feat_strlen = token_strlen[token_ids_flat].float()

    all_features = {
        "position": feat_position,
        "log_frequency": feat_logfreq,
        "is_function_word": feat_is_function,
        "is_punctuation": feat_is_punct,
        "is_word_start": feat_is_word_start,
        "token_str_length": feat_strlen,
    }

    results = {}

    for l in layers:
        logger.info(f"  Layer {l}...")
        H_all = torch.cat(all_hidden[l], dim=0)

        n_total = H_all.shape[0]
        if n_total > n_samples:
            idx = torch.randperm(n_total)[:n_samples]
        else:
            idx = torch.arange(n_total)

        H = H_all[idx].to(device)

        # Compute OMP residuals
        _, R = compute_omp_residuals(H, E, k=k_omp)

        # PCA of residual to get main directions
        Rc = R - R.mean(0, keepdim=True)
        _, _, Vh_r = torch.linalg.svd(Rc, full_matrices=False)
        residual_pcs = Vh_r[:n_pcs]

        # Project onto residual PCs
        proj_residual = (H @ residual_pcs.T).cpu()  # [n, n_pcs]

        # Also: project onto full residual (not just PCs) for aggregate probing
        # Use residual norm as aggregate feature
        residual_norm = R.pow(2).sum(dim=1).cpu().float()

        layer_results = {"per_pc": {}, "aggregate": {}}

        # Probe each feature against each PC
        for feat_name, feat_vals in all_features.items():
            feat_sub = feat_vals[idx].numpy()

            # Per-PC correlations
            for pc_idx in range(n_pcs):
                proj = proj_residual[:, pc_idx].numpy()
                if feat_name in ("is_function_word", "is_punctuation", "is_word_start"):
                    # Binary feature: use point-biserial correlation (= Pearson)
                    if feat_sub.std() < 1e-8:
                        r_val, p_val = 0.0, 1.0
                    else:
                        r_val, p_val = stats.pearsonr(feat_sub, proj)
                else:
                    r_val, p_val = stats.pearsonr(feat_sub, proj)

                layer_results["per_pc"].setdefault(f"PC{pc_idx}", {})[feat_name] = {
                    "pearson_r": float(r_val),
                    "p_value": float(p_val),
                }

            # Aggregate: correlation with residual norm
            r_val, p_val = stats.pearsonr(feat_sub, residual_norm.numpy())
            layer_results["aggregate"][feat_name] = {
                "pearson_r_with_residual_norm": float(r_val),
                "p_value": float(p_val),
            }

        results[f"layer_{l}"] = layer_results

        # Log summary
        logger.info(f"    Aggregate correlations with residual norm:")
        for feat_name, info in layer_results["aggregate"].items():
            r = info["pearson_r_with_residual_norm"]
            logger.info(f"      {feat_name}: r={r:.3f}")
        # Log strongest per-PC correlation
        for pc_idx in range(min(3, n_pcs)):
            pc_data = layer_results["per_pc"][f"PC{pc_idx}"]
            best_feat = max(pc_data.items(), key=lambda x: abs(x[1]["pearson_r"]))
            logger.info(f"    PC{pc_idx} best: {best_feat[0]} r={best_feat[1]['pearson_r']:.3f}")

    save_json(results, os.path.join(output_dir, "expD_linguistic_probes.json"))

    # Plot: heatmap of correlations
    feat_names = list(all_features.keys())
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: per-PC heatmap (aggregate across layers, show strongest)
    # Actually: show per-layer aggregate correlations
    agg_matrix = np.zeros((len(feat_names), len(layers)))
    for j, l in enumerate(layers):
        for i, fn in enumerate(feat_names):
            agg_matrix[i, j] = results[f"layer_{l}"]["aggregate"][fn]["pearson_r_with_residual_norm"]

    im0 = axes[0].imshow(agg_matrix, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    axes[0].set_xticks(range(len(layers)))
    axes[0].set_xticklabels([str(l) for l in layers])
    axes[0].set_yticks(range(len(feat_names)))
    axes[0].set_yticklabels(feat_names)
    axes[0].set_title("Feature correlation with residual norm")
    for i in range(len(feat_names)):
        for j in range(len(layers)):
            axes[0].text(j, i, f"{agg_matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im0, ax=axes[0])

    # Right: per-PC for a deep layer (e.g., max layer)
    deep_l = layers[-1]
    pc_matrix = np.zeros((len(feat_names), n_pcs))
    for pc_idx in range(n_pcs):
        for i, fn in enumerate(feat_names):
            pc_matrix[i, pc_idx] = results[f"layer_{deep_l}"]["per_pc"][f"PC{pc_idx}"][fn]["pearson_r"]

    im1 = axes[1].imshow(pc_matrix, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    axes[1].set_xticks(range(n_pcs))
    axes[1].set_xticklabels([f"PC{i}" for i in range(n_pcs)])
    axes[1].set_yticks(range(len(feat_names)))
    axes[1].set_yticklabels(feat_names)
    axes[1].set_title(f"Layer {deep_l}: per-PC feature correlations")
    for i in range(len(feat_names)):
        for j in range(n_pcs):
            axes[1].text(j, i, f"{pc_matrix[i, j]:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "expD_linguistic_probes.png"), dpi=150)
    plt.close(fig)

    logger.info("Experiment D done.")
    return results


# ---------------------------------------------------------------------------
# Experiment E: Write-read circuit identification
# "Which upstream component WROTE it, and which downstream component READS it?"
# ---------------------------------------------------------------------------


def expE_write_read_circuits(
    model, tokenizer, texts, layers, device, output_dir,
    n_samples=512, k_omp=8, n_pcs=8, ov_rank=8,
):
    """Match upstream writers with downstream readers through the non-sparse subspace.

    For each OMP residual PC:
    1. Find which upstream OV/MLP writes it (top projection onto OV output space)
    2. Find which downstream QK reads it (top QK eigenvalue along that direction)
    3. Report matched write→read circuits
    """
    logger.info("=== Exp E: Write-read circuit identification ===")
    E = get_token_embeddings(model).to(device)

    all_acts = collect_sublayer_activations(model, tokenizer, texts, layers, device)
    results = {}

    for l in layers:
        logger.info(f"  Layer {l}...")
        block_act = flatten_activations(all_acts[f"block.{l}"])
        H = sample_flat(block_act, n_samples).to(device)

        _, R = compute_omp_residuals(H, E, k=k_omp)
        Rc = R - R.mean(0, keepdim=True)
        _, _, Vh_r = torch.linalg.svd(Rc, full_matrices=False)
        residual_pcs = Vh_r[:n_pcs].to(device)  # [n_pcs, d]

        circuits = {}
        for pc_idx in range(n_pcs):
            pc = residual_pcs[pc_idx]  # [d]

            # --- Find writers: which upstream OV/MLP has output space aligned with pc ---
            writer_scores = []

            for ul in range(l + 1):
                # Attention OV
                W_OV = extract_ov_matrices(model, ul).to(device)
                for h in range(12):
                    # Output space of OV = column space of W_OV
                    # Projection of pc onto column space: use SVD
                    U_ov, _, _ = torch.linalg.svd(W_OV[h], full_matrices=False)
                    write_dirs = U_ov[:, :ov_rank]  # [768, ov_rank]
                    proj = (write_dirs.T @ pc).pow(2).sum().item()
                    writer_scores.append((f"L{ul}_H{h}_OV", proj))

                # MLP
                W_out = model.transformer.h[ul].mlp.c_proj.weight.detach().float().to(device)
                # Output space = row space of W_out. Use SVD.
                _, _, Vh_mlp = torch.linalg.svd(W_out, full_matrices=False)
                mlp_dirs = Vh_mlp[:ov_rank].T  # [768, ov_rank]
                proj = (mlp_dirs.T @ pc).pow(2).sum().item()
                writer_scores.append((f"L{ul}_MLP", proj))

            writer_scores.sort(key=lambda x: x[1], reverse=True)

            # --- Find readers: which downstream QK reads this direction ---
            reader_scores = []

            for dl in range(l + 1, 12):
                W_QK = extract_qk_matrices(model, dl).to(device)
                for h in range(12):
                    # How much does QK depend on this direction?
                    # As key: k_readout = ||W_QK^T @ pc||^2
                    k_readout = (W_QK[h].T @ pc).pow(2).sum().item()
                    # As query: q_readout = ||W_QK @ pc||^2
                    q_readout = (W_QK[h] @ pc).pow(2).sum().item()
                    reader_scores.append((f"L{dl}_H{h}_QK", q_readout + k_readout))

            reader_scores.sort(key=lambda x: x[1], reverse=True)

            circuits[f"PC{pc_idx}"] = {
                "top_writers": [{"component": w, "projection": p} for w, p in writer_scores[:5]],
                "top_readers": [{"component": r, "readout": s} for r, s in reader_scores[:5]],
            }

            top_w = writer_scores[0] if writer_scores else ("none", 0)
            top_r = reader_scores[0] if reader_scores else ("none", 0)
            logger.info(f"    PC{pc_idx}: writer={top_w[0]}({top_w[1]:.3f}) -> reader={top_r[0]}({top_r[1]:.3f})")

        results[f"layer_{l}"] = {"circuits": circuits}

    save_json(results, os.path.join(output_dir, "expE_write_read_circuits.json"))

    # Plot: for each analyzed layer, show the write-read circuit diagram as a table
    for l in layers:
        circ = results[f"layer_{l}"]["circuits"]
        n_pcs_actual = len(circ)

        fig, ax = plt.subplots(figsize=(14, max(4, n_pcs_actual * 0.6)))
        ax.axis("off")
        table_data = []
        for pc_idx in range(n_pcs_actual):
            pc_key = f"PC{pc_idx}"
            writers = circ[pc_key]["top_writers"][:3]
            readers = circ[pc_key]["top_readers"][:3]
            w_str = ", ".join(f"{w['component']}({w['projection']:.2f})" for w in writers)
            r_str = ", ".join(f"{r['component']}({r['readout']:.2f})" for r in readers)
            table_data.append([pc_key, w_str, r_str])

        table = ax.table(
            cellText=table_data,
            colLabels=["PC", "Top Writers", "Top Readers"],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.auto_set_column_width([0, 1, 2])
        ax.set_title(f"Layer {l}: Write→Read circuits through non-sparse subspace", fontsize=10, pad=20)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"expE_circuits_L{l}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info("Experiment E done.")
    return results


# ---------------------------------------------------------------------------
# Experiment F: Ablation impact — what breaks when we remove the non-sparse part?
# ---------------------------------------------------------------------------


def expF_ablation_impact(
    model, tokenizer, texts, layers, device, output_dir,
    n_samples=512, k_omp=8, n_pcs=8, max_length=64,
):
    """Remove the non-sparse component and measure impact on:
    1. Logit lens top-1 prediction (does next-token prediction change?)
    2. Logit lens entropy (does prediction confidence change?)
    3. Rank of correct token (does the correct token become less likely?)

    Compare: ablating non-sparse PCs vs ablating top activation PCs (highest variance).
    The contrast tells us: is the non-sparse part more about prediction or about
    something else (routing, bookkeeping)?
    """
    logger.info("=== Exp F: Ablation impact on model predictions ===")
    E = get_token_embeddings(model).to(device)
    W_U = model.lm_head.weight.detach().float().to(device)

    logger.info("  Collecting activations...")
    all_hidden = {l: [] for l in layers}
    for text in texts:
        tokens = tokenizer(
            text, return_tensors="pt", max_length=max_length,
            truncation=True, padding="max_length",
        ).to(device)
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        for l in layers:
            h = outputs.hidden_states[l + 1].squeeze(0).float().cpu()
            all_hidden[l].append(h)

    results = {}

    for l in layers:
        logger.info(f"  Layer {l}...")
        H_all = torch.cat(all_hidden[l], dim=0)
        n_total = H_all.shape[0]
        idx = torch.randperm(n_total)[:min(n_samples, n_total)]
        H = H_all[idx].to(device)

        # Compute non-sparse PCs
        _, R = compute_omp_residuals(H, E, k=k_omp)
        Rc = R - R.mean(0, keepdim=True)
        _, _, Vh_r = torch.linalg.svd(Rc, full_matrices=False)
        residual_pcs = Vh_r[:n_pcs]

        # Compute activation PCs (highest variance directions)
        Hc = H - H.mean(0, keepdim=True)
        _, _, Vh_act = torch.linalg.svd(Hc, full_matrices=False)
        act_pcs = Vh_act[:n_pcs]

        # Original predictions
        logits_orig = H @ W_U.T  # [n, V]
        pred_orig = logits_orig.argmax(dim=-1)
        prob_orig = logits_orig.softmax(dim=-1)
        entropy_orig = -(prob_orig * (prob_orig + 1e-12).log()).sum(dim=-1)

        ablation_results = {}
        for k_ablate in [1, 2, 4, 8]:
            if k_ablate > n_pcs:
                break

            # --- Ablate non-sparse PCs ---
            res_dirs = residual_pcs[:k_ablate]
            H_abl_res = H - (H @ res_dirs.T) @ res_dirs
            logits_res = H_abl_res @ W_U.T
            pred_res = logits_res.argmax(dim=-1)
            prob_res = logits_res.softmax(dim=-1)
            entropy_res = -(prob_res * (prob_res + 1e-12).log()).sum(dim=-1)

            # How many predictions changed?
            pred_change_res = (pred_res != pred_orig).float().mean().item()
            # How much did entropy change?
            entropy_change_res = (entropy_res - entropy_orig).mean().item()
            # KL divergence from original
            kl_res = F.kl_div(logits_res.log_softmax(dim=-1), prob_orig, reduction="batchmean").item()

            # --- Ablate activation PCs ---
            act_dirs = act_pcs[:k_ablate]
            H_abl_act = H - (H @ act_dirs.T) @ act_dirs
            logits_act = H_abl_act @ W_U.T
            pred_act = logits_act.argmax(dim=-1)
            prob_act = logits_act.softmax(dim=-1)
            entropy_act = -(prob_act * (prob_act + 1e-12).log()).sum(dim=-1)

            pred_change_act = (pred_act != pred_orig).float().mean().item()
            entropy_change_act = (entropy_act - entropy_orig).mean().item()
            kl_act = F.kl_div(logits_act.log_softmax(dim=-1), prob_orig, reduction="batchmean").item()

            ablation_results[f"k={k_ablate}"] = {
                "nonsparse_pcs": {
                    "prediction_change_rate": pred_change_res,
                    "entropy_change": entropy_change_res,
                    "kl_divergence": kl_res,
                },
                "activation_pcs": {
                    "prediction_change_rate": pred_change_act,
                    "entropy_change": entropy_change_act,
                    "kl_divergence": kl_act,
                },
            }

        results[f"layer_{l}"] = ablation_results

        # Log
        for k_str, abl in ablation_results.items():
            ns = abl["nonsparse_pcs"]
            ac = abl["activation_pcs"]
            logger.info(f"    {k_str}: nonsparse pred_change={ns['prediction_change_rate']:.3f}, "
                        f"KL={ns['kl_divergence']:.3f} | "
                        f"act_pc pred_change={ac['prediction_change_rate']:.3f}, "
                        f"KL={ac['kl_divergence']:.3f}")

    save_json(results, os.path.join(output_dir, "expF_ablation_impact.json"))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    k_vals = [1, 2, 4, 8]

    for l in layers:
        abl = results[f"layer_{l}"]
        ks = [k for k in k_vals if f"k={k}" in abl]

        pred_ns = [abl[f"k={k}"]["nonsparse_pcs"]["prediction_change_rate"] for k in ks]
        pred_ac = [abl[f"k={k}"]["activation_pcs"]["prediction_change_rate"] for k in ks]
        axes[0].plot(ks, pred_ns, "o-", label=f"L{l} nonsparse", alpha=0.7)
        axes[0].plot(ks, pred_ac, "s--", label=f"L{l} act_pc", alpha=0.4)

    axes[0].set_title("Prediction change rate after ablation")
    axes[0].set_xlabel("PCs removed")
    axes[0].set_ylabel("Fraction of predictions changed")
    axes[0].legend(fontsize=6, ncol=2)

    for l in layers:
        abl = results[f"layer_{l}"]
        ks = [k for k in k_vals if f"k={k}" in abl]
        kl_ns = [abl[f"k={k}"]["nonsparse_pcs"]["kl_divergence"] for k in ks]
        kl_ac = [abl[f"k={k}"]["activation_pcs"]["kl_divergence"] for k in ks]
        axes[1].plot(ks, kl_ns, "o-", label=f"L{l} nonsparse", alpha=0.7)
        axes[1].plot(ks, kl_ac, "s--", label=f"L{l} act_pc", alpha=0.4)

    axes[1].set_title("KL divergence after ablation")
    axes[1].set_xlabel("PCs removed")
    axes[1].set_ylabel("KL(original || ablated)")
    axes[1].legend(fontsize=6, ncol=2)

    for l in layers:
        abl = results[f"layer_{l}"]
        ks = [k for k in k_vals if f"k={k}" in abl]
        ent_ns = [abl[f"k={k}"]["nonsparse_pcs"]["entropy_change"] for k in ks]
        ent_ac = [abl[f"k={k}"]["activation_pcs"]["entropy_change"] for k in ks]
        axes[2].plot(ks, ent_ns, "o-", label=f"L{l} nonsparse", alpha=0.7)
        axes[2].plot(ks, ent_ac, "s--", label=f"L{l} act_pc", alpha=0.4)

    axes[2].set_title("Entropy change after ablation")
    axes[2].set_xlabel("PCs removed")
    axes[2].set_ylabel("Entropy change (+ = less confident)")
    axes[2].legend(fontsize=6, ncol=2)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "expF_ablation_impact.png"), dpi=150)
    plt.close(fig)

    logger.info("Experiment F done.")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Identify WHAT the non-sparsely-representable component of GPT-2 residual streams IS"
    )
    parser.add_argument(
        "--exp", type=str, default="A",
        help="Experiment(s) to run: A,B,C,D,E,F or 'all'",
    )
    parser.add_argument("--layers", type=str, default="0,3,5,7,9,11",
                        help="Comma-separated layer indices")
    parser.add_argument("--n_samples", type=int, default=512,
                        help="Number of activation vectors to sample per layer")
    parser.add_argument("--n_texts", type=int, default=256,
                        help="Number of text samples to run through GPT-2")
    parser.add_argument("--k", type=int, default=8,
                        help="Sparsity level for OMP")
    parser.add_argument("--n_pcs", type=int, default=8,
                        help="Number of principal components to analyze")
    parser.add_argument("--output_dir", type=str, default="exp/nonvocab_subspace",
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    layers = [int(x) for x in args.layers.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device()
    logger.info(f"Device: {device}")

    model, tokenizer = load_gpt2(device)
    logger.info("GPT-2 loaded")

    texts = get_sample_texts(args.n_texts)
    logger.info(f"Loaded {len(texts)} texts")

    exps_to_run = (["A", "B", "C", "D", "E", "F"] if args.exp.lower() == "all"
                   else [x.strip().upper() for x in args.exp.split(",")])

    exp_funcs = {
        "A": lambda: expA_source_decomposition(
            model, tokenizer, texts, layers, device, args.output_dir,
            n_samples=args.n_samples, k_omp=args.k,
        ),
        "B": lambda: expB_dense_vocab_structure(
            model, tokenizer, texts, layers, device, args.output_dir,
            n_samples=args.n_samples, k_omp=args.k,
        ),
        "C": lambda: expC_downstream_readers(
            model, tokenizer, texts, layers, device, args.output_dir,
            n_samples=args.n_samples, k_omp=args.k, n_pcs=args.n_pcs,
        ),
        "D": lambda: expD_linguistic_probes(
            model, tokenizer, texts, layers, device, args.output_dir,
            n_samples=args.n_samples, k_omp=args.k, n_pcs=args.n_pcs,
        ),
        "E": lambda: expE_write_read_circuits(
            model, tokenizer, texts, layers, device, args.output_dir,
            n_samples=args.n_samples, k_omp=args.k, n_pcs=args.n_pcs,
        ),
        "F": lambda: expF_ablation_impact(
            model, tokenizer, texts, layers, device, args.output_dir,
            n_samples=args.n_samples, k_omp=args.k, n_pcs=args.n_pcs,
        ),
    }

    for exp_id in exps_to_run:
        if exp_id in exp_funcs:
            exp_funcs[exp_id]()
        else:
            logger.warning(f"Unknown experiment: {exp_id}")

    logger.info("All done.")


if __name__ == "__main__":
    main()
