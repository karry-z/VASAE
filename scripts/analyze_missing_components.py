"""
Diagnostic analysis of why mid/deep-layer GPT-2 activations cannot be sparsely
represented in the token embedding basis.

Six experiments, selectable via --exp {1,2,3,4,5,6,all}.
Results (JSON + PNG) are saved to --output_dir (default: exp/missing_components/).
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
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

logging.basicConfig(
    format="[%(levelname)s] %(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_gpt2(device):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device).eval()
    return model, tokenizer


def get_token_embeddings(model):
    """Return E [V, d] on CPU as float32."""
    return model.transformer.wte.weight.detach().float().cpu()


def get_pos_embeddings(model):
    """Return pos embedding [max_pos, d] on CPU as float32."""
    return model.transformer.wpe.weight.detach().float().cpu()


# ---- OMP utilities (from notebooks/lower_bound.ipynb) ----


@torch.no_grad()
def streaming_topM_candidates(
    H: torch.Tensor,
    E: torch.Tensor,
    M: int = 1024,
    chunk: int = 4096,
) -> torch.Tensor:
    """For each h find top-M candidate atoms by |dot product|."""
    N, d = H.shape
    V = E.shape[0]
    device = H.device
    E = E.to(device)

    best_vals = torch.full((N, M), -float("inf"), device=device)
    best_idx = torch.zeros((N, M), dtype=torch.long, device=device)

    for start in range(0, V, chunk):
        end = min(V, start + chunk)
        scores = (H @ E[start:end].T).abs()
        k_ = min(M, end - start)
        vals, idx = torch.topk(scores, k=k_, dim=1)
        idx = idx + start

        merged_vals = torch.cat([best_vals, vals], dim=1)
        merged_idx = torch.cat([best_idx, idx], dim=1)
        new_vals, new_pos = torch.topk(merged_vals, k=M, dim=1)
        best_idx = merged_idx.gather(1, new_pos)
        best_vals = new_vals

    return best_idx


@torch.no_grad()
def omp_k_error(
    H: torch.Tensor,
    E: torch.Tensor,
    k: int = 8,
    M: int = 1024,
) -> torch.Tensor:
    """OMP@k mean squared error. H [N,d], E [V,d] (dictionary rows)."""
    N, d = H.shape
    device = H.device
    E = E.to(device)

    cand_idx = streaming_topM_candidates(H, E, M=M)

    errs = []
    for n in range(N):
        h = H[n]
        D = E[cand_idx[n]]  # [M, d]
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
        errs.append(r.pow(2).sum())
    return torch.stack(errs).mean()


@torch.no_grad()
def omp_k_error_custom_dict(
    H: torch.Tensor,
    D: torch.Tensor,
    k: int = 8,
) -> torch.Tensor:
    """OMP@k with an arbitrary dictionary D [n_atoms, d].
    Uses direct correlation (no streaming) — D should fit in memory."""
    N, d = H.shape
    device = H.device
    D = D.to(device)
    n_atoms = D.shape[0]

    errs = []
    for n in range(N):
        h = H[n]
        r = h.clone()
        selected = []
        for _ in range(k):
            corr = (D @ r).abs()
            j = int(torch.argmax(corr).item())
            if j in selected:
                break
            selected.append(j)
            A = D[selected].T
            sol = torch.linalg.lstsq(A, h).solution
            r = h - A @ sol
        errs.append(r.pow(2).sum())
    return torch.stack(errs).mean()


# ---- Sub-layer hook collector ----


class SubLayerHookCollector:
    """Collects wte, wpe, and per-block attn / mlp outputs."""

    def __init__(self, model: GPT2LMHeadModel, layers: list[int]):
        self.data = {}
        self.hooks = []
        self.layers = layers

        # token embeddings
        hook = model.transformer.wte.register_forward_hook(self._make_hook("wte"))
        self.hooks.append(hook)
        # positional embeddings — capture via drop (after wte+wpe addition)
        # Instead we'll capture wpe separately in forward
        hook = model.transformer.wpe.register_forward_hook(self._make_hook("wpe"))
        self.hooks.append(hook)

        for l in layers:
            block = model.transformer.h[l]
            h_attn = block.attn.register_forward_hook(self._make_hook(f"attn.{l}"))
            h_mlp = block.mlp.register_forward_hook(self._make_hook(f"mlp.{l}"))
            # block output (residual stream after this block)
            h_block = block.register_forward_hook(self._make_hook(f"block.{l}"))
            self.hooks.extend([h_attn, h_mlp, h_block])

    def _make_hook(self, name):
        def hook(_, __, output):
            if isinstance(output, tuple):
                output = output[0]
            self.data[name] = output.detach().cpu().float()
        return hook

    def clear(self):
        self.data = {}

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def collect_sublayer_activations(model, tokenizer, dataset_texts, layers, device,
                                  max_length=64):
    """Run forward passes, return dict of {component_name: [N, seq, d]}."""
    collector = SubLayerHookCollector(model, layers)
    all_data = {}

    for text in dataset_texts:
        collector.clear()
        tokens = tokenizer(
            text, return_tensors="pt", max_length=max_length,
            truncation=True, padding="max_length",
        ).to(device)
        with torch.no_grad():
            model(**tokens)

        for name, tensor in collector.data.items():
            all_data.setdefault(name, []).append(tensor.squeeze(0))  # [seq, d]

    collector.remove()

    # stack: [N, seq, d]
    return {name: torch.stack(tensors) for name, tensors in all_data.items()}


def get_sample_texts(n=512):
    """Load a small sample of texts from openwebtext (or fallback to simple generation)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("Geralt-Targaryen/openwebtext2", split="train")
        ds = ds.shuffle(seed=42).select(range(n))
        return [ex["text"] for ex in ds]
    except Exception as e:
        logger.warning(f"Could not load openwebtext, using simple texts: {e}")
        return [f"The quick brown fox jumps over the lazy dog. Sentence number {i}." for i in range(n)]


def flatten_activations(acts: torch.Tensor) -> torch.Tensor:
    """[N, seq, d] -> [N*seq, d]"""
    return acts.flatten(0, 1)


def sample_flat(acts_flat: torch.Tensor, n: int = 1024) -> torch.Tensor:
    """Subsample n vectors from flattened activations."""
    if acts_flat.shape[0] <= n:
        return acts_flat
    idx = torch.randperm(acts_flat.shape[0])[:n]
    return acts_flat[idx]


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {path}")


# ---------------------------------------------------------------------------
# Experiment 1: Residual stream sub-layer decomposition
# ---------------------------------------------------------------------------


def exp1_residual_decomposition(model, tokenizer, texts, layers, device, output_dir,
                                 n_samples=1024, k=8):
    logger.info("=== Experiment 1: Residual stream sub-layer decomposition ===")
    E = get_token_embeddings(model).to(device)

    all_acts = collect_sublayer_activations(model, tokenizer, texts, layers, device)
    results = {}

    for l in layers:
        logger.info(f"  Layer {l}...")
        block_act = flatten_activations(all_acts[f"block.{l}"])
        H_block = sample_flat(block_act, n_samples).to(device)
        block_norm = H_block.pow(2).sum(dim=1).mean().item()
        block_omp = omp_k_error(H_block, E, k=k).item()

        components = {}
        # Embedding components
        for comp_name in ["wte", "wpe"]:
            if comp_name in all_acts:
                comp = flatten_activations(all_acts[comp_name])
                H_comp = sample_flat(comp, n_samples).to(device)
                comp_norm = H_comp.pow(2).sum(dim=1).mean().item()
                comp_omp = omp_k_error(H_comp, E, k=k).item()
                components[comp_name] = {
                    "norm": comp_norm,
                    "omp_error": comp_omp,
                    "norm_fraction": comp_norm / max(block_norm, 1e-12),
                }

        # Attention and MLP for layers 0..l
        for ll in range(l + 1):
            if ll not in layers:
                continue
            for sub in ["attn", "mlp"]:
                comp_key = f"{sub}.{ll}"
                if comp_key in all_acts:
                    comp = flatten_activations(all_acts[comp_key])
                    H_comp = sample_flat(comp, n_samples).to(device)
                    comp_norm = H_comp.pow(2).sum(dim=1).mean().item()
                    comp_omp = omp_k_error(H_comp, E, k=k).item()
                    components[comp_key] = {
                        "norm": comp_norm,
                        "omp_error": comp_omp,
                        "norm_fraction": comp_norm / max(block_norm, 1e-12),
                    }

        results[f"layer_{l}"] = {
            "block_norm": block_norm,
            "block_omp_error": block_omp,
            "components": components,
        }

    save_json(results, os.path.join(output_dir, "exp1_residual_decomposition.json"))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for l in layers:
        res = results[f"layer_{l}"]
        comps = res["components"]
        names = list(comps.keys())
        norms = [comps[n]["norm_fraction"] for n in names]
        omp_errs = [comps[n]["omp_error"] for n in names]

        axes[0].bar([f"L{l}:{n}" for n in names], norms, alpha=0.7, label=f"Layer {l}")
        axes[1].bar([f"L{l}:{n}" for n in names], omp_errs, alpha=0.7, label=f"Layer {l}")

    axes[0].set_title("Norm fraction of block output")
    axes[0].tick_params(axis="x", rotation=90)
    axes[1].set_title(f"OMP@{k} error per component")
    axes[1].tick_params(axis="x", rotation=90)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "exp1_residual_decomposition.png"), dpi=150)
    plt.close(fig)
    logger.info("Experiment 1 done.")
    return results


# ---------------------------------------------------------------------------
# Experiment 2: Positional encoding subspace analysis
# ---------------------------------------------------------------------------


def exp2_positional_analysis(model, tokenizer, texts, layers, device, output_dir,
                              n_samples=1024, k=8):
    logger.info("=== Experiment 2: Positional encoding subspace analysis ===")
    E = get_token_embeddings(model).to(device)
    P = get_pos_embeddings(model).to(device)  # [max_pos, d]

    # SVD of positional embeddings
    U, S, Vh = torch.linalg.svd(P, full_matrices=False)
    eps = 1e-6 * S.max()
    pos_rank = int((S > eps).sum().item())
    logger.info(f"  Positional embedding effective rank: {pos_rank}")
    V_pos = Vh[:pos_rank].T  # [d, pos_rank], orthonormal basis of pos subspace

    # Cosine similarity between pos PCs and token embeddings
    E_normed = E / E.norm(dim=1, keepdim=True).clamp(min=1e-8)
    pos_pcs = Vh[:pos_rank]  # [pos_rank, d]
    pos_pcs_normed = pos_pcs / pos_pcs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    max_cos_sim = (pos_pcs_normed @ E_normed.T).abs().max(dim=1).values  # [pos_rank]

    all_acts = collect_sublayer_activations(model, tokenizer, texts, layers, device)
    results = {"pos_rank": pos_rank, "per_layer": {}}

    for l in layers:
        logger.info(f"  Layer {l}...")
        block_act = flatten_activations(all_acts[f"block.{l}"])
        H = sample_flat(block_act, n_samples).to(device)

        # Project onto pos subspace
        H_pos = (H @ V_pos) @ V_pos.T  # [N, d]
        H_no_pos = H - H_pos

        var_total = H.pow(2).sum(dim=1).mean().item()
        var_pos = H_pos.pow(2).sum(dim=1).mean().item()

        # OMP on original vs pos-removed
        omp_orig = omp_k_error(H, E, k=k).item()
        omp_no_pos = omp_k_error(H_no_pos, E, k=k).item()

        results["per_layer"][f"layer_{l}"] = {
            "var_pos_fraction": var_pos / max(var_total, 1e-12),
            "omp_error_original": omp_orig,
            "omp_error_pos_removed": omp_no_pos,
            "omp_error_change": (omp_no_pos - omp_orig) / max(omp_orig, 1e-12),
        }

    results["pos_E_max_cosine"] = max_cos_sim.cpu().tolist()[:16]  # first 16 PCs

    save_json(results, os.path.join(output_dir, "exp2_positional_analysis.json"))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    layer_labels = [str(l) for l in layers]
    var_fracs = [results["per_layer"][f"layer_{l}"]["var_pos_fraction"] for l in layers]
    omp_orig = [results["per_layer"][f"layer_{l}"]["omp_error_original"] for l in layers]
    omp_nop = [results["per_layer"][f"layer_{l}"]["omp_error_pos_removed"] for l in layers]

    axes[0].bar(layer_labels, var_fracs)
    axes[0].set_title("Pos subspace variance fraction")
    axes[0].set_xlabel("Layer")

    axes[1].plot(layer_labels, omp_orig, "o-", label="Original")
    axes[1].plot(layer_labels, omp_nop, "s-", label="Pos removed")
    axes[1].set_title(f"OMP@{k} error")
    axes[1].legend()
    axes[1].set_xlabel("Layer")

    axes[2].bar(range(len(results["pos_E_max_cosine"])), results["pos_E_max_cosine"])
    axes[2].set_title("Max |cos| of pos PCs with token embeddings")
    axes[2].set_xlabel("PC index")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "exp2_positional_analysis.png"), dpi=150)
    plt.close(fig)
    logger.info("Experiment 2 done.")
    return results


# ---------------------------------------------------------------------------
# Experiment 3: Activation covariance vs E geometry
# ---------------------------------------------------------------------------


def exp3_geometry_evolution(model, tokenizer, texts, layers, device, output_dir,
                             n_samples=1024, top_k_pca=64):
    logger.info("=== Experiment 3: Activation covariance vs E geometry ===")
    E = get_token_embeddings(model).to(device)

    # SVD of E
    _, S_E, Vh_E = torch.linalg.svd(E, full_matrices=False)
    # Top-k right singular vectors of E
    V_E_topk = Vh_E[:top_k_pca].T  # [d, top_k]

    # E^T E
    EtE = E.T @ E  # [d, d]
    trace_EtE = torch.trace(EtE).item()

    all_acts = collect_sublayer_activations(model, tokenizer, texts, layers, device)
    results = {}

    for l in layers:
        logger.info(f"  Layer {l}...")
        block_act = flatten_activations(all_acts[f"block.{l}"])
        H = sample_flat(block_act, n_samples).to(device)

        # Covariance
        H_centered = H - H.mean(dim=0, keepdim=True)
        C = (H_centered.T @ H_centered) / (H.shape[0] - 1)  # [d, d]

        # Trace alignment: tr(C @ E^T E) / (||C||_F * ||E^T E||_F)
        trace_align = torch.trace(C @ EtE).item()
        norm_C = torch.norm(C, p="fro").item()
        norm_EtE = torch.norm(EtE, p="fro").item()
        alignment = trace_align / max(norm_C * norm_EtE, 1e-12)

        # Principal angles between top-k eigenvectors of C and top-k right SVs of E
        eigvals, eigvecs = torch.linalg.eigh(C)
        # eigh returns ascending order, take last top_k
        V_C_topk = eigvecs[:, -top_k_pca:]  # [d, top_k]

        # Principal angles via SVD of V_C^T V_E
        M = V_C_topk.T @ V_E_topk  # [top_k, top_k]
        svals = torch.linalg.svdvals(M)
        principal_angles_deg = torch.acos(svals.clamp(-1, 1)).rad2deg()

        # Condition number of E restricted to activation principal subspace
        E_proj = E @ V_C_topk  # [V, top_k]
        cond = torch.linalg.cond(E_proj).item()

        results[f"layer_{l}"] = {
            "trace_alignment": alignment,
            "principal_angles_deg_mean": principal_angles_deg.mean().item(),
            "principal_angles_deg_median": principal_angles_deg.median().item(),
            "principal_angles_deg_max": principal_angles_deg.max().item(),
            "condition_number": cond,
            "top_eigvals": eigvals[-16:].flip(0).cpu().tolist(),
        }

    save_json(results, os.path.join(output_dir, "exp3_geometry_evolution.json"))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    layer_labels = [str(l) for l in layers]
    aligns = [results[f"layer_{l}"]["trace_alignment"] for l in layers]
    angles = [results[f"layer_{l}"]["principal_angles_deg_mean"] for l in layers]
    conds = [results[f"layer_{l}"]["condition_number"] for l in layers]

    axes[0].plot(layer_labels, aligns, "o-")
    axes[0].set_title("Trace alignment C vs E^T E")
    axes[0].set_xlabel("Layer")

    axes[1].plot(layer_labels, angles, "o-")
    axes[1].set_title("Mean principal angle (deg)")
    axes[1].set_xlabel("Layer")

    axes[2].plot(layer_labels, conds, "o-")
    axes[2].set_title("Condition number of E|_{act PCA}")
    axes[2].set_xlabel("Layer")
    axes[2].set_yscale("log")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "exp3_geometry_evolution.png"), dpi=150)
    plt.close(fig)
    logger.info("Experiment 3 done.")
    return results


# ---------------------------------------------------------------------------
# Experiment 4: Augmented dictionary experiment
# ---------------------------------------------------------------------------


def exp4_augmented_dictionary(model, tokenizer, texts, layers, device, output_dir,
                               n_samples=512, k=8, r_values=(8, 16, 32, 64, 128)):
    logger.info("=== Experiment 4: Augmented dictionary experiment ===")
    E = get_token_embeddings(model).to(device)
    P = get_pos_embeddings(model).to(device)
    d = E.shape[1]

    all_acts = collect_sublayer_activations(model, tokenizer, texts, layers, device)
    results = {}

    for l in layers:
        logger.info(f"  Layer {l}...")
        block_act = flatten_activations(all_acts[f"block.{l}"])
        H = sample_flat(block_act, n_samples).to(device)

        # Baseline
        baseline_err = omp_k_error(H, E, k=k).item()

        layer_results = {"baseline_omp_error": baseline_err}

        for r in r_values:
            aug_results = {}

            # E + pos (top-r PCA of positional embeddings)
            _, _, Vh_pos = torch.linalg.svd(P, full_matrices=False)
            pos_dirs = Vh_pos[:min(r, Vh_pos.shape[0])]
            D_pos = torch.cat([E, pos_dirs], dim=0)
            err = omp_k_error_custom_dict(H, D_pos, k=k).item()
            aug_results["E+pos"] = {"error": err, "rel_drop": (baseline_err - err) / max(baseline_err, 1e-12)}

            # E + mlp_dirs
            mlp_key = f"mlp.{l}"
            if mlp_key in all_acts:
                mlp_flat = flatten_activations(all_acts[mlp_key]).to(device)
                mlp_centered = mlp_flat - mlp_flat.mean(0, keepdim=True)
                _, _, Vh_mlp = torch.linalg.svd(mlp_centered[:min(2048, mlp_centered.shape[0])], full_matrices=False)
                mlp_dirs = Vh_mlp[:r]
                D_mlp = torch.cat([E, mlp_dirs], dim=0)
                err = omp_k_error_custom_dict(H, D_mlp, k=k).item()
                aug_results["E+mlp_dirs"] = {"error": err, "rel_drop": (baseline_err - err) / max(baseline_err, 1e-12)}

            # E + attn_dirs
            attn_key = f"attn.{l}"
            if attn_key in all_acts:
                attn_flat = flatten_activations(all_acts[attn_key]).to(device)
                attn_centered = attn_flat - attn_flat.mean(0, keepdim=True)
                _, _, Vh_attn = torch.linalg.svd(attn_centered[:min(2048, attn_centered.shape[0])], full_matrices=False)
                attn_dirs = Vh_attn[:r]
                D_attn = torch.cat([E, attn_dirs], dim=0)
                err = omp_k_error_custom_dict(H, D_attn, k=k).item()
                aug_results["E+attn_dirs"] = {"error": err, "rel_drop": (baseline_err - err) / max(baseline_err, 1e-12)}

            # E + LN(E) — apply layer norm from this layer to E
            ln = model.transformer.h[l].ln_1
            E_ln = ln(E).detach()
            D_ln = torch.cat([E, E_ln], dim=0)
            err = omp_k_error_custom_dict(H, D_ln, k=k).item()
            aug_results["E+LN(E)"] = {"error": err, "rel_drop": (baseline_err - err) / max(baseline_err, 1e-12)}

            # E + act_pca
            H_centered = H - H.mean(0, keepdim=True)
            _, _, Vh_act = torch.linalg.svd(H_centered[:min(2048, H_centered.shape[0])], full_matrices=False)
            act_dirs = Vh_act[:r]
            D_act = torch.cat([E, act_dirs], dim=0)
            err = omp_k_error_custom_dict(H, D_act, k=k).item()
            aug_results["E+act_pca"] = {"error": err, "rel_drop": (baseline_err - err) / max(baseline_err, 1e-12)}

            # E + random (control)
            rand_dirs = torch.randn(r, d, device=device)
            rand_dirs = rand_dirs / rand_dirs.norm(dim=1, keepdim=True)
            D_rand = torch.cat([E, rand_dirs], dim=0)
            err = omp_k_error_custom_dict(H, D_rand, k=k).item()
            aug_results["E+random"] = {"error": err, "rel_drop": (baseline_err - err) / max(baseline_err, 1e-12)}

            layer_results[f"r={r}"] = aug_results

        results[f"layer_{l}"] = layer_results

    save_json(results, os.path.join(output_dir, "exp4_augmented_dictionary.json"))

    # Heatmap: for each r, rows=aug types, cols=layers, values=rel_drop
    for r in r_values:
        aug_types = ["E+pos", "E+mlp_dirs", "E+attn_dirs", "E+LN(E)", "E+act_pca", "E+random"]
        matrix = []
        for aug in aug_types:
            row = []
            for l in layers:
                entry = results[f"layer_{l}"].get(f"r={r}", {}).get(aug, {})
                row.append(entry.get("rel_drop", 0.0))
            matrix.append(row)

        fig, ax = plt.subplots(figsize=(max(8, len(layers) * 1.2), 6))
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([str(l) for l in layers])
        ax.set_yticks(range(len(aug_types)))
        ax.set_yticklabels(aug_types)
        ax.set_xlabel("Layer")
        ax.set_title(f"Relative OMP@{k} error drop (r={r})")
        for i in range(len(aug_types)):
            for j in range(len(layers)):
                ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"exp4_heatmap_r{r}.png"), dpi=150)
        plt.close(fig)

    logger.info("Experiment 4 done.")
    return results


# ---------------------------------------------------------------------------
# Experiment 5: Sparsity-density transition
# ---------------------------------------------------------------------------


def exp5_sparsity_density(model, tokenizer, texts, layers, device, output_dir,
                           n_samples=512, k_values=(1, 2, 4, 8, 16, 32, 64, 128)):
    logger.info("=== Experiment 5: Sparsity-density transition ===")
    E = get_token_embeddings(model).to(device)
    d = E.shape[1]

    all_acts = collect_sublayer_activations(model, tokenizer, texts, layers, device)
    results = {}

    for l in layers:
        logger.info(f"  Layer {l}...")
        block_act = flatten_activations(all_acts[f"block.{l}"])
        H = sample_flat(block_act, n_samples).to(device)
        h_norm_sq = H.pow(2).sum(dim=1).mean().item()

        # Dense coefficients c = pinv(E) @ h => E is [V, d], pinv(E) is [d, V]
        # c = H @ pinv(E).T = H @ E^T @ (E E^T)^{-1}... but E is V>>d so pinv = E^T (E E^T)^{-1}
        # Actually pinv(E) where E is [V,d] with V>>d: pinv = (E^T E)^{-1} E^T  [d, V]
        # c = h @ pinv(E)^T = [N, V]
        # This is too large. Instead compute stats without materializing c fully.
        # Gini and L1/L2 on c: c = (E^T E)^{-1} E^T h^T => c_i for each h
        EtE_inv = torch.linalg.inv(E.T @ E)  # [d, d]
        # For each h: c = E @ EtE_inv @ h  [V]
        # Too expensive to store [N, V]. Do per-sample.
        gini_list = []
        l1_l2_list = []
        for n in range(min(n_samples, 256)):
            c = E @ (EtE_inv @ H[n])  # [V]
            c_abs = c.abs()
            # Gini coefficient
            sorted_c, _ = c_abs.sort()
            V = sorted_c.shape[0]
            index = torch.arange(1, V + 1, device=device, dtype=torch.float32)
            gini = (2 * (index * sorted_c).sum() / (V * sorted_c.sum()) - (V + 1) / V).item()
            gini_list.append(gini)
            l1_l2_list.append((c_abs.sum() / c_abs.norm()).item())

        # k-error curves with E
        k_errors_E = {}
        for kk in k_values:
            err = omp_k_error(H, E, k=kk).item()
            k_errors_E[str(kk)] = err
            logger.info(f"    k={kk}: OMP error = {err:.2f}")

        # k-error with PCA basis
        H_centered = H - H.mean(0, keepdim=True)
        _, _, Vh_act = torch.linalg.svd(H_centered[:min(2048, H_centered.shape[0])], full_matrices=False)
        pca_dict = Vh_act  # [d, d] — all d directions

        k_errors_pca = {}
        for kk in k_values:
            if kk <= d:
                err = omp_k_error_custom_dict(H, pca_dict, k=kk).item()
                k_errors_pca[str(kk)] = err

        # k-error with random Gaussian dictionary (same size as E)
        rand_dict = torch.randn(E.shape[0], d, device=device)
        rand_dict = rand_dict / rand_dict.norm(dim=1, keepdim=True)
        k_errors_rand = {}
        for kk in [1, 2, 4, 8]:  # limit to small k for speed
            err = omp_k_error(H, rand_dict, k=kk).item()
            k_errors_rand[str(kk)] = err

        # Find k needed for 10% relative error
        k_for_10pct = None
        for kk in k_values:
            rel_err = k_errors_E[str(kk)] / max(h_norm_sq, 1e-12)
            if rel_err < 0.1:
                k_for_10pct = kk
                break

        results[f"layer_{l}"] = {
            "gini_mean": float(np.mean(gini_list)),
            "gini_std": float(np.std(gini_list)),
            "l1_l2_mean": float(np.mean(l1_l2_list)),
            "k_errors_E": k_errors_E,
            "k_errors_pca": k_errors_pca,
            "k_errors_random": k_errors_rand,
            "h_norm_sq_mean": h_norm_sq,
            "k_for_10pct_error": k_for_10pct,
        }

    save_json(results, os.path.join(output_dir, "exp5_sparsity_density.json"))

    # Plot k-error curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for l in layers:
        res = results[f"layer_{l}"]
        ks = [int(x) for x in res["k_errors_E"].keys()]
        errs = list(res["k_errors_E"].values())
        axes[0].plot(ks, errs, "o-", label=f"L{l}")

    axes[0].set_title("OMP@k error (E dict)")
    axes[0].set_xlabel("k")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].legend()

    # Gini
    layer_labels = [str(l) for l in layers]
    ginis = [results[f"layer_{l}"]["gini_mean"] for l in layers]
    axes[1].bar(layer_labels, ginis)
    axes[1].set_title("Gini coefficient of dense coefficients")
    axes[1].set_xlabel("Layer")

    # k needed for 10%
    k10s = [results[f"layer_{l}"]["k_for_10pct_error"] for l in layers]
    axes[2].bar(layer_labels, [k if k is not None else max(k_values) * 2 for k in k10s])
    axes[2].set_title("k needed for <10% relative error")
    axes[2].set_xlabel("Layer")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "exp5_sparsity_density.png"), dpi=150)
    plt.close(fig)
    logger.info("Experiment 5 done.")
    return results


# ---------------------------------------------------------------------------
# Experiment 6: LayerNorm geometric distortion
# ---------------------------------------------------------------------------


def exp6_layernorm_distortion(model, tokenizer, texts, layers, device, output_dir,
                               n_samples=512, k=8):
    logger.info("=== Experiment 6: LayerNorm geometric distortion ===")
    E = get_token_embeddings(model).to(device)
    d = E.shape[1]

    all_acts = collect_sublayer_activations(model, tokenizer, texts, layers, device)
    results = {}

    for l in layers:
        logger.info(f"  Layer {l}...")
        block_act = flatten_activations(all_acts[f"block.{l}"])
        H = sample_flat(block_act, n_samples).to(device)

        # Apply LayerNorm from the next layer (or final LN for last layer)
        if l < 11:
            ln = model.transformer.h[l + 1].ln_1
        else:
            ln = model.transformer.ln_f
        H_ln = ln(H).detach()

        # OMP on pre-LN activations
        omp_pre = omp_k_error(H, E, k=k).item()
        # OMP on post-LN activations
        omp_post = omp_k_error(H_ln, E, k=k).item()

        # LN-adapted dictionary: apply LN to E
        E_ln = ln(E).detach()
        omp_adapted = omp_k_error(H_ln, E_ln, k=k).item()

        # Angle distortion: sample pairs of token embeddings, compare angles before/after LN
        n_pairs = 500
        idx1 = torch.randint(0, E.shape[0], (n_pairs,))
        idx2 = torch.randint(0, E.shape[0], (n_pairs,))
        e1, e2 = E[idx1], E[idx2]
        e1_ln, e2_ln = E_ln[idx1], E_ln[idx2]

        cos_before = torch.nn.functional.cosine_similarity(e1, e2, dim=1)
        cos_after = torch.nn.functional.cosine_similarity(e1_ln, e2_ln, dim=1)
        angle_distortion = (cos_after - cos_before).abs()

        results[f"layer_{l}"] = {
            "omp_pre_ln": omp_pre,
            "omp_post_ln": omp_post,
            "omp_ln_adapted_dict": omp_adapted,
            "angle_distortion_mean": angle_distortion.mean().item(),
            "angle_distortion_std": angle_distortion.std().item(),
            "angle_distortion_max": angle_distortion.max().item(),
        }

    save_json(results, os.path.join(output_dir, "exp6_layernorm_distortion.json"))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    layer_labels = [str(l) for l in layers]
    omp_pre = [results[f"layer_{l}"]["omp_pre_ln"] for l in layers]
    omp_post = [results[f"layer_{l}"]["omp_post_ln"] for l in layers]
    omp_adapt = [results[f"layer_{l}"]["omp_ln_adapted_dict"] for l in layers]

    axes[0].plot(layer_labels, omp_pre, "o-", label="Pre-LN (E dict)")
    axes[0].plot(layer_labels, omp_post, "s-", label="Post-LN (E dict)")
    axes[0].plot(layer_labels, omp_adapt, "^-", label="Post-LN (LN(E) dict)")
    axes[0].set_title(f"OMP@{k} error: LN effect")
    axes[0].legend()
    axes[0].set_xlabel("Layer")

    dist_means = [results[f"layer_{l}"]["angle_distortion_mean"] for l in layers]
    axes[1].bar(layer_labels, dist_means)
    axes[1].set_title("Mean |cos angle distortion| from LN")
    axes[1].set_xlabel("Layer")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "exp6_layernorm_distortion.png"), dpi=150)
    plt.close(fig)
    logger.info("Experiment 6 done.")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose why mid/deep-layer activations can't be sparsely represented in token embedding basis"
    )
    parser.add_argument(
        "--exp", type=str, default="1",
        help="Experiment(s) to run: 1,2,3,4,5,6 or 'all'",
    )
    parser.add_argument("--layers", type=str, default="0,1,3,5,7,9,11",
                        help="Comma-separated layer indices")
    parser.add_argument("--n_samples", type=int, default=512,
                        help="Number of activation vectors to sample per layer")
    parser.add_argument("--n_texts", type=int, default=256,
                        help="Number of text samples to run through GPT-2")
    parser.add_argument("--k", type=int, default=8,
                        help="Sparsity level for OMP")
    parser.add_argument("--output_dir", type=str, default="exp/missing_components",
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

    exps_to_run = list(range(1, 7)) if args.exp == "all" else [int(x) for x in args.exp.split(",")]

    exp_funcs = {
        1: lambda: exp1_residual_decomposition(model, tokenizer, texts, layers, device, args.output_dir, args.n_samples, args.k),
        2: lambda: exp2_positional_analysis(model, tokenizer, texts, layers, device, args.output_dir, args.n_samples, args.k),
        3: lambda: exp3_geometry_evolution(model, tokenizer, texts, layers, device, args.output_dir, args.n_samples),
        4: lambda: exp4_augmented_dictionary(model, tokenizer, texts, layers, device, args.output_dir, args.n_samples, args.k),
        5: lambda: exp5_sparsity_density(model, tokenizer, texts, layers, device, args.output_dir, args.n_samples),
        6: lambda: exp6_layernorm_distortion(model, tokenizer, texts, layers, device, args.output_dir, args.n_samples, args.k),
    }

    for exp_id in exps_to_run:
        if exp_id in exp_funcs:
            exp_funcs[exp_id]()
        else:
            logger.warning(f"Unknown experiment: {exp_id}")

    logger.info("All done.")


if __name__ == "__main__":
    main()
