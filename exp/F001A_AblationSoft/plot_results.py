"""Generate ablation study figures for F001A_AblationSoft."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS_DIR = Path("/scratch/b5bq/pu22650.b5bq/VASAE_out/001A_F_AblationSoft")
OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def load(name: str) -> dict | None:
    p = RESULTS_DIR / name / "results.json"
    if not p.is_file():
        # follow symlink
        p = p.resolve()
    if not p.is_file():
        return None
    with open(p) as f:
        return json.load(f)


# ── Exp 1a: Lambda sweep ─────────────────────────────────────────────────────

def plot_lambda_sweep():
    lambdas_str = ["0", "1e-5", "1e-4", "5e-4", "1e-3", "5e-3"]
    lambdas_val = [0, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3]

    for model, layers, tag in [("gpt2", [0, 5, 11], "GPT-2"), ("llama", [0, 15, 31], "Llama-3.1-8B")]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for li, layer in enumerate(layers):
            ve_vals, ce_vals, xs = [], [], []
            for lam_s, lam_v in zip(lambdas_str, lambdas_val):
                r = load(f"001AF_{model}_lambda_L{layer}_a{lam_s}")
                if r is None:
                    continue
                ve_vals.append(r["test"]["variance_explained"])
                ce_vals.append(r["test"]["loss_recovered"])
                xs.append(lam_v)

            # shift 0 → small value for log scale
            xs_plot = [max(x, 5e-6) for x in xs]
            axes[0].plot(xs_plot, ve_vals, "o-", color=COLORS[li], label=f"L{layer}", markersize=5)
            axes[1].plot(xs_plot, ce_vals, "o-", color=COLORS[li], label=f"L{layer}", markersize=5)

        for ax, metric in zip(axes, ["Variance Explained", "CE Recovery"]):
            ax.set_xscale("log")
            ax.set_xlabel("$\\lambda$ (anchor coefficient)")
            ax.set_ylabel(metric)
            ax.set_title(f"{tag} — {metric}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            # Custom x ticks
            ax.set_xticks([5e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3])
            ax.set_xticklabels(["0", "1e-5", "1e-4", "5e-4", "1e-3", "5e-3"], fontsize=9)

        fig.suptitle(f"Exp 1a: Anchor λ Sweep — {tag}", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"exp1a_lambda_{model}.pdf")
        fig.savefig(OUT_DIR / f"exp1a_lambda_{model}.png")
        plt.close(fig)
        print(f"  saved exp1a_lambda_{model}")


# ── Exp 1b: Mode comparison ──────────────────────────────────────────────────

def plot_mode_comparison():
    modes = ["hard", "logsumexp", "softmax"]

    for model, layers, tag in [("gpt2", [0, 5, 11], "GPT-2"), ("llama", [0, 15, 31], "Llama-3.1-8B")]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        ve_data = {m: [] for m in modes}
        ce_data = {m: [] for m in modes}
        valid_layers = []

        for layer in layers:
            row_ok = True
            for mode in modes:
                if mode == "hard":
                    r = load(f"001AF_{model}_lambda_L{layer}_a1e-4")
                else:
                    r = load(f"001AF_{model}_mode_L{layer}_{mode}")
                if r is None:
                    row_ok = False
                    break
                ve_data[mode].append(r["test"]["variance_explained"])
                ce_data[mode].append(r["test"]["loss_recovered"])
            if not row_ok:
                # remove partial entries
                for m in modes:
                    if len(ve_data[m]) > len(valid_layers):
                        ve_data[m].pop()
                        ce_data[m].pop()
                continue
            valid_layers.append(layer)

        x = np.arange(len(valid_layers))
        width = 0.25

        for mi, mode in enumerate(modes):
            axes[0].bar(x + mi * width, ve_data[mode], width, label=mode, color=COLORS[mi])
            axes[1].bar(x + mi * width, ce_data[mode], width, label=mode, color=COLORS[mi])

        for ax, metric in zip(axes, ["Variance Explained", "CE Recovery"]):
            ax.set_xticks(x + width)
            ax.set_xticklabels([f"L{l}" for l in valid_layers])
            ax.set_ylabel(metric)
            ax.set_title(f"{tag} — {metric}")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            # zoom y-axis to show differences
            all_vals = sum((ve_data[m] if "Variance" in metric else ce_data[m] for m in modes), [])
            if all_vals:
                lo = min(all_vals)
                hi = max(all_vals)
                margin = max((hi - lo) * 0.5, 0.005)
                ax.set_ylim(lo - margin, min(hi + margin, 1.0))

        fig.suptitle(f"Exp 1b: Anchor Mode Comparison — {tag}", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"exp1b_mode_{model}.pdf")
        fig.savefig(OUT_DIR / f"exp1b_mode_{model}.png")
        plt.close(fig)
        print(f"  saved exp1b_mode_{model}")


# ── Exp 2: Sparsity Pareto curve ─────────────────────────────────────────────

def plot_pareto():
    ks = [8, 16, 32, 64, 128]

    for model, layers, tag in [("gpt2", [0, 5, 11], "GPT-2"), ("llama", [0, 15, 31], "Llama-3.1-8B")]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for li, layer in enumerate(layers):
            l0_vals, ve_vals, ce_vals = [], [], []
            for k in ks:
                if k == 32:
                    r = load(f"001AF_{model}_lambda_L{layer}_a1e-4")
                else:
                    r = load(f"001AF_{model}_k_L{layer}_k{k}")
                if r is None:
                    continue
                t = r["test"]
                l0 = r.get("l0", k)  # 001_F doesn't have l0
                if l0 is None:
                    l0 = k  # approximate
                l0_vals.append(l0)
                ve_vals.append(t["variance_explained"])
                ce_vals.append(t["loss_recovered"])

            axes[0].plot(l0_vals, ve_vals, "o-", color=COLORS[li], label=f"L{layer}", markersize=6)
            axes[1].plot(l0_vals, ce_vals, "o-", color=COLORS[li], label=f"L{layer}", markersize=6)

        for ax, metric in zip(axes, ["Variance Explained", "CE Recovery"]):
            ax.set_xlabel("L0 (avg active features)")
            ax.set_ylabel(metric)
            ax.set_title(f"{tag} — {metric}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Exp 2: Sparsity–Quality Pareto — {tag}", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"exp2_pareto_{model}.pdf")
        fig.savefig(OUT_DIR / f"exp2_pareto_{model}.png")
        plt.close(fig)
        print(f"  saved exp2_pareto_{model}")


# ── Exp 2 supplement: Dead Rate vs k ──────────────────────────────────────────

def plot_dead_rate():
    ks = [8, 16, 64, 128]  # skip k=32 (no dead_rate from 001_F)

    for model, layers, tag in [("gpt2", [0, 5, 11], "GPT-2"), ("llama", [0, 15, 31], "Llama-3.1-8B")]:
        fig, ax = plt.subplots(figsize=(6, 4))

        for li, layer in enumerate(layers):
            k_vals, dead_vals = [], []
            for k in ks:
                r = load(f"001AF_{model}_k_L{layer}_k{k}")
                if r is None or r.get("dead_rate") is None:
                    continue
                k_vals.append(k)
                dead_vals.append(r["dead_rate"])
            ax.plot(k_vals, dead_vals, "s-", color=COLORS[li], label=f"L{layer}", markersize=6)

        ax.set_xlabel("$k$ (TopK)")
        ax.set_ylabel("Dead Feature Rate")
        ax.set_title(f"{tag} — Dead Rate vs $k$")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        fig.tight_layout()
        fig.savefig(OUT_DIR / f"exp2_deadrate_{model}.pdf")
        fig.savefig(OUT_DIR / f"exp2_deadrate_{model}.png")
        plt.close(fig)
        print(f"  saved exp2_deadrate_{model}")


# ── Exp 3: Frequency ablation ────────────────────────────────────────────────

def plot_frequency():
    freqs = [1, 10, 50, 100, 500]
    layers = [0, 15, 31]

    # Wall time per epoch (minutes), from log analysis
    wall_times = {
        (0, 1): 360, (0, 10): 44, (0, 100): 12, (0, 500): 9,
        (15, 1): 364, (15, 500): 11,
        (31, 1): 365, (31, 10): 47, (31, 100): 16, (31, 500): 13,
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: CE Recovery vs anchor_every
    for li, layer in enumerate(layers):
        xs, ys = [], []
        for freq in freqs:
            r = load(f"001AF_llama_freq_L{layer}_every{freq}")
            if r is None:
                continue
            xs.append(freq)
            ys.append(r["test"]["loss_recovered"])
        axes[0].plot(xs, ys, "o-", color=COLORS[li], label=f"L{layer}", markersize=6)

    axes[0].set_xscale("log")
    axes[0].set_xlabel("anchor_every")
    axes[0].set_ylabel("CE Recovery")
    axes[0].set_title("Quality vs Frequency")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(freqs)
    axes[0].get_xaxis().set_major_formatter(mticker.ScalarFormatter())

    # Right: Wall time per epoch vs anchor_every
    for li, layer in enumerate(layers):
        xs, ys = [], []
        for freq in freqs:
            if (layer, freq) in wall_times:
                xs.append(freq)
                ys.append(wall_times[(layer, freq)])
        axes[1].plot(xs, ys, "s-", color=COLORS[li], label=f"L{layer}", markersize=6)

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("anchor_every")
    axes[1].set_ylabel("Wall time / epoch (min)")
    axes[1].set_title("Training Cost vs Frequency")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(freqs)
    axes[1].get_xaxis().set_major_formatter(mticker.ScalarFormatter())

    fig.suptitle("Exp 3: Anchor Computation Frequency — Llama-3.1-8B", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"exp3_frequency_llama.pdf")
    fig.savefig(OUT_DIR / f"exp3_frequency_llama.png")
    plt.close(fig)
    print("  saved exp3_frequency_llama")


if __name__ == "__main__":
    print("Plotting Exp 1a...")
    plot_lambda_sweep()
    print("Plotting Exp 1b...")
    plot_mode_comparison()
    print("Plotting Exp 2 (Pareto)...")
    plot_pareto()
    print("Plotting Exp 2 (Dead Rate)...")
    plot_dead_rate()
    print("Plotting Exp 3...")
    plot_frequency()
    print(f"\nAll figures saved to {OUT_DIR}")
