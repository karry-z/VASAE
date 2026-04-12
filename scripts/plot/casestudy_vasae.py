"""F002 case study: VASAE logit-lens visualization on hand-picked texts.

For each layer of a trained VASAE-Soft stack (the F002 checkpoints), encode
the LM's hidden states with that layer's SAE, take the top-1 vocab token from
the sparse code at every (layer, position), and render the canonical
"logit lens" heatmap (rows = layers, cols = sequence positions; cell text =
predicted token, cell color = its probability).

This is the script form of ``notebooks/logitlens/vasae.ipynb``, set up to
produce qualitative case studies for the F002 alignment-quality report:

  * Loads SAEs via ``load_sae_for_analysis`` (HF ``from_pretrained`` format).
  * Discovers per-layer checkpoints from the F002 directory layout
    (``001F_{model_tag}_L{layer}_soft`` under ``001_F_Benchmarking``, or the
    Llama 5e-3 ablation dirs under ``001A_F_AblationSoft``).
  * Runs a list of input texts and saves one PDF + PNG per text plus a
    ``manifest.json`` capturing the per-cell predictions.

Examples:

    # GPT-2 — F002 main checkpoints (12 layers)
    uv run python scripts/plot/casestudy_vasae.py \\
        --model gpt2 \\
        --sae-root /scratch/b5bq/pu22650.b5bq/VASAE_out/001_F_Benchmarking \\
        --sae-pattern '001F_gpt2_L{layer}_soft' \\
        --layers 0-11 \\
        --output-dir exp/F002_AlignmentAnalysis/casestudy/gpt2 \\
        --device cuda

    # Llama-3.1-8B — λ=5e-3 ablation, only L0/L15/L31 saved
    uv run python scripts/plot/casestudy_vasae.py \\
        --model meta-llama/Llama-3.1-8B \\
        --sae-root /scratch/b5bq/pu22650.b5bq/VASAE_out/001A_F_AblationSoft \\
        --sae-pattern '001AF_llama_lambda_L{layer}_a5e-3' \\
        --layers 0,15,31 \\
        --output-dir exp/F002_AlignmentAnalysis/casestudy/llama_5e-3 \\
        --device cuda
"""

from __future__ import annotations

import os

os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap

from vasae.analysis.alignment import compute_geometric_alignment
from vasae.analysis.sae_loader import get_decoder_features, load_sae_for_analysis
from vasae.models.factory import get_embedding, load_model
from vasae.models.sae import SAEModel
from vasae.utils.log import get_logger
from vasae.utils.seed import set_seed

logger = get_logger("casestudy_logitlens")


# Default cases. Each entry is (slug, text).
DEFAULT_CASES: List[Tuple[str, str]] = [
    (
        "names_townsend",
        "Pete Townsend, the legendary guitarist of The Who, walked down the street",
    ),
    (
        "names_fey",
        "Tina Fey hosted the awards ceremony alongside her longtime collaborator",
    ),
    (
        "names_nicole",
        "Nicole Kidman accepted the award and thanked her family and friends",
    ),
    (
        "morph_ible",
        "The proposal was perfectly feasible, entirely sensible, and quite reasonable",
    ),
    (
        "place_street",
        "The cafe is located on Baker Street, just around the corner from the avenue",
    ),
    (
        "self_intro",
        "my name is Bryan, I am a huge fan of yours. I was always very impressed with you",
    ),
    (
        "factual_einstein",
        "Albert Einstein was born in the year 1879 in the city of Ulm in the country of",
    ),
    (
        "ioi_simple",
        "When John and Mary went to the store, John gave a drink to",
    ),
]


# ---------------------------------------------------------------------------
# Plotting (mirrors vasae.ipynb)
# ---------------------------------------------------------------------------


def clip_cmap(base_cmap, lo: float = 0.08, hi: float = 0.75, n: int = 256):
    """Crop a matplotlib colormap to ``[lo, hi]`` so extreme shades are dropped."""
    colors = base_cmap(np.linspace(lo, hi, n))
    return LinearSegmentedColormap.from_list(f"{base_cmap.name}_clip", colors)


def plot_logit_lens(
    tokens_by_layer: np.ndarray,  # [L, S] strings
    probs_by_layer: np.ndarray,  # [L, S] floats in [0, 1]
    x_tokens: Sequence[str],
    layer_labels: Sequence[int],
    out_path: Path,
    title: str | None = None,
    fontsize: int = 9,
    dpi: int = 300,
    cbar_ticks: Sequence[float] = (0, 0.2, 0.4, 0.6, 0.8, 1.0),
) -> None:
    """Render the canonical logit-lens heatmap and save to ``out_path``."""
    Z = probs_by_layer[::-1]  # high layers on top
    T = tokens_by_layer[::-1]
    y_labels = list(layer_labels)[::-1]

    H, W = Z.shape

    fig_w = max(1.2, 0.9 * W)
    fig_h = max(1.5, 0.45 * H)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    norm = mcolors.Normalize(0, 1)
    cmap = clip_cmap(plt.cm.Blues, lo=0.1, hi=0.7)

    im = ax.imshow(
        Z,
        cmap=cmap,
        norm=norm,
        interpolation="none",
        aspect="auto",
        origin="upper",
    )

    for i in range(H):
        for j in range(W):
            ax.text(
                j,
                i,
                T[i, j],
                ha="center",
                va="center",
                fontsize=fontsize,
            )

    ax.set_xticks(range(W))
    ax.set_xticklabels(list(x_tokens), fontsize=fontsize)
    ax.tick_params(axis="x", pad=4)

    yt = np.linspace(0, H - 1, min(8, H), dtype=int)
    ax.set_yticks(yt)
    ax.set_yticklabels([y_labels[i] for i in yt], fontsize=fontsize)
    ax.set_ylabel("layer", fontsize=fontsize)

    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.grid(False)

    if title:
        ax.set_title(title, fontsize=fontsize + 1, pad=6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_ticks(list(cbar_ticks))
    cbar.ax.tick_params(labelsize=fontsize)

    plt.tight_layout(rect=(0, 0, 0.97, 1))
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Layer + checkpoint handling
# ---------------------------------------------------------------------------


def parse_layer_spec(spec: str) -> List[int]:
    """Parse ``"0-11"`` / ``"0,3,6"`` / ``"0-11,15"`` into a sorted layer list."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def resolve_sae_paths(
    sae_root: Path, pattern: str, layers: Sequence[int]
) -> List[Tuple[int, Path]]:
    """For each layer, format ``pattern`` and check the directory exists."""
    resolved: List[Tuple[int, Path]] = []
    for layer in layers:
        sub = pattern.format(layer=layer)
        path = sae_root / sub
        if not path.exists():
            logger.warning("missing SAE checkpoint for layer %d: %s", layer, path)
            continue
        resolved.append((layer, path))
    return resolved


def load_sae_stack(
    sae_root: Path,
    pattern: str,
    layers: Sequence[int],
    device: str,
) -> Tuple[List[int], List[SAEModel]]:
    """Load every available SAE in ``layers``. Returns parallel lists."""
    paths = resolve_sae_paths(sae_root, pattern, layers)
    if not paths:
        raise FileNotFoundError(
            f"no SAE checkpoints found under {sae_root} matching {pattern!r} "
            f"for layers {list(layers)}"
        )
    actual_layers: List[int] = []
    saes: List[SAEModel] = []
    for layer_i, path in paths:
        logger.info("loading SAE layer %d from %s", layer_i, path)
        sae = load_sae_for_analysis(path, device=device)
        actual_layers.append(layer_i)
        saes.append(sae)
    return actual_layers, saes


def compute_feature_to_token_maps(
    saes: Sequence[SAEModel],
    W_E: torch.Tensor,
    device: str,
) -> List[torch.Tensor]:
    """For each SAE, compute the vocab token id each feature direction aligns to.

    With ``tied_decoder=False`` (F002 soft variant), the feature index ``f`` is
    only *softly* aligned to vocab id ``f`` via the anchor loss; in general the
    decoder direction ``d_f`` aligns to ``argmax_v cos(d_f, W_E[v])``. This is
    the mapping ``analyze_alignment_quality.py`` calls ``top1_tokens``.

    Returns one ``LongTensor`` of shape ``(n_features,)`` per SAE.
    """
    maps: List[torch.Tensor] = []
    for i, sae in enumerate(saes):
        geo = compute_geometric_alignment(
            get_decoder_features(sae), W_E, top_k=1, device=device
        )
        feat_to_tok = geo.topk_indices[:, 0]  # (n_features,) on cpu
        maps.append(feat_to_tok)
        n_aligned = int((geo.max_sims >= 0.8).sum().item())
        logger.info(
            "  SAE %d/%d: aligned features (>=0.8) %d/%d",
            i + 1,
            len(saes),
            n_aligned,
            geo.max_sims.numel(),
        )
    return maps


# ---------------------------------------------------------------------------
# Per-case forward pass
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_case(
    text: str,
    model,
    tokenizer,
    sae_layers: Sequence[int],
    saes: Sequence[SAEModel],
    feat_to_tok_maps: Sequence[torch.Tensor],
    n_positions: int,
    max_length: int,
    device: str,
    mode: str = "relative",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Encode ``text``, push hidden states through every SAE, return top-1 grids.

    Returns ``(tokens [L, n_show], probs [L, n_show], x_tokens)``, where
    ``L = len(saes)`` and the displayed positions are the first ``n_positions``
    non-padding tokens (or fewer if the text is shorter).
    """
    toks = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
        add_special_tokens=True,
    ).to(device)

    attn = toks.get("attention_mask")
    eff_len = (
        int(attn[0].sum().item()) if attn is not None else toks["input_ids"].shape[1]
    )
    n_show = min(n_positions, eff_len)

    out = model(**toks, output_hidden_states=True)

    token_ids = toks["input_ids"][0].detach().cpu().tolist()
    x_tokens = [tokenizer.decode(token_ids[s]) for s in range(n_show)]

    pred_tokens: List[List[str]] = []
    pred_probs: List[List[float]] = []
    # ``hidden_states`` has L+1 entries (embedding output + L layer outputs).
    # We want the post-residual stream of layer ``sae_layer``, which is
    # ``hidden_states[sae_layer + 1]``. (vasae.ipynb skips index 0 for the same
    # reason.)
    for sae_layer, sae, feat_to_tok in zip(sae_layers, saes, feat_to_tok_maps):
        hs_idx = sae_layer + 1
        if hs_idx >= len(out.hidden_states):
            logger.warning(
                "layer %d out of range (model has %d hidden states)",
                sae_layer,
                len(out.hidden_states),
            )
            continue
        h = out.hidden_states[hs_idx]
        # Match SAE dtype/device.
        target_dtype = next(sae.parameters()).dtype
        _, z = sae.encode(h.to(target_dtype))
        # z: (1, S, n_features). For TopK SAE most entries are zero. We pick a
        # top-1 active feature per position, then map feature → aligned vocab
        # token via the per-SAE feature_to_token map.
        z_cpu = z.detach().float().cpu()[0]  # (S, n_features)
        probs = F.softmax(z_cpu, dim=-1)  # for cell colors
        if mode == "softmax":
            # vasae.ipynb-style: argmax over softmaxed sparse code.
            top_feat_ids = probs.argmax(dim=-1)  # (S,)
        elif mode == "relative":
            # Subtract per-feature mean activation across the sentence's
            # non-pad positions, so we surface features that are *distinctively*
            # active here vs elsewhere — counters the "high-freq feature
            # dominates everywhere" failure mode of softmax mode.
            z_valid = z_cpu[:eff_len]  # (eff_len, F)
            feat_mean = z_valid.mean(dim=0, keepdim=True)  # (1, F)
            z_relative = z_cpu - feat_mean  # (S, F)
            top_feat_ids = z_relative.argmax(dim=-1)  # (S,)
        else:
            raise ValueError(f"unknown mode: {mode!r}")
        # Probability for the cell color: softmax probability of the chosen
        # feature (so colors stay comparable across modes / layers).
        top_probs = probs.gather(1, top_feat_ids.unsqueeze(1)).squeeze(1)  # (S,)
        top_token_ids = feat_to_tok[top_feat_ids[:n_show]]  # (n_show,)
        pred_tokens.append(
            [tokenizer.decode(top_token_ids[s].item()) for s in range(n_show)]
        )
        pred_probs.append(top_probs[:n_show].tolist())

    return (
        np.array(pred_tokens),
        np.array(pred_probs, dtype=np.float32),
        x_tokens,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="gpt2", help="HF model name")
    p.add_argument(
        "--sae-root",
        type=Path,
        default=Path("/scratch/b5bq/pu22650.b5bq/VASAE_out/001_F_Benchmarking"),
        help="parent directory containing per-layer SAE checkpoint subdirs",
    )
    p.add_argument(
        "--sae-pattern",
        default="001F_gpt2_L{layer}_soft",
        help="checkpoint subdir pattern with a {layer} placeholder",
    )
    p.add_argument(
        "--layers",
        default="0-11",
        help='layers to visualize, e.g. "0-11", "0,3,6,9,11", or "0-11,15"',
    )
    p.add_argument(
        "--n-positions",
        type=int,
        default=12,
        help="how many input positions to display per case",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=64,
        help="tokenizer max_length / pad target",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp/F002_AlignmentAnalysis/casestudy/gpt2"),
    )
    p.add_argument(
        "--cases-file",
        type=Path,
        default=None,
        help='optional JSON file: [{"slug": "...", "text": "..."}, ...]; '
        "overrides DEFAULT_CASES",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--mode",
        choices=("relative", "softmax"),
        default="relative",
        help='per-position feature selection: "softmax" reproduces '
        "vasae.ipynb (argmax over softmax(z) — dominated by globally "
        'most-active features); "relative" picks the feature whose '
        "activation deviates most above its per-sentence mean (makes "
        "F002 case studies position-dependent).",
    )
    return p.parse_args()


def load_cases(path: Path | None) -> List[Tuple[str, str]]:
    if path is None:
        return DEFAULT_CASES
    data = json.loads(path.read_text())
    return [(item["slug"], item["text"]) for item in data]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("output dir: %s", args.output_dir)

    layers = parse_layer_spec(args.layers)
    logger.info("requested layers: %s", layers)

    logger.info("loading model %s on %s", args.model, args.device)
    model, tokenizer = load_model(args.model, device=args.device)
    W_E = get_embedding(model).weight.data

    actual_layers, saes = load_sae_stack(
        args.sae_root, args.sae_pattern, layers, args.device
    )
    logger.info("loaded %d SAEs (layers %s)", len(saes), actual_layers)

    logger.info("computing per-SAE feature -> aligned token maps")
    feat_to_tok_maps = compute_feature_to_token_maps(saes, W_E, args.device)

    cases = load_cases(args.cases_file)
    logger.info("running %d case studies", len(cases))

    manifest = []
    for slug, text in cases:
        logger.info("case=%s text=%r", slug, text)
        tokens, probs, x_tokens = run_case(
            text=text,
            model=model,
            tokenizer=tokenizer,
            sae_layers=actual_layers,
            saes=saes,
            feat_to_tok_maps=feat_to_tok_maps,
            n_positions=args.n_positions,
            max_length=args.max_length,
            device=args.device,
            mode=args.mode,
        )
        out_stem = args.output_dir / f"logitlens_{slug}"
        plot_logit_lens(
            tokens_by_layer=tokens,
            probs_by_layer=probs,
            x_tokens=x_tokens,
            layer_labels=actual_layers,
            out_path=out_stem,
            title=slug,
        )
        logger.info("saved %s.{pdf,png}", out_stem)
        manifest.append(
            {
                "slug": slug,
                "text": text,
                "layers": actual_layers,
                "x_tokens": x_tokens,
                "pred_tokens": tokens.tolist(),
                "pred_probs": probs.tolist(),
                "figure": str(out_stem.with_suffix(".pdf")),
            }
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info("wrote manifest %s", manifest_path)


if __name__ == "__main__":
    main()
