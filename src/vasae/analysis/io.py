"""I/O utilities for analysis scripts: checkpoint discovery, result saving, plot setup."""

import json
import re
from pathlib import Path
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def discover_checkpoints(
    results_dir: str | Path,
    pattern: str | re.Pattern,
    group_index: int = 1,
) -> dict[int, Path]:
    """Scan *results_dir* for sub-directories whose names match *pattern*.

    Args:
        results_dir: directory to scan.
        pattern: regex with at least one capture group for the layer index.
            Can be a compiled ``re.Pattern`` or a string.
        group_index: which capture group contains the layer index (1-indexed).

    Returns:
        dict mapping ``layer_idx -> Path``.
    """
    results_path = Path(results_dir)
    if isinstance(pattern, str):
        pattern = re.compile(pattern)

    checkpoints: dict[int, Path] = {}
    for d in sorted(results_path.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if m:
            layer = int(m.group(group_index))
            checkpoints[layer] = d
    return checkpoints


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------


def save_results(
    output_dir: str | Path,
    json_data: dict[str, Any],
    json_filename: str = "results.json",
    tensors: dict[str, torch.Tensor] | None = None,
    tensor_filename: str = "tensors.pt",
) -> None:
    """Save JSON results and optional PT tensors to a directory.

    Creates *output_dir* if it does not exist.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / json_filename, "w") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    if tensors is not None:
        torch.save(tensors, output_dir / tensor_filename)


def load_layer_results(input_dir: str | Path) -> dict[int, dict]:
    """Load per-layer JSON results from a ``L0/``, ``L1/``, ... directory structure.

    Returns dict mapping ``layer_idx -> parsed JSON data``.
    """
    input_dir = Path(input_dir)
    results: dict[int, dict] = {}
    for d in sorted(input_dir.iterdir()):
        if d.is_dir() and d.name.startswith("L"):
            rpath = d / "results.json"
            if rpath.exists():
                with open(rpath) as f:
                    r = json.load(f)
                    results[r["layer_idx"]] = r
    return results


# ---------------------------------------------------------------------------
# Plot setup
# ---------------------------------------------------------------------------


def setup_matplotlib():
    """Configure matplotlib for non-interactive (Agg) backend."""
    import matplotlib

    matplotlib.use("Agg")


def save_figure(
    fig,
    output_dir: str | Path,
    name: str,
    dpi: int = 150,
) -> None:
    """Save *fig* as both PNG and PDF.

    Args:
        fig: matplotlib figure.
        output_dir: target directory.
        name: base name without extension (e.g. ``"fig1_distribution"``).
        dpi: DPI for the PNG variant.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
