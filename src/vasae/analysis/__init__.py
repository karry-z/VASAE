"""Analysis utilities for VASAE scripts.

Shared computations extracted from scripts/ to reduce duplication.
"""

from vasae.analysis.alignment import (
    GeometricAlignmentResult,
    LogitAttributionResult,
    compute_geometric_alignment,
    compute_logit_attribution,
)
from vasae.analysis.hooks import make_intervention_hook, run_with_hook
from vasae.analysis.io import (
    discover_checkpoints,
    load_layer_results,
    save_figure,
    save_results,
    setup_matplotlib,
)
from vasae.analysis.sae_loader import get_decoder_features, load_sae_for_analysis
from vasae.analysis.stats import summarize_tensor

__all__ = [
    "GeometricAlignmentResult",
    "LogitAttributionResult",
    "compute_geometric_alignment",
    "compute_logit_attribution",
    "make_intervention_hook",
    "run_with_hook",
    "discover_checkpoints",
    "load_layer_results",
    "save_figure",
    "save_results",
    "setup_matplotlib",
    "get_decoder_features",
    "load_sae_for_analysis",
    "summarize_tensor",
]
