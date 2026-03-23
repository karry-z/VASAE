"""Per-feature causal intervention sweep on IOI prompts.

For each clean IOI prompt, ablate every active VASAE feature one-at-a-time
and measure Recovery (shift toward corrupted baseline) and Specificity
(Recovery / KL divergence).

Example (local test):
    uv run python scripts/eval_ioi_feature_sweep.py \
        --layer-idx 8 --n-prompts 4 --device cpu \
        --sae-root /scratch/b5bq/pu22650.b5bq/VASAE_out/010_soft_align \
        --output-dir /tmp/ioi_sweep_test
"""

from __future__ import annotations

import logging
import os

os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
for _name in ("transformers", "huggingface_hub", "nnsight", "datasets"):
    logging.getLogger(_name).setLevel(logging.WARNING)

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from nnsight import NNsight

from easy_transformer.ioi_redwood_adapter import load_redwood_ioi_examples
from vasae.engine.intervention import extract_activations, patch_and_forward
from vasae.models.factory import load_model
from vasae.models.sae import SAEModel
from vasae.utils.log import get_logger
from vasae.utils.seed import set_seed

logger = get_logger("eval_ioi_feature_sweep")

EPSILON = 1e-8


@dataclass
class IOIExample:
    clean_text: str
    corrupted_text: str
    correct: str
    wrong: str


def load_examples(tokenizer, n_prompts: int, seed: int) -> List[IOIExample]:
    redwood_examples = load_redwood_ioi_examples(
        tokenizer=tokenizer,
        n_prompts=n_prompts,
        seed=seed,
        prompt_type="mixed",
    )
    return [
        IOIExample(
            clean_text=ex.clean_text,
            corrupted_text=ex.corrupted_text,
            correct=ex.correct,
            wrong=ex.wrong,
        )
        for ex in redwood_examples
    ]


def validate_answer_tokens(tokenizer, examples: List[IOIExample]):
    bad = []
    for ex in examples:
        for token_text in [ex.correct, ex.wrong]:
            ids = tokenizer.encode(token_text, add_special_tokens=False)
            if len(ids) != 1:
                bad.append((token_text, ids))
    if bad:
        details = ", ".join(f"{tok!r}->{ids}" for tok, ids in bad[:8])
        raise ValueError(f"All answer names must be single-token. Bad: {details}")


def answer_token_ids(tokenizer, examples: List[IOIExample], device: torch.device):
    correct_ids, wrong_ids = [], []
    for ex in examples:
        correct_ids.append(tokenizer.encode(ex.correct, add_special_tokens=False)[0])
        wrong_ids.append(tokenizer.encode(ex.wrong, add_special_tokens=False)[0])
    return (
        torch.tensor(correct_ids, device=device, dtype=torch.long),
        torch.tensor(wrong_ids, device=device, dtype=torch.long),
    )


def tokenize_texts(tokenizer, texts: List[str], device: torch.device):
    return tokenizer(texts, return_tensors="pt", padding=True).to(device)


def last_nonpad_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.sum(dim=1) - 1


def gather_final_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    pos = last_nonpad_positions(attention_mask)
    batch_idx = torch.arange(logits.size(0), device=logits.device)
    return logits[batch_idx, pos]


def compute_logit_diff(logits: torch.Tensor, attention_mask: torch.Tensor,
                       correct_ids: torch.Tensor, wrong_ids: torch.Tensor) -> torch.Tensor:
    final_logits = gather_final_logits(logits, attention_mask)
    batch_idx = torch.arange(final_logits.size(0), device=logits.device)
    return final_logits[batch_idx, correct_ids] - final_logits[batch_idx, wrong_ids]


def kl_div_at_last_pos(
    clean_logits: torch.Tensor,
    intervened_logits: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """KL(P_clean || P_intervened) at the last non-pad position.

    Returns a scalar (mean over the batch dimension that was already reduced
    to 1 by the caller).
    """
    clean_final = gather_final_logits(clean_logits, attention_mask)       # (V,)
    interv_final = gather_final_logits(intervened_logits, attention_mask)  # (V,)
    log_p = F.log_softmax(clean_final, dim=-1)
    log_q = F.log_softmax(interv_final, dim=-1)
    # KL(P || Q) = sum P * (log P - log Q)
    kl = F.kl_div(log_q, log_p, log_target=True, reduction="batchmean")
    return kl


def get_active_features(z: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Return indices of nonzero features at the last non-pad position.

    Args:
        z: (seq, dim_sparse) sparse activations for a single example
        attention_mask: (seq,) mask
    Returns:
        1-D tensor of feature indices
    """
    last_pos = int(attention_mask.sum() - 1)
    z_last = z[last_pos]  # (dim_sparse,)
    return z_last.nonzero(as_tuple=False).squeeze(-1)


@torch.no_grad()
def sweep_single_example(
    nn_model: NNsight,
    sae_model: SAEModel,
    tokenizer,
    layer_idx: int,
    example: IOIExample,
    device: torch.device,
) -> dict:
    """Run one-at-a-time feature ablation for a single IOI example."""

    # --- baselines ---
    clean_enc = tokenize_texts(tokenizer, [example.clean_text], device)
    corr_enc = tokenize_texts(tokenizer, [example.corrupted_text], device)
    correct_ids, wrong_ids = answer_token_ids(tokenizer, [example], device)

    logits_clean = patch_and_forward(
        nn_model, clean_enc["input_ids"], clean_enc["attention_mask"],
        layer_idx, lambda h: h,
    )
    logits_corr = patch_and_forward(
        nn_model, corr_enc["input_ids"], corr_enc["attention_mask"],
        layer_idx, lambda h: h,
    )

    du_clean = compute_logit_diff(logits_clean, clean_enc["attention_mask"], correct_ids, wrong_ids).item()
    du_corr = compute_logit_diff(logits_corr, corr_enc["attention_mask"], correct_ids, wrong_ids).item()

    # --- encode clean activations ---
    h_clean = extract_activations(nn_model, clean_enc["input_ids"], layer_idx)
    _, z_clean = sae_model.encode(h_clean.squeeze(0))  # (seq, dim_sparse)

    active_feats = get_active_features(z_clean, clean_enc["attention_mask"].squeeze(0))
    K = active_feats.numel()

    # --- SAE reconstruction baseline (no ablation) ---
    h_recon = sae_model.decode(z_clean).unsqueeze(0)  # (1, seq, dim_input)
    logits_recon = patch_and_forward(
        nn_model, clean_enc["input_ids"], clean_enc["attention_mask"],
        layer_idx, lambda h, _hr=h_recon: _hr,
    )
    du_recon = compute_logit_diff(logits_recon, clean_enc["attention_mask"], correct_ids, wrong_ids).item()

    if K == 0:
        return {
            "clean_text": example.clean_text,
            "corrupted_text": example.corrupted_text,
            "correct": example.correct,
            "wrong": example.wrong,
            "du_clean": du_clean,
            "du_corr": du_corr,
            "du_recon": du_recon,
            "features": [],
        }

    gap = du_clean - du_corr
    feat_indices = active_feats.tolist()

    # Activation strengths at last position
    last_pos = int(clean_enc["attention_mask"].squeeze(0).sum() - 1)
    z_last = z_clean[last_pos]

    feature_results = []
    for fid in feat_indices:
        # Zero out one feature and decode
        z_ablated = z_clean.clone()
        z_ablated[:, fid] = 0.0
        h_ablated = sae_model.decode(z_ablated).unsqueeze(0)  # (1, seq, dim_input)

        logits_i = patch_and_forward(
            nn_model, clean_enc["input_ids"], clean_enc["attention_mask"],
            layer_idx, lambda h, _ha=h_ablated: _ha,
        )
        attn_i = clean_enc["attention_mask"]

        du_interv = compute_logit_diff(logits_i, attn_i, correct_ids, wrong_ids).item()
        # Use du_recon (not du_clean) as numerator baseline to isolate feature effect
        effect = du_recon - du_interv
        recovery = effect / (gap + EPSILON) if abs(gap) > EPSILON else 0.0
        kl = kl_div_at_last_pos(logits_recon, logits_i, attn_i).item()
        specificity = recovery / (kl + EPSILON) if kl > EPSILON else 0.0

        feature_results.append({
            "feature_id": fid,
            "strength": float(z_last[fid].item()),
            "du_clean": du_clean,
            "du_corr": du_corr,
            "du_recon": du_recon,
            "du_intervened": du_interv,
            "effect": effect,
            "recovery": recovery,
            "kl_divergence": kl,
            "specificity": specificity,
        })

    return {
        "clean_text": example.clean_text,
        "corrupted_text": example.corrupted_text,
        "correct": example.correct,
        "wrong": example.wrong,
        "du_clean": du_clean,
        "du_corr": du_corr,
        "du_recon": du_recon,
        "features": feature_results,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Per-feature IOI causal intervention sweep")
    p.add_argument("--layer-idx", type=int, required=True)
    p.add_argument("--model-name", type=str, default="gpt2")
    p.add_argument("--sae-root", type=str, required=True,
                   help="Root dir containing 010_soft_gpt2_L{layer}_k32_a1e-3 subdirs")
    p.add_argument("--n-prompts", type=int, default=100)
    p.add_argument("--min-gap", type=float, default=0.5,
                   help="Minimum |du_clean - du_corr| to include a prompt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    logger.info("Loading LLM %s on %s", args.model_name, device)
    llm, tokenizer = load_model(args.model_name, device=str(device))
    nn_model = NNsight(llm)

    sae_dir = Path(args.sae_root) / f"010_soft_gpt2_L{args.layer_idx}_k32_a1e-3"
    sae_path = str(sae_dir.resolve())
    logger.info("Loading SAE from %s", sae_path)
    sae_model = SAEModel.from_pretrained(sae_path).to(device)
    sae_model.eval()

    examples = load_examples(tokenizer, args.n_prompts, args.seed)
    if not examples:
        raise ValueError("No IOI examples available.")
    validate_answer_tokens(tokenizer, examples)
    logger.info("Loaded %d examples", len(examples))

    # Pre-filter: compute baselines and skip prompts with small gap or du_clean <= 0
    valid_examples = []
    for idx, ex in enumerate(examples):
        clean_enc = tokenize_texts(tokenizer, [ex.clean_text], device)
        corr_enc = tokenize_texts(tokenizer, [ex.corrupted_text], device)
        correct_ids, wrong_ids = answer_token_ids(tokenizer, [ex], device)
        with torch.no_grad():
            logits_clean = patch_and_forward(
                nn_model, clean_enc["input_ids"], clean_enc["attention_mask"],
                args.layer_idx, lambda h: h,
            )
            logits_corr = patch_and_forward(
                nn_model, corr_enc["input_ids"], corr_enc["attention_mask"],
                args.layer_idx, lambda h: h,
            )
        du_clean = compute_logit_diff(logits_clean, clean_enc["attention_mask"], correct_ids, wrong_ids).item()
        du_corr = compute_logit_diff(logits_corr, corr_enc["attention_mask"], correct_ids, wrong_ids).item()
        gap = du_clean - du_corr
        if du_clean <= 0:
            logger.info("Skipping example %d: du_clean=%.4f <= 0", idx, du_clean)
            continue
        if abs(gap) < args.min_gap:
            logger.info("Skipping example %d: |gap|=%.4f < %.2f", idx, abs(gap), args.min_gap)
            continue
        valid_examples.append(ex)
    logger.info("Valid examples after filtering: %d / %d", len(valid_examples), len(examples))

    results = []
    for idx, ex in enumerate(valid_examples):
        logger.info("Processing example %d/%d", idx + 1, len(valid_examples))
        res = sweep_single_example(nn_model, sae_model, tokenizer, args.layer_idx, ex, device)
        results.append(res)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"layer_{args.layer_idx}.json"
    with out_path.open("w") as f:
        json.dump({
            "layer_idx": args.layer_idx,
            "n_prompts": len(valid_examples),
            "n_prompts_total": len(examples),
            "model_name": args.model_name,
            "sae_path": sae_path,
            "examples": results,
        }, f, indent=2)
    logger.info("Saved results to %s", out_path)


if __name__ == "__main__":
    main()
