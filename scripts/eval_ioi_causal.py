"""Evaluate causal effects of VASAE features on the IOI task.

This script measures whether SAE features are causally related to model
behavior on Indirect Object Identification (IOI) prompts via:

1. Baseline next-token preference on clean / corrupted prompts
2. SAE reconstruction at a chosen layer
3. Feature ablation on clean prompts
4. Feature transplant from clean -> corrupted prompts
5. Random-feature control

The main metric is:
    logit_diff = logit(correct_name) - logit(incorrect_name)

Example:
    python scripts/eval_ioi_causal.py \
        --sae-path out/debug_online/debug_online \
        --model-name gpt2 \
        --layer-idx 1 \
        --feature-select top_clean_minus_corr \
        --topk-features 4 \
        --batch-size 2 \
        --max-examples 8 \
        --device cpu
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from nnsight import NNsight

from vasae.engine.intervention import extract_activations, patch_and_forward
from vasae.models.factory import load_model
from vasae.models.sae import SAEModel
from vasae.utils.seed import set_seed


#  dataset example
class IOIExample:
    clean_text: str
    corrupted_text: str
    correct: str
    wrong: str


def build_builtin_ioi_dataset() -> List[IOIExample]:
    names = [
        (" John", " Mary"),
        (" Tom", " James"),
        (" Alice", " Sarah"),
        (" Robert", " Daniel"),
        (" Anna", " Emma"),
        (" Michael", " David"),
        (" Lisa", " Karen"),
        (" Peter", " Martin"),
    ]
    templates = [
        "When{name_a} and{name_b} went to the store,{name_a} gave a book to",
        "After{name_a} met{name_b} at the park,{name_a} handed the keys to",
        "While{name_a} and{name_b} were working together,{name_a} sent a message to",
        "Because{name_a} trusted{name_b},{name_a} showed the letter to",
    ]

    data: List[IOIExample] = []
    for i, (name_a, name_b) in enumerate(names):
        template = templates[i % len(templates)]
        clean = template.format(name_a=name_a, name_b=name_b)
        corrupted = template.format(name_a=name_b, name_b=name_a)
        data.append(
            IOIExample(
                clean_text=clean,
                corrupted_text=corrupted,
                correct=name_b,
                wrong=name_a,
            )
        )
    return data


def load_jsonl_dataset(path: Path) -> List[IOIExample]:
    examples = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            examples.append(IOIExample(**obj))
    return examples


def parse_args():
    p = argparse.ArgumentParser(description="Causal IOI evaluation for VASAE features")
    p.add_argument("--sae-path", type=str, required=True, help="Path to save_pretrained SAE dir")
    p.add_argument("--model-name", type=str, default="gpt2")
    p.add_argument("--layer-idx", type=int, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default=None, choices=["float16", "bfloat16", "float32"])
    p.add_argument("--dataset-jsonl", type=str, default=None)
    p.add_argument("--max-examples", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--feature-select",
        type=str,
        default="top_clean_minus_corr",
        choices=["manual", "top_clean", "top_clean_minus_corr"],
    )
    p.add_argument("--feature-ids", type=int, nargs="*", default=None)
    p.add_argument("--topk-features", type=int, default=8)
    p.add_argument("--random-control-seed", type=int, default=123)
    p.add_argument("--output-path", type=str, default=None)
    return p.parse_args()


def get_dtype(dtype_name: str | None):
    if dtype_name is None:
        return None
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def batch_iter(examples: List[IOIExample], batch_size: int) -> Iterable[List[IOIExample]]:
    for i in range(0, len(examples), batch_size):
        yield examples[i:i + batch_size]


def last_nonpad_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.sum(dim=1) - 1


def gather_final_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    pos = last_nonpad_positions(attention_mask)
    batch_idx = torch.arange(logits.size(0), device=logits.device)
    return logits[batch_idx, pos]


def tokenize_texts(tokenizer, texts: List[str], device: torch.device):
    return tokenizer(texts, return_tensors="pt", padding=True).to(device)


def validate_answer_tokens(tokenizer, examples: List[IOIExample]):
    bad = []
    for ex in examples:
        for token_text in [ex.correct, ex.wrong]:
            ids = tokenizer.encode(token_text, add_special_tokens=False)
            if len(ids) != 1:
                bad.append((token_text, ids))
    if bad:
        details = ", ".join(f"{tok!r}->{ids}" for tok, ids in bad[:8])
        raise ValueError(f"All answer names must be single-token. Bad examples: {details}")


def answer_token_ids(tokenizer, examples: List[IOIExample], device: torch.device):
    correct_ids = []
    wrong_ids = []
    for ex in examples:
        correct_ids.append(tokenizer.encode(ex.correct, add_special_tokens=False)[0])
        wrong_ids.append(tokenizer.encode(ex.wrong, add_special_tokens=False)[0])
    return (
        torch.tensor(correct_ids, device=device, dtype=torch.long),
        torch.tensor(wrong_ids, device=device, dtype=torch.long),
    )


def compute_logit_diff(logits: torch.Tensor, attention_mask: torch.Tensor,
                       correct_ids: torch.Tensor, wrong_ids: torch.Tensor) -> torch.Tensor:
    final_logits = gather_final_logits(logits, attention_mask)
    batch_idx = torch.arange(final_logits.size(0), device=logits.device)
    return final_logits[batch_idx, correct_ids] - final_logits[batch_idx, wrong_ids]


@torch.no_grad()
def get_latents_and_recon(sae_model: SAEModel, h: torch.Tensor):
    out = sae_model(h)
    return out.sparse_activations, out.hidden_states_recon


def select_feature_ids(z_clean: torch.Tensor, z_corr: torch.Tensor, args) -> List[int]:
    flat_clean = z_clean.reshape(-1, z_clean.size(-1))
    flat_corr = z_corr.reshape(-1, z_corr.size(-1))

    if args.feature_select == "manual":
        if not args.feature_ids:
            raise ValueError("--feature-select manual requires --feature-ids")
        return sorted(set(args.feature_ids))

    if args.feature_select == "top_clean":
        scores = flat_clean.mean(dim=0)
    elif args.feature_select == "top_clean_minus_corr":
        scores = (flat_clean - flat_corr).mean(dim=0)
    else:
        raise ValueError(args.feature_select)

    topk = min(args.topk_features, scores.numel())
    feature_ids = torch.topk(scores, k=topk).indices.tolist()
    return sorted(feature_ids)


def random_feature_ids(dim_sparse: int, n_features: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    return sorted(rng.sample(range(dim_sparse), k=n_features))


@torch.no_grad()
def collect_dataset_latents(
    nn_model: NNsight,
    sae_model: SAEModel,
    tokenizer,
    layer_idx: int,
    examples: List[IOIExample],
    device: torch.device,
    batch_size: int,
):
    z_clean_all = []
    z_corr_all = []
    for batch_examples in batch_iter(examples, batch_size):
        clean_texts = [ex.clean_text for ex in batch_examples]
        corr_texts = [ex.corrupted_text for ex in batch_examples]
        clean_batch = tokenize_texts(tokenizer, clean_texts, device)
        corr_batch = tokenize_texts(tokenizer, corr_texts, device)

        h_clean = extract_activations(nn_model, clean_batch["input_ids"], layer_idx)
        h_corr = extract_activations(nn_model, corr_batch["input_ids"], layer_idx)
        z_clean, _ = get_latents_and_recon(sae_model, h_clean)
        z_corr, _ = get_latents_and_recon(sae_model, h_corr)
        z_clean_all.append(z_clean.cpu())
        z_corr_all.append(z_corr.cpu())

    return torch.cat(z_clean_all, dim=0), torch.cat(z_corr_all, dim=0)


@torch.no_grad()
def eval_batch(
    nn_model: NNsight,
    sae_model: SAEModel,
    tokenizer,
    layer_idx: int,
    examples: List[IOIExample],
    device: torch.device,
    feature_ids: List[int],
    control_ids: List[int],
) -> Dict:
    clean_texts = [ex.clean_text for ex in examples]
    corr_texts = [ex.corrupted_text for ex in examples]
    clean_batch = tokenize_texts(tokenizer, clean_texts, device)
    corr_batch = tokenize_texts(tokenizer, corr_texts, device)
    correct_ids, wrong_ids = answer_token_ids(tokenizer, examples, device)

    logits_clean_base = patch_and_forward(
        nn_model,
        clean_batch["input_ids"],
        clean_batch["attention_mask"],
        layer_idx,
        lambda h: h,
    )
    logits_corr_base = patch_and_forward(
        nn_model,
        corr_batch["input_ids"],
        corr_batch["attention_mask"],
        layer_idx,
        lambda h: h,
    )

    h_clean = extract_activations(nn_model, clean_batch["input_ids"], layer_idx)
    h_corr = extract_activations(nn_model, corr_batch["input_ids"], layer_idx)
    z_clean, h_clean_recon = get_latents_and_recon(sae_model, h_clean)
    z_corr, h_corr_recon = get_latents_and_recon(sae_model, h_corr)

    def apply_ablation(z_source: torch.Tensor, ids: List[int]) -> torch.Tensor:
        z_new = z_source.clone()
        z_new[..., ids] = 0.0
        return sae_model.decode(z_new)

    def apply_transplant(z_source: torch.Tensor, z_donor: torch.Tensor, ids: List[int]) -> torch.Tensor:
        z_new = z_source.clone()
        z_new[..., ids] = z_donor[..., ids]
        return sae_model.decode(z_new)

    logits_clean_recon = patch_and_forward(
        nn_model,
        clean_batch["input_ids"],
        clean_batch["attention_mask"],
        layer_idx,
        lambda h: h_clean_recon,
    )
    logits_clean_ablate = patch_and_forward(
        nn_model,
        clean_batch["input_ids"],
        clean_batch["attention_mask"],
        layer_idx,
        lambda h: apply_ablation(z_clean, feature_ids),
    )
    logits_clean_rand = patch_and_forward(
        nn_model,
        clean_batch["input_ids"],
        clean_batch["attention_mask"],
        layer_idx,
        lambda h: apply_ablation(z_clean, control_ids),
    )
    logits_corr_recon = patch_and_forward(
        nn_model,
        corr_batch["input_ids"],
        corr_batch["attention_mask"],
        layer_idx,
        lambda h: h_corr_recon,
    )
    logits_corr_transplant = patch_and_forward(
        nn_model,
        corr_batch["input_ids"],
        corr_batch["attention_mask"],
        layer_idx,
        lambda h: apply_transplant(z_corr, z_clean, feature_ids),
    )
    logits_corr_rand_transplant = patch_and_forward(
        nn_model,
        corr_batch["input_ids"],
        corr_batch["attention_mask"],
        layer_idx,
        lambda h: apply_transplant(z_corr, z_clean, control_ids),
    )

    clean_base = compute_logit_diff(logits_clean_base, clean_batch["attention_mask"], correct_ids, wrong_ids)
    clean_recon = compute_logit_diff(logits_clean_recon, clean_batch["attention_mask"], correct_ids, wrong_ids)
    clean_ablate = compute_logit_diff(logits_clean_ablate, clean_batch["attention_mask"], correct_ids, wrong_ids)
    clean_rand = compute_logit_diff(logits_clean_rand, clean_batch["attention_mask"], correct_ids, wrong_ids)
    corr_base = compute_logit_diff(logits_corr_base, corr_batch["attention_mask"], correct_ids, wrong_ids)
    corr_recon = compute_logit_diff(logits_corr_recon, corr_batch["attention_mask"], correct_ids, wrong_ids)
    corr_trans = compute_logit_diff(
        logits_corr_transplant, corr_batch["attention_mask"], correct_ids, wrong_ids
    )
    corr_rand_trans = compute_logit_diff(
        logits_corr_rand_transplant, corr_batch["attention_mask"], correct_ids, wrong_ids
    )

    gap = clean_base - corr_base
    transplant_recovery = (corr_trans - corr_base) / gap.clamp(min=1e-8)
    random_recovery = (corr_rand_trans - corr_base) / gap.clamp(min=1e-8)

    return {
        "feature_ids": feature_ids,
        "random_feature_ids": control_ids,
        "clean_baseline": clean_base.cpu(),
        "clean_recon": clean_recon.cpu(),
        "clean_ablate": clean_ablate.cpu(),
        "clean_random_ablate": clean_rand.cpu(),
        "corr_baseline": corr_base.cpu(),
        "corr_recon": corr_recon.cpu(),
        "corr_transplant": corr_trans.cpu(),
        "corr_random_transplant": corr_rand_trans.cpu(),
        "transplant_recovery": transplant_recovery.cpu(),
        "random_recovery": random_recovery.cpu(),
    }


def tensor_mean(xs: List[torch.Tensor]) -> float:
    return torch.cat(xs).mean().item() if xs else float("nan")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    if args.dataset_jsonl:
        examples = load_jsonl_dataset(Path(args.dataset_jsonl))
    else:
        examples = build_builtin_ioi_dataset()
    if args.max_examples > 0:
        examples = examples[:args.max_examples]
    if not examples:
        raise ValueError("No IOI examples available.")

    dtype = get_dtype(args.dtype)
    llm, tokenizer = load_model(args.model_name, device=str(device), dtype=dtype)
    nn_model = NNsight(llm)
    sae_model = SAEModel.from_pretrained(args.sae_path).to(device)
    sae_model.eval()

    validate_answer_tokens(tokenizer, examples)
    z_clean_all, z_corr_all = collect_dataset_latents(
        nn_model, sae_model, tokenizer, args.layer_idx, examples, device, args.batch_size
    )
    feature_ids = select_feature_ids(z_clean_all, z_corr_all, args)
    control_ids = random_feature_ids(z_clean_all.size(-1), len(feature_ids), args.random_control_seed)

    agg: Dict[str, List[torch.Tensor]] = {
        "clean_baseline": [],
        "clean_recon": [],
        "clean_ablate": [],
        "clean_random_ablate": [],
        "corr_baseline": [],
        "corr_recon": [],
        "corr_transplant": [],
        "corr_random_transplant": [],
        "transplant_recovery": [],
        "random_recovery": [],
    }
    per_example = []
    for batch_examples in batch_iter(examples, args.batch_size):
        batch_out = eval_batch(
            nn_model,
            sae_model,
            tokenizer,
            args.layer_idx,
            batch_examples,
            device,
            feature_ids,
            control_ids,
        )
        for key in agg:
            agg[key].append(batch_out[key])

        for i, ex in enumerate(batch_examples):
            per_example.append({
                **asdict(ex),
                "clean_baseline": float(batch_out["clean_baseline"][i]),
                "clean_recon": float(batch_out["clean_recon"][i]),
                "clean_ablate": float(batch_out["clean_ablate"][i]),
                "clean_random_ablate": float(batch_out["clean_random_ablate"][i]),
                "corr_baseline": float(batch_out["corr_baseline"][i]),
                "corr_recon": float(batch_out["corr_recon"][i]),
                "corr_transplant": float(batch_out["corr_transplant"][i]),
                "corr_random_transplant": float(batch_out["corr_random_transplant"][i]),
                "transplant_recovery": float(batch_out["transplant_recovery"][i]),
                "random_recovery": float(batch_out["random_recovery"][i]),
            })

    summary = {
        "config": vars(args),
        "n_examples": len(examples),
        "selected_feature_ids": feature_ids,
        "random_feature_ids": control_ids,
        "metrics": {k: tensor_mean(v) for k, v in agg.items()},
        "derived": {
            "ablation_drop": tensor_mean(agg["clean_baseline"]) - tensor_mean(agg["clean_ablate"]),
            "random_ablation_drop": tensor_mean(agg["clean_baseline"]) - tensor_mean(agg["clean_random_ablate"]),
            "transplant_gain": tensor_mean(agg["corr_transplant"]) - tensor_mean(agg["corr_baseline"]),
            "random_transplant_gain": tensor_mean(agg["corr_random_transplant"]) - tensor_mean(agg["corr_baseline"]),
        },
        "per_example": per_example,
    }

    print(json.dumps(summary, indent=2))
    if args.output_path:
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
