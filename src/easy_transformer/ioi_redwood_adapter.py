"""Adapter for Redwood's vendored IOI dataset generator.

This module wraps the vendored Easy-Transformer IOI prompt generator and
converts its outputs into the simple schema used by our causal evaluation:
clean_text / corrupted_text / correct / wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .ioi_dataset import IOIDataset


@dataclass
class RedwoodIOIExample:
    clean_text: str
    corrupted_text: str
    correct: str
    wrong: str
    pattern: str
    template_idx: int
    subject: str
    indirect_object: str


def normalize_answer_text(text: str) -> str:
    return text if text.startswith(" ") else f" {text}"


def _is_single_token_name(tokenizer, name: str) -> bool:
    return (
        len(tokenizer.encode(normalize_answer_text(name), add_special_tokens=False))
        == 1
    )


def _iter_examples_from_pair(
    clean_ds: IOIDataset, corr_ds: IOIDataset, tokenizer
) -> Iterable[RedwoodIOIExample]:
    for clean_prompt, corr_prompt, pattern in zip(
        clean_ds.ioi_prompts,
        corr_ds.ioi_prompts,
        clean_ds.templates_by_prompt,
    ):
        correct = clean_prompt["IO"]
        wrong = clean_prompt["S"]
        if not (
            _is_single_token_name(tokenizer, correct)
            and _is_single_token_name(tokenizer, wrong)
        ):
            continue
        yield RedwoodIOIExample(
            clean_text=clean_prompt["text"],
            corrupted_text=corr_prompt["text"],
            correct=normalize_answer_text(correct),
            wrong=normalize_answer_text(wrong),
            pattern=pattern,
            template_idx=int(clean_prompt["TEMPLATE_IDX"]),
            subject=clean_prompt["S"],
            indirect_object=clean_prompt["IO"],
        )


def load_redwood_ioi_examples(
    tokenizer,
    n_prompts: int,
    seed: int = 42,
    prompt_type: str = "mixed",
    nb_templates: int | None = None,
    prefixes: list[str] | None = None,
    symmetric: bool = False,
    prepend_bos: bool = False,
) -> list[RedwoodIOIExample]:
    """Generate Redwood-style IOI examples and align clean/corrupted pairs."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    clean_ds = IOIDataset(
        prompt_type=prompt_type,
        N=n_prompts,
        tokenizer=tokenizer,
        symmetric=symmetric,
        prefixes=prefixes,
        nb_templates=nb_templates,
        prepend_bos=prepend_bos,
    )
    corr_ds = clean_ds.gen_flipped_prompts(("S2", "IO"))

    examples = list(_iter_examples_from_pair(clean_ds, corr_ds, tokenizer))
    if len(examples) < n_prompts:
        raise ValueError(
            f"Generated only {len(examples)} valid single-token Redwood IOI examples, "
            f"fewer than requested {n_prompts}."
        )
    return examples[:n_prompts]
