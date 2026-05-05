import json
import os
from pathlib import Path
from typing import Iterator

import torch


CORPORA = ("fineweb", "dclm", "pile")


def default_corpus_dir() -> Path:
    vasae_out = os.environ.get("VASAE_OUT")
    if not vasae_out:
        raise ValueError("VASAE_OUT is required unless --corpus-dir is provided")
    return Path(vasae_out) / "Dataset" / "data"


def default_dataset_run_dir() -> Path:
    vasae_out = os.environ.get("VASAE_OUT")
    if not vasae_out:
        raise ValueError("VASAE_OUT is required unless --run-dir is provided")
    return Path(vasae_out) / "Dataset" / "runs" / "gpt2_L5_mixture_soft"


def corpus_jsonl(corpus_dir: Path, corpus: str, split: str) -> Path:
    return corpus_dir / corpus / "raw" / f"{split}.jsonl"


def require_jsonl_paths(paths: dict[str, Path]) -> None:
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing corpus JSONL file(s): {missing}")


def allocate_balanced_token_budgets(
    corpora: list[str] | tuple[str, ...],
    total_token_budget: int,
) -> dict[str, int]:
    if not corpora:
        raise ValueError("At least one corpus is required.")
    if total_token_budget < 0:
        raise ValueError("total_token_budget must be non-negative.")

    per_corpus = total_token_budget // len(corpora)
    budgets = {corpus: per_corpus for corpus in corpora}
    budgets[corpora[-1]] += total_token_budget - per_corpus * len(corpora)
    return budgets


def iter_token_windows(
    path: Path,
    tokenizer,
    *,
    max_length: int,
    token_budget: int,
) -> Iterator[torch.Tensor]:
    if max_length <= 0:
        raise ValueError("max_length must be positive.")
    if token_budget < 0:
        raise ValueError("token_budget must be non-negative.")

    tokens_seen = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if tokens_seen >= token_budget:
                break
            text = json.loads(line)["text"]
            if not text.strip():
                continue
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            for start in range(0, len(ids), max_length):
                if tokens_seen >= token_budget:
                    return
                chunk = ids[start : start + max_length]
                if not chunk:
                    continue
                tokens_seen += len(chunk)
                yield torch.tensor(chunk, dtype=torch.long)


def pad_batch(
    windows: list[torch.Tensor],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    if not windows:
        raise ValueError("Cannot pad an empty batch.")

    max_len = max(int(w.numel()) for w in windows)
    input_ids = torch.full((len(windows), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(windows), max_len), dtype=torch.long)
    for i, window in enumerate(windows):
        n = int(window.numel())
        input_ids[i, :n] = window
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def iter_batches(
    windows: Iterator[torch.Tensor],
    *,
    batch_size: int,
    pad_token_id: int,
) -> Iterator[dict[str, torch.Tensor]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    batch = []
    for window in windows:
        batch.append(window)
        if len(batch) == batch_size:
            yield pad_batch(batch, pad_token_id)
            batch = []
    if batch:
        yield pad_batch(batch, pad_token_id)


class HeldoutCorpusSource:
    def __init__(
        self,
        *,
        model,
        tokenizer,
        layer_idx: int,
        jsonl_path: Path,
        token_budget: int,
        batch_size: int,
        max_length: int,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.jsonl_path = jsonl_path
        self.token_budget = token_budget
        self.batch_size = batch_size
        self.max_length = max_length

    def __iter__(self):
        from vasae.engine.intervention import extract_activations

        windows = iter_token_windows(
            self.jsonl_path,
            self.tokenizer,
            max_length=self.max_length,
            token_budget=self.token_budget,
        )
        for batch in iter_batches(
            windows,
            batch_size=self.batch_size,
            pad_token_id=self.tokenizer.pad_token_id,
        ):
            input_ids = batch["input_ids"].to(self.model.device)
            attention_mask = batch["attention_mask"].to(self.model.device)
            activations = extract_activations(
                self.model,
                input_ids,
                self.layer_idx,
                attention_mask=attention_mask,
            )
            yield {
                "activations": activations,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }


class BalancedMixtureSource:
    def __init__(
        self,
        *,
        model,
        tokenizer,
        layer_idx: int,
        corpus_paths: dict[str, Path],
        total_token_budget: int,
        batch_size: int,
        max_length: int,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.corpus_paths = corpus_paths
        self.total_token_budget = total_token_budget
        self.batch_size = batch_size
        self.max_length = max_length
        require_jsonl_paths(corpus_paths)

    def __iter__(self):
        from vasae.engine.intervention import extract_activations

        corpora = list(self.corpus_paths)
        budgets = allocate_balanced_token_budgets(corpora, self.total_token_budget)

        iterators = {
            corpus: iter(
                iter_batches(
                    iter_token_windows(
                        path,
                        self.tokenizer,
                        max_length=self.max_length,
                        token_budget=budgets[corpus],
                    ),
                    batch_size=self.batch_size,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            )
            for corpus, path in self.corpus_paths.items()
        }

        active = corpora[:]
        while active:
            for corpus in active[:]:
                try:
                    batch = next(iterators[corpus])
                except StopIteration:
                    active.remove(corpus)
                    continue
                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch["attention_mask"].to(self.model.device)
                activations = extract_activations(
                    self.model,
                    input_ids,
                    self.layer_idx,
                    attention_mask=attention_mask,
                )
                yield {
                    "activations": activations,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
