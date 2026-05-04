import json

import pytest
import torch

from scripts.aggregate.summarize_dataset_results import compute_alive_overlap
from vasae.data.corpus_windows import (
    allocate_balanced_token_budgets,
    iter_token_windows,
    pad_batch,
)


class WhitespaceTokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [int(token) for token in text.split()]}


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_iter_token_windows_respects_window_and_budget(tmp_path):
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [{"text": "1 2 3 4 5"}, {"text": "6 7"}])

    windows = list(
        iter_token_windows(
            path,
            WhitespaceTokenizer(),
            max_length=2,
            token_budget=4,
        )
    )

    assert [window.tolist() for window in windows] == [[1, 2], [3, 4]]


def test_pad_batch_builds_attention_mask():
    batch = pad_batch(
        [torch.tensor([1, 2, 3]), torch.tensor([4])],
        pad_token_id=0,
    )

    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_allocate_balanced_token_budgets_assigns_remainder_to_last_corpus():
    budgets = allocate_balanced_token_budgets(("fineweb", "dclm", "pile"), 10)

    assert budgets == {"fineweb": 3, "dclm": 3, "pile": 4}


def test_allocate_balanced_token_budgets_requires_corpus():
    with pytest.raises(ValueError, match="At least one corpus"):
        allocate_balanced_token_budgets((), 10)


def test_compute_alive_overlap_reports_pairwise_and_all_three_intersections():
    overlap = compute_alive_overlap(
        {
            "fineweb": {1, 2, 3},
            "dclm": {2, 3, 4},
            "pile": {3, 4, 5},
        }
    )

    assert overlap["fineweb_dclm"]["intersection"] == 2
    assert overlap["fineweb_dclm"]["union"] == 4
    assert overlap["fineweb_dclm"]["jaccard"] == pytest.approx(0.5)
    assert overlap["all_three"]["intersection"] == 1
