import json
import logging
import os
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorpusSource:
    name: str
    dataset: str
    config: str
    split: str
    text_column: str


DEFAULT_CORPUS_SOURCES: dict[str, CorpusSource] = {
    "dclm": CorpusSource(
        name="dclm",
        dataset="mlfoundations/dclm-baseline-1.0-parquet",
        config="default",
        split="train",
        text_column="text",
    ),
    "fineweb": CorpusSource(
        name="fineweb",
        dataset="HuggingFaceFW/fineweb",
        config="sample-10BT",
        split="train",
        text_column="text",
    ),
    "pile": CorpusSource(
        name="pile",
        dataset="gmongaras/EleutherAI_the_pile_deduplicated",
        config="default",
        split="train",
        text_column="text",
    ),
}


@dataclass(frozen=True)
class CountedDocument:
    source_index: int
    text: str
    token_count: int
    metadata: dict[str, Any]


def default_corpus_out_dir() -> Path:
    vasae_out = os.environ.get("VASAE_OUT")
    if vasae_out is None:
        logger.error("env var `VASAE_OUT` does not exists.")
        raise ValueError("env var `VASAE_OUT` does not exists.")
    dataset_dir = Path(vasae_out) / "Dataset" / "data"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def metadata_without_text(example: dict[str, Any], text_column: str) -> dict[str, Any]:
    metadata = {key: value for key, value in example.items() if key != text_column}
    return {
        key: value
        for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }


def write_jsonl_document(
    handle,
    *,
    source: str,
    split: str,
    source_index: int,
    text: str,
    token_count: int,
    metadata: dict[str, Any],
) -> None:
    record = {
        "source": source,
        "split": split,
        "source_index": source_index,
        "token_count": token_count,
        "text": text,
        "metadata": metadata,
    }
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_fetch_args(
    *,
    train_token_target: int,
    heldout_token_target: int,
    max_docs: int | None,
    batch_size: int,
) -> None:
    """Validate required numeric invariants before the fetch loop starts."""
    if train_token_target < 0 or heldout_token_target < 0:
        raise ValueError("token targets must be non-negative.")
    if max_docs is not None and max_docs < 0:
        raise ValueError("max_docs must be non-negative or None.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")


def stream_tokenizer_batches(
    stream,
    *,
    source: CorpusSource,
    batch_size: int,
    max_docs: int | None,
    start_index: int = 0,
) -> Iterator[tuple[list[int], list[str], list[dict[str, Any]]]]:
    """Yield tokenizer-ready batches from an example iterator."""
    source_index = start_index
    iterator = iter(stream)

    try:
        while max_docs is None or source_index < max_docs:
            remaining_docs = None if max_docs is None else max_docs - source_index
            current_batch_size = (
                batch_size
                if remaining_docs is None
                else min(batch_size, remaining_docs)
            )
            rows = []
            for _ in range(current_batch_size):
                try:
                    rows.append(next(iterator))
                except StopIteration:
                    break

            if not rows:
                break

            source_indices = list(range(source_index, source_index + len(rows)))
            texts = [str(example[source.text_column]) for example in rows]
            metadata_items = [
                metadata_without_text(example, source.text_column) for example in rows
            ]
            source_index += len(rows)

            yield source_indices, texts, metadata_items

            if len(rows) < current_batch_size:
                break
    finally:
        # Hugging Face streaming iterators can keep background HTTP readers alive.
        # Explicitly closing them after early stopping has produced noisy shutdown
        # errors on some parquet streams; the process exits immediately after
        # corpus fetching, so normal generator teardown is sufficient here.
        pass


def count_tokens_for_streamed_docs(
    stream,
    *,
    source: CorpusSource,
    tokenizer,
    batch_size: int,
    max_docs: int | None,
    start_index: int = 0,
) -> Iterator[CountedDocument]:
    """Yield documents with token counts, ready to be assigned to a split."""
    for source_indices, texts, metadata_items in stream_tokenizer_batches(
        stream,
        source=source,
        batch_size=batch_size,
        max_docs=max_docs,
        start_index=start_index,
    ):
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )

        for source_index, text, metadata, input_ids in zip(
            source_indices, texts, metadata_items, encoded["input_ids"], strict=True
        ):
            yield CountedDocument(
                source_index=source_index,
                text=text,
                token_count=len(input_ids),
                metadata=metadata,
            )


def write_heldout_then_train(
    records: Iterator[CountedDocument],
    *,
    source_name: str,
    train_f,
    heldout_f,
    train_token_target: int,
    heldout_token_target: int,
    initial_counts: dict[str, dict[str, int]] | None = None,
    progress_callback=None,
) -> dict[str, dict[str, int]]:
    """Write heldout first, then train, until the train token target is reached."""
    counts = (
        {
            split: {"docs": values["docs"], "tokens": values["tokens"]}
            for split, values in initial_counts.items()
        }
        if initial_counts is not None
        else {
            "heldout": {"docs": 0, "tokens": 0},
            "train": {"docs": 0, "tokens": 0},
        }
    )
    current_split = (
        "heldout"
        if counts["heldout"]["tokens"] < heldout_token_target
        else "train"
    )
    if heldout_token_target <= 0:
        current_split = "train"

    try:
        for record in records:
            split = current_split
            handle = heldout_f if split == "heldout" else train_f

            write_jsonl_document(
                handle,
                source=source_name,
                split=split,
                source_index=record.source_index,
                text=record.text,
                token_count=record.token_count,
                metadata=record.metadata,
            )
            counts[split]["docs"] += 1
            counts[split]["tokens"] += record.token_count
            if progress_callback is not None:
                progress_callback(counts, record)

            if (
                split == "heldout"
                and counts["heldout"]["tokens"] >= heldout_token_target
            ):
                current_split = "train"
            if split == "train" and counts["train"]["tokens"] >= train_token_target:
                break
    finally:
        close = getattr(records, "close", None)
        if close is not None:
            close()

    return counts


def scan_jsonl_counts(path: Path) -> tuple[dict[str, int], int]:
    """Return document/token counts and truncate a trailing partial JSONL record."""
    counts = {"docs": 0, "tokens": 0}
    max_source_index = -1
    valid_end = 0

    if not path.exists():
        return counts, max_source_index

    with path.open("rb") as handle:
        while True:
            line_start = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("truncating partial JSONL record in %s", path)
                break
            counts["docs"] += 1
            counts["tokens"] += int(record["token_count"])
            max_source_index = max(max_source_index, int(record["source_index"]))
            valid_end = handle.tell()
            if valid_end <= line_start:
                break

    if path.stat().st_size != valid_end:
        with path.open("ab") as handle:
            handle.truncate(valid_end)

    return counts, max_source_index


def existing_raw_state(
    train_path: Path,
    heldout_path: Path,
) -> tuple[dict[str, dict[str, int]], int]:
    """Inspect existing raw files so an interrupted fetch can continue."""
    train_counts, train_max_index = scan_jsonl_counts(train_path)
    heldout_counts, heldout_max_index = scan_jsonl_counts(heldout_path)
    counts = {"heldout": heldout_counts, "train": train_counts}
    next_source_index = max(train_max_index, heldout_max_index) + 1
    return counts, next_source_index


def fetch_source(
    *,
    source: CorpusSource,
    out_dir: Path,
    tokenizer_name: str,
    train_token_target: int,
    heldout_token_target: int,
    max_docs: int | None,
    batch_size: int,
    resume: bool = True,
) -> dict[str, Any]:
    """Fetch and split a text corpus into train and heldout JSONL files.

    The source dataset is streamed from Hugging Face, tokenized in batches for
    counting, and written as JSONL records. Heldout examples are written first
    until the heldout token target is reached; all following examples are
    written to train until the train token target or ``max_docs`` limit is
    reached.

    Parameters
    ----------
    source : CorpusSource
        Dataset descriptor, including Hugging Face dataset/config/split and
        the column that contains the text.
    out_dir : Path
        Root output directory. Files for this source are written under
        ``out_dir / source.name``.
    tokenizer_name : str
        Hugging Face tokenizer name or path used only to count tokens.
    train_token_target : int
        Minimum number of train tokens to collect before stopping.
    heldout_token_target : int
        Minimum number of heldout tokens to collect before switching to train.
        If zero, examples are written directly to train.
    max_docs : int | None
        Optional cap on the number of source documents consumed.
    batch_size : int
        Number of streamed examples to tokenize at once.

    Returns
    -------
    dict[str, Any]
        Manifest-style metadata containing source settings, output paths,
        token/document counts, and the manifest path.

    Raises
    ------
    ValueError
        If token targets, ``max_docs``, or ``batch_size`` are invalid.

    Notes
    -----
    Token targets are lower bounds: the final document that crosses a target is
    included, so the resulting token counts may exceed the requested values.
    """
    validate_fetch_args(
        train_token_target=train_token_target,
        heldout_token_target=heldout_token_target,
        max_docs=max_docs,
        batch_size=batch_size,
    )

    # Keep all raw corpus artifacts for one logical source in a stable folder.
    source_dir = out_dir / source.name
    raw_dir = source_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    train_path = raw_dir / "train.jsonl"
    heldout_path = raw_dir / "heldout.jsonl"
    manifest_path = source_dir / "manifest.json"

    initial_counts = {
        "heldout": {"docs": 0, "tokens": 0},
        "train": {"docs": 0, "tokens": 0},
    }
    start_index = 0
    file_mode = "w"
    if resume:
        initial_counts, start_index = existing_raw_state(train_path, heldout_path)
        file_mode = "a"
        if start_index > 0:
            logger.info(
                "[%s] resuming from source_index=%s with train=%s tokens and heldout=%s tokens",
                source.name,
                f"{start_index:,}",
                f"{initial_counts['train']['tokens']:,}",
                f"{initial_counts['heldout']['tokens']:,}",
            )

    if (
        initial_counts["heldout"]["tokens"] >= heldout_token_target
        and initial_counts["train"]["tokens"] >= train_token_target
    ):
        result = {
            "source": asdict(source),
            "tokenizer": tokenizer_name,
            "paths": {
                "train": str(train_path),
                "heldout": str(heldout_path),
            },
            "targets": {
                "train_tokens": train_token_target,
                "heldout_tokens": heldout_token_target,
            },
            "counts": initial_counts,
            "batch_size": batch_size,
            "format": "jsonl",
            "status": "complete",
            "resume": resume,
            "next_source_index": start_index,
            "updated_at": time.time(),
        }
        manifest_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        result["manifest_path"] = str(manifest_path)
        return result

    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    tokenizer.model_max_length = 10**9

    stream = load_dataset(
        source.dataset,
        source.config,
        split=source.split,
        streaming=True,
    )
    if start_index > 0:
        stream = stream.skip(start_index)

    counted_docs = count_tokens_for_streamed_docs(
        stream,
        source=source,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_docs=max_docs,
        start_index=start_index,
    )

    last_progress_time = 0.0

    def write_progress(
        counts: dict[str, dict[str, int]], record: CountedDocument
    ) -> None:
        nonlocal last_progress_time
        now = time.monotonic()
        if now - last_progress_time < 30:
            return
        last_progress_time = now
        progress = {
            "source": asdict(source),
            "tokenizer": tokenizer_name,
            "paths": {
                "train": str(train_path),
                "heldout": str(heldout_path),
            },
            "targets": {
                "train_tokens": train_token_target,
                "heldout_tokens": heldout_token_target,
            },
            "counts": counts,
            "batch_size": batch_size,
            "format": "jsonl",
            "status": "partial",
            "resume": resume,
            "next_source_index": record.source_index + 1,
            "updated_at": time.time(),
        }
        manifest_path.write_text(
            json.dumps(progress, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    with train_path.open(file_mode, encoding="utf-8") as train_f, heldout_path.open(
        file_mode, encoding="utf-8"
    ) as heldout_f:
        counts = write_heldout_then_train(
            counted_docs,
            source_name=source.name,
            train_f=train_f,
            heldout_f=heldout_f,
            train_token_target=train_token_target,
            heldout_token_target=heldout_token_target,
            initial_counts=initial_counts,
            progress_callback=write_progress,
        )

    # Persist enough provenance to reproduce or inspect this corpus shard later.
    result = {
        "source": asdict(source),
        "tokenizer": tokenizer_name,
        "paths": {
            "train": str(train_path),
            "heldout": str(heldout_path),
        },
        "targets": {
            "train_tokens": train_token_target,
            "heldout_tokens": heldout_token_target,
        },
        "counts": counts,
        "batch_size": batch_size,
        "format": "jsonl",
        "status": "complete",
        "resume": resume,
        "next_source_index": counts["heldout"]["docs"] + counts["train"]["docs"],
        "updated_at": time.time(),
    }
    manifest_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    result["manifest_path"] = str(manifest_path)
    return result
