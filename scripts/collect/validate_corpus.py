import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vasae.data.corpus import DEFAULT_CORPUS_SOURCES, default_corpus_out_dir


@dataclass(frozen=True)
class SplitStats:
    docs: int
    tokens: int
    min_source_index: int | None
    max_source_index: int | None
    first_source_index: int | None
    last_source_index: int | None
    first_non_monotonic: tuple[int, int, int] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate fetched JSONL corpora and manifests."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Corpus root directory. Defaults to $VASAE_OUT/Dataset/data.",
    )
    parser.add_argument(
        "--corpora",
        nargs="+",
        choices=sorted(DEFAULT_CORPUS_SOURCES),
        default=sorted(DEFAULT_CORPUS_SOURCES),
        help="Corpus presets to validate.",
    )
    parser.add_argument(
        "--train-tokens",
        type=int,
        default=None,
        help="Expected minimum train tokens per corpus. Defaults to manifest target.",
    )
    parser.add_argument(
        "--heldout-tokens",
        type=int,
        default=None,
        help="Expected minimum heldout tokens per corpus. Defaults to manifest target.",
    )
    parser.add_argument(
        "--total-train-tokens",
        type=int,
        default=200_000_000,
        help="Expected minimum train tokens across all selected corpora.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer name for spot-checking stored token_count fields.",
    )
    parser.add_argument(
        "--spot-check-docs",
        type=int,
        default=0,
        help="Number of documents per split to retokenize when --tokenizer is set.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL record") from exc
            yield line_number, record


def scan_split(path: Path, *, expected_source: str, expected_split: str) -> SplitStats:
    docs = 0
    tokens = 0
    min_source_index = None
    max_source_index = None
    first_source_index = None
    last_source_index = None
    previous_source_index = None
    first_non_monotonic = None

    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise ValueError(f"{path} is empty")

    required_keys = {"source", "split", "source_index", "token_count", "text"}
    for line_number, record in iter_jsonl(path):
        missing = required_keys - record.keys()
        if missing:
            raise ValueError(
                f"{path}:{line_number}: missing required keys {sorted(missing)}"
            )
        if record["source"] != expected_source:
            raise ValueError(
                f"{path}:{line_number}: source={record['source']!r}, "
                f"expected {expected_source!r}"
            )
        if record["split"] != expected_split:
            raise ValueError(
                f"{path}:{line_number}: split={record['split']!r}, "
                f"expected {expected_split!r}"
            )
        if not isinstance(record["text"], str) or record["text"] == "":
            raise ValueError(f"{path}:{line_number}: text must be a non-empty string")

        source_index = int(record["source_index"])
        token_count = int(record["token_count"])
        if source_index < 0:
            raise ValueError(f"{path}:{line_number}: source_index must be non-negative")
        if token_count <= 0:
            raise ValueError(f"{path}:{line_number}: token_count must be positive")

        if first_source_index is None:
            first_source_index = source_index
        if (
            first_non_monotonic is None
            and previous_source_index is not None
            and source_index <= previous_source_index
        ):
            first_non_monotonic = (line_number, previous_source_index, source_index)
        previous_source_index = source_index
        last_source_index = source_index
        min_source_index = (
            source_index
            if min_source_index is None
            else min(min_source_index, source_index)
        )
        max_source_index = (
            source_index
            if max_source_index is None
            else max(max_source_index, source_index)
        )
        docs += 1
        tokens += token_count

    return SplitStats(
        docs=docs,
        tokens=tokens,
        min_source_index=min_source_index,
        max_source_index=max_source_index,
        first_source_index=first_source_index,
        last_source_index=last_source_index,
        first_non_monotonic=first_non_monotonic,
    )


def assert_manifest_matches_split(
    *,
    manifest_path: Path,
    split: str,
    manifest: dict[str, Any],
    stats: SplitStats,
) -> None:
    expected = manifest["counts"][split]
    if stats.docs != int(expected["docs"]) or stats.tokens != int(expected["tokens"]):
        raise ValueError(
            f"{manifest_path}: {split} manifest mismatch: "
            f"manifest docs/tokens={expected['docs']}/{expected['tokens']}, "
            f"raw docs/tokens={stats.docs}/{stats.tokens}"
        )


def assert_contiguous_source_indices(
    *,
    corpus: str,
    heldout: SplitStats,
    train: SplitStats,
    expected_next_source_index: int,
) -> None:
    if heldout.first_non_monotonic is not None:
        line_number, previous_source_index, source_index = heldout.first_non_monotonic
        raise ValueError(
            f"{corpus}: heldout source_index is not strictly increasing at line "
            f"{line_number}: {previous_source_index} -> {source_index}"
        )
    if train.first_non_monotonic is not None:
        line_number, previous_source_index, source_index = train.first_non_monotonic
        raise ValueError(
            f"{corpus}: train source_index is not strictly increasing at line "
            f"{line_number}: {previous_source_index} -> {source_index}"
        )
    if heldout.first_source_index != 0:
        raise ValueError(
            f"{corpus}: heldout starts at source_index={heldout.first_source_index}, "
            "expected 0"
        )
    if (
        heldout.max_source_index is None
        or heldout.last_source_index is None
        or train.min_source_index is None
        or train.first_source_index is None
    ):
        raise ValueError(f"{corpus}: cannot validate source index continuity")
    if train.min_source_index <= heldout.max_source_index:
        raise ValueError(
            f"{corpus}: heldout/train split boundary is interleaved: "
            f"train min source_index={train.min_source_index}, "
            f"heldout max source_index={heldout.max_source_index}"
        )
    if train.first_source_index != heldout.max_source_index + 1:
        raise ValueError(
            f"{corpus}: train first source_index={train.first_source_index}, "
            f"expected {heldout.max_source_index + 1}"
        )
    if train.last_source_index != expected_next_source_index - 1:
        raise ValueError(
            f"{corpus}: last train source_index={train.last_source_index}, "
            f"expected {expected_next_source_index - 1}"
        )


def spot_check_token_counts(
    *,
    tokenizer_name: str,
    corpus: str,
    split: str,
    path: Path,
    limit: int,
) -> None:
    if limit <= 0:
        return

    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    tokenizer.model_max_length = 10**9

    for checked, (line_number, record) in enumerate(iter_jsonl(path), start=1):
        expected = int(record["token_count"])
        actual = len(tokenizer(record["text"], add_special_tokens=False)["input_ids"])
        if actual != expected:
            raise ValueError(
                f"{path}:{line_number}: retokenized {corpus}/{split} token_count "
                f"mismatch: stored={expected}, actual={actual}"
            )
        if checked >= limit:
            return


def validate_corpus(
    *,
    corpus: str,
    out_dir: Path,
    train_tokens: int | None,
    heldout_tokens: int | None,
    tokenizer: str | None,
    spot_check_docs: int,
) -> tuple[int, int]:
    source = DEFAULT_CORPUS_SOURCES[corpus]
    source_dir = out_dir / corpus
    manifest_path = source_dir / "manifest.json"
    manifest = load_json(manifest_path)

    if manifest.get("status") != "complete":
        raise ValueError(f"{manifest_path}: status={manifest.get('status')!r}")
    if manifest["source"]["name"] != source.name:
        raise ValueError(f"{manifest_path}: source name does not match preset {corpus}")
    if manifest["source"]["dataset"] != source.dataset:
        raise ValueError(f"{manifest_path}: source dataset does not match preset")
    if manifest.get("format") != "jsonl":
        raise ValueError(f"{manifest_path}: format={manifest.get('format')!r}")

    train_path = Path(manifest["paths"]["train"])
    heldout_path = Path(manifest["paths"]["heldout"])
    train = scan_split(train_path, expected_source=corpus, expected_split="train")
    heldout = scan_split(
        heldout_path, expected_source=corpus, expected_split="heldout"
    )

    assert_manifest_matches_split(
        manifest_path=manifest_path, split="train", manifest=manifest, stats=train
    )
    assert_manifest_matches_split(
        manifest_path=manifest_path, split="heldout", manifest=manifest, stats=heldout
    )
    assert_contiguous_source_indices(
        corpus=corpus,
        heldout=heldout,
        train=train,
        expected_next_source_index=int(manifest["next_source_index"]),
    )

    min_train_tokens = (
        int(manifest["targets"]["train_tokens"])
        if train_tokens is None
        else train_tokens
    )
    min_heldout_tokens = (
        int(manifest["targets"]["heldout_tokens"])
        if heldout_tokens is None
        else heldout_tokens
    )
    if train.tokens < min_train_tokens:
        raise ValueError(
            f"{corpus}: train has {train.tokens:,} tokens, "
            f"expected at least {min_train_tokens:,}"
        )
    if heldout.tokens < min_heldout_tokens:
        raise ValueError(
            f"{corpus}: heldout has {heldout.tokens:,} tokens, "
            f"expected at least {min_heldout_tokens:,}"
        )

    if tokenizer is not None:
        spot_check_token_counts(
            tokenizer_name=tokenizer,
            corpus=corpus,
            split="heldout",
            path=heldout_path,
            limit=spot_check_docs,
        )
        spot_check_token_counts(
            tokenizer_name=tokenizer,
            corpus=corpus,
            split="train",
            path=train_path,
            limit=spot_check_docs,
        )

    print(
        f"[OK] {corpus}: train={train.tokens:,} tokens/{train.docs:,} docs; "
        f"heldout={heldout.tokens:,} tokens/{heldout.docs:,} docs"
    )
    return train.tokens, heldout.tokens


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or default_corpus_out_dir()

    total_train_tokens = 0
    total_heldout_tokens = 0
    failures = []
    for corpus in args.corpora:
        try:
            train_tokens, heldout_tokens = validate_corpus(
                corpus=corpus,
                out_dir=out_dir,
                train_tokens=args.train_tokens,
                heldout_tokens=args.heldout_tokens,
                tokenizer=args.tokenizer,
                spot_check_docs=args.spot_check_docs,
            )
        except Exception as exc:
            failures.append(f"{corpus}: {exc}")
            print(f"[FAIL] {corpus}: {exc}")
            continue
        total_train_tokens += train_tokens
        total_heldout_tokens += heldout_tokens

    if not failures and total_train_tokens < args.total_train_tokens:
        failures.append(
            f"mixture: total train tokens={total_train_tokens:,}, "
            f"expected at least {args.total_train_tokens:,}"
        )
        print(f"[FAIL] {failures[-1]}")

    if failures:
        print(f"[FAIL] validation failed for {len(failures)} check(s)")
        raise SystemExit(1)

    print(
        f"[OK] mixture: train={total_train_tokens:,} tokens; "
        f"heldout={total_heldout_tokens:,} tokens"
    )


if __name__ == "__main__":
    main()
