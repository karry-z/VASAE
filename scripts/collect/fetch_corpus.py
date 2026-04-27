import argparse
import logging
import os
from pathlib import Path

from vasae.data.corpus import (
    DEFAULT_CORPUS_SOURCES,
    default_corpus_out_dir,
    fetch_source,
)
from vasae.utils.log import get_logger


logger = get_logger("fetch_corpus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch a configured corpus shard.")
    parser.add_argument(
        "corpus",
        choices=sorted(DEFAULT_CORPUS_SOURCES),
        help="Corpus preset to fetch.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Root output directory. The dataset is written under <out-dir>/<source>/.",
    )
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--train-tokens", type=int, default=200_000_000)
    parser.add_argument("--heldout-tokens", type=int, default=1_000_000)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Overwrite existing raw files instead of appending from the current offset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or default_corpus_out_dir()
    source = DEFAULT_CORPUS_SOURCES[args.corpus]

    logger.info(
        "[%s] streaming %s/%s:%s until heldout=%s and train=%s tokens",
        source.name,
        source.dataset,
        source.config,
        source.split,
        f"{args.heldout_tokens:,}",
        f"{args.train_tokens:,}",
    )
    result = fetch_source(
        source=source,
        out_dir=out_dir,
        tokenizer_name=args.tokenizer,
        train_token_target=args.train_tokens,
        heldout_token_target=args.heldout_tokens,
        max_docs=args.max_docs,
        batch_size=args.batch_size,
        resume=not args.no_resume,
    )
    logger.info(
        "[%s] wrote train=%s tokens, heldout=%s tokens; manifest=%s",
        source.name,
        f"{result['counts']['train']['tokens']:,}",
        f"{result['counts']['heldout']['tokens']:,}",
        result["manifest_path"],
    )


if __name__ == "__main__":
    main()
    logging.shutdown()
    os._exit(0)
