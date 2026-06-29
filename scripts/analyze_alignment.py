"""Assign geometric nearest-token labels to SAE decoder features for paper analysis."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from vasae.analysis import nearest_token_alignment
from vasae.models import SAEConfig, SAEModel


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(asctime)s][%(name)s] %(message)s",
    datefmt="%Y%m%d %H:%M:%S",
)
LOGGER = logging.getLogger("analyze_alignment")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute nearest-token geometric labels for SAE decoder features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt or a run directory containing it.")
    parser.add_argument("--model-name", default="gpt2", help="Hugging Face causal LM that supplies vocabulary embeddings.")
    parser.add_argument("--top-k", type=int, default=5, help="Nearest token labels per feature.")
    parser.add_argument("--chunk-size", type=int, default=2048, help="Feature chunk size for similarity search.")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or a torch device string.")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default=None, help="Vocabulary embedding dtype.")
    return parser


def prepare_runtime(args) -> tuple[torch.device, torch.dtype | None]:
    transformers.logging.set_verbosity_error()
    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    dtype = {
        None: None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    return torch.device(device_name), dtype


def checkpoint_file(path: str) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def load_decoder_and_vocab(checkpoint_path: Path, model_name: str, device: torch.device, dtype):
    checkpoint_payload = torch.load(checkpoint_path, map_location=device)
    sae_model = SAEModel(SAEConfig(**checkpoint_payload["sae_config"])).to(device).float()
    sae_model.load_state_dict(checkpoint_payload["model_state_dict"])
    sae_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    lm.to(device)
    lm.eval()

    feature_directions = sae_model.decoder.weight.detach().T.contiguous()
    vocab_embeddings = lm.get_input_embeddings().weight.detach()
    if feature_directions.size(1) != vocab_embeddings.size(1):
        raise ValueError(
            f"Decoder feature dim {feature_directions.size(1)} does not match "
            f"vocabulary embedding dim {vocab_embeddings.size(1)}."
        )
    return tokenizer, feature_directions, vocab_embeddings


def alignment_bucket(score: float) -> str:
    if score >= 0.8:
        return "strong"
    if score >= 0.5:
        return "medium"
    if score >= 0.3:
        return "weak"
    return "none"


def build_alignment_results(args, tokenizer, feature_directions, vocab_embeddings):
    LOGGER.info("Computing nearest-token geometric labels for %d features", feature_directions.size(0))
    token_ids, scores = nearest_token_alignment(
        feature_directions=feature_directions,
        vocab_embeddings=vocab_embeddings,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
    )

    bucket_counts = {"strong": 0, "medium": 0, "weak": 0, "none": 0}
    max_scores: list[float] = []
    features = []
    for feature_idx, (ids_row, scores_row) in enumerate(zip(token_ids.cpu().tolist(), scores.cpu().tolist())):
        max_score = float(scores_row[0])
        bucket = alignment_bucket(max_score)
        bucket_counts[bucket] += 1
        max_scores.append(max_score)
        features.append(
            {
                "feature": feature_idx,
                "alignment_bucket": bucket,
                "max_cosine": max_score,
                "nearest_tokens": [
                    {
                        "token_id": int(token_id),
                        "token_label": tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False),
                        "cosine": float(score),
                    }
                    for token_id, score in zip(ids_row, scores_row)
                ],
            }
        )

    summary = {"n_features": 0, **bucket_counts}
    if max_scores:
        summary = {
            "n_features": len(max_scores),
            "max_cosine_mean": sum(max_scores) / len(max_scores),
            "max_cosine_min": min(max_scores),
            "max_cosine_max": max(max_scores),
            **bucket_counts,
        }

    return {"summary": summary, "features": features}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

    device, dtype = prepare_runtime(args)
    checkpoint_path = checkpoint_file(args.checkpoint)
    tokenizer, feature_directions, vocab_embeddings = load_decoder_and_vocab(
        checkpoint_path, args.model_name, device, dtype
    )
    results = build_alignment_results(
        args, tokenizer, feature_directions, vocab_embeddings
    )
    output_dir = checkpoint_path.parent / "alignment"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    LOGGER.info("Saved geometric nearest-token alignment results to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
