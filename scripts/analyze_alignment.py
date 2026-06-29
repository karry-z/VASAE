"""Assign geometric nearest-token labels to SAE decoder features for paper analysis."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


GEOMETRIC_LABEL_NOTE = (
    "Nearest-token labels are cosine-nearest vocabulary embedding labels for decoder "
    "directions. They are not semantic explanations or evidence of context-invariant behavior."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute nearest-token geometric labels and cosine similarities for SAE "
            "decoder features. These labels are vocabulary-embedding neighbors, not "
            "semantic explanations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt or a run directory containing it.")
    parser.add_argument("--model-name", default="gpt2", help="Hugging Face causal LM that supplies vocabulary embeddings.")
    parser.add_argument("--top-k", type=int, default=5, help="Nearest token labels per feature.")
    parser.add_argument("--chunk-size", type=int, default=2048, help="Feature chunk size for similarity search.")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or a torch device string.")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default=None, help="Vocabulary embedding dtype.")
    parser.add_argument("--output-dir", default=None, help="Directory for results.json.")
    return parser


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s][%(asctime)s][%(name)s] %(message)s",
        datefmt="%Y%m%d %H:%M:%S",
    )
    return logging.getLogger("analyze_alignment")


def checkpoint_file(path: Path) -> Path:
    if path.is_dir():
        path = path / "checkpoint.pt"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype_name: str | None):
    import torch

    return {
        None: None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def load_sae(checkpoint_path: Path, device):
    import torch

    from vasae.models import SAEConfig, SAEModel

    payload = torch.load(checkpoint_path, map_location=device)
    config = SAEConfig(**payload["sae_config"])
    model = SAEModel(config).to(device).float()
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def load_vocab(model_name: str, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return tokenizer, model.get_input_embeddings().weight.detach()


def token_label(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)


def alignment_bucket(score: float) -> str:
    if score >= 0.40:
        return "strong"
    if score >= 0.25:
        return "medium"
    return "weak"


def summarize(max_scores: list[float]) -> dict:
    if not max_scores:
        return {"n_features": 0, "strong": 0, "medium": 0, "weak": 0}
    buckets = {"strong": 0, "medium": 0, "weak": 0}
    for score in max_scores:
        buckets[alignment_bucket(score)] += 1
    return {
        "n_features": len(max_scores),
        "max_cosine_mean": sum(max_scores) / len(max_scores),
        "max_cosine_min": min(max_scores),
        "max_cosine_max": max(max_scores),
        **buckets,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = setup_logger()

    import transformers

    transformers.logging.set_verbosity_error()

    from vasae.analysis import nearest_token_alignment

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

    checkpoint_path = checkpoint_file(Path(args.checkpoint))
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "alignment"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    sae_model, checkpoint_payload = load_sae(checkpoint_path, device)
    tokenizer, vocab_embeddings = load_vocab(args.model_name, device, dtype)
    feature_directions = sae_model.decoder.weight.detach().T.contiguous()
    if feature_directions.size(1) != vocab_embeddings.size(1):
        raise ValueError(
            f"Decoder feature dim {feature_directions.size(1)} does not match "
            f"vocabulary embedding dim {vocab_embeddings.size(1)}."
        )

    logger.info("Computing nearest-token geometric labels for %d features", feature_directions.size(0))
    token_ids, scores = nearest_token_alignment(
        feature_directions=feature_directions,
        vocab_embeddings=vocab_embeddings,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
    )
    token_ids_cpu = token_ids.cpu()
    scores_cpu = scores.cpu()
    max_scores = scores_cpu[:, 0].tolist()

    features = []
    for feature_idx, (ids_row, scores_row) in enumerate(zip(token_ids_cpu.tolist(), scores_cpu.tolist())):
        nearest = [
            {
                "token_id": int(token_id),
                "token_label": token_label(tokenizer, int(token_id)),
                "cosine": float(score),
            }
            for token_id, score in zip(ids_row, scores_row)
        ]
        features.append(
            {
                "feature": feature_idx,
                "alignment_bucket": alignment_bucket(float(scores_row[0])),
                "max_cosine": float(scores_row[0]),
                "nearest_tokens": nearest,
            }
        )

    results = {
        "note": GEOMETRIC_LABEL_NOTE,
        "config": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_method": checkpoint_payload.get("method"),
            "model_name": args.model_name,
            "top_k": args.top_k,
            "chunk_size": args.chunk_size,
        },
        "summary": summarize(max_scores),
        "features": features,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    logger.info("Saved geometric nearest-token alignment results to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
