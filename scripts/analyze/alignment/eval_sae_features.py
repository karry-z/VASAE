"""Dump per-(layer, prompt, position, feature) firing records to Parquet.

For each trained VASAE checkpoint in the given set (default: all discovered
layers), runs the language model + SAE on a text test set and writes every
z > 0 firing event with the fields needed for downstream alignment analysis.

Output layout
-------------
{output-dir}/
  meta.json
  prompts.parquet          prompt_id, text, n_tokens
  tokens.parquet           prompt_id, position, token_id, token_string
  L{i}/
    features.parquet       feature_id, feature_token_id, feature_token_string, alignment
    records.parquet        prompt_id, position, feature_id, z, current_token_id
"""

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from nnsight import NNsight

from shared_utils.log import get_logger
from vasae.analysis.alignment import compute_geometric_alignment
from vasae.analysis.sae_loader import get_decoder_features, load_sae_for_analysis
from vasae.engine.intervention import get_layer_proxy
from vasae.models.factory import get_embedding, load_model
from vasae.models.sae import SAEModel

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--sae-dir",
        type=str,
        help="Root dir with 001F_{model_tag}_L{i}_{variant} subdirs, use without specifying --sae-paths; use with --layers if we only need part of layers",
    )
    grp.add_argument(
        "--sae-paths",
        type=str,
        nargs="+",
        help="Explicit layer:path pairs, e.g. 0:/path/L0 6:/path/L6, use when sae_dir is not like regular format. e.g., the folder path that contain safetensor.",
    )
    p.add_argument("--model-name", type=str, required=True)
    p.add_argument("--variant", type=str, default="soft")
    p.add_argument(
        "--layers",
        type=str,
        default=None,
        help='Subset of layers to dump, e.g. "0-11" or "0,6,10". '
        "Default: all discovered.",
    )
    p.add_argument("--dataset", type=str, default="wikitext")
    p.add_argument("--dataset-config", type=str, default="wikitext-103-raw-v1")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--n-samples", type=int, default=5000)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def discover_checkpoints(
    sae_dir: str, model_name: str, variant: str
) -> dict[int, Path]:
    r"""
    Checkpoint discovery (duplicated from analyze_alignment_quality.py)

    Args
    ---
    sae_dir : str
        directory with sae checkpoints. Root dir with 001F\_{model_name}\_L{i}\_{variant} subdirs
    model_name : str
        language model name the sae trained on, such as 'gpt2'.
    variant : str
        variant for sae. possible variants: 'plain', 'hard', 'soft'.

    Return
    ---
    dict[int, Path]
        checkpoints dict with layer id as key and checkpoint path as value.
    """
    sae_dir = Path(sae_dir)
    model_tag = "gpt2" if "gpt2" in model_name else "llama"
    pattern = re.compile(rf"001F_{model_tag}_L(\d+)_{variant}$")
    checkpoints: dict[int, Path] = {}
    for d in sorted(sae_dir.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if m:
            checkpoints[int(m.group(1))] = d
    return checkpoints


def parse_sae_paths(entries: list[str]) -> dict[int, Path]:
    """
    我们只分析没有按照001F格式存储的 layers 时使用
    """
    out: dict[int, Path] = {}
    for e in entries:
        layer_str, path_str = e.split(":", 1)
        out[int(layer_str)] = Path(path_str)
    return out


def parse_layers_spec(spec: str | None, available: list[int]) -> list[int]:
    """
    这是为了用 sae_dir 的时候会加载所有 checkpoint，但仅需要留部分层使用
    """
    if spec is None:
        return sorted(available)
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        req = set(range(int(lo), int(hi) + 1))
    else:
        req = {int(x) for x in spec.split(",") if x.strip()}
    missing = req - set(available)
    if missing:
        raise ValueError(
            f"Requested layers not available: {sorted(missing)}. "
            f"Available: {sorted(available)}"
        )
    return sorted(req)


def select_prompts(ds, n_samples: int) -> list[str]:
    """Take the first ``n_samples`` non-empty texts from the dataset."""
    selected: list[str] = []
    for item in ds:
        text = item["text"] if isinstance(item, dict) else item
        if isinstance(text, str) and text.strip():
            selected.append(text)
            if len(selected) >= n_samples:
                break
    return selected


def build_prompt_tables(
    prompts: list[str],
    tokenizer,
    max_length: int,
):
    """Tokenize prompts and build prompt/token metadata tables.

    Parameters
    ----------
    prompts : list[str]
        Input text prompts to tokenize.
    tokenizer : PreTrainedTokenizerBase
        Tokenizer used to encode prompts and decode token ids.
    max_length : int
        Maximum sequence length for truncation.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, pandas.DataFrame, pandas.DataFrame]
        A 4-tuple containing:
        - input_ids: (N, S_max) int tensor on CPU.
        - valid_token_mask: (N, S_max) int tensor on CPU, where 1 means
          non-pad and non-special token.
        - prompts_df: columns [prompt_id, text, n_tokens].
        - tokens_df: columns [prompt_id, position, token_id, token_string].
    """
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        return_special_tokens_mask=True,
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"]
    pad_mask = enc["attention_mask"].bool()
    special_tokens_mask = enc.get("special_tokens_mask")
    if special_tokens_mask is None:
        special_tokens_mask = torch.zeros_like(pad_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    # Keep only non-pad and non-special tokens.
    valid_token_mask = (pad_mask & (~special_tokens_mask)).to(torch.int64)
    n_tokens = valid_token_mask.sum(dim=1).tolist()

    prompts_df = pd.DataFrame(
        {
            "prompt_id": np.arange(len(prompts), dtype=np.int32),
            "text": prompts,
            "n_tokens": np.asarray(n_tokens, dtype=np.int32),
        }
    )

    # Decode every unique token id once to avoid re-decoding in the hot loop.
    unique_ids = torch.unique(input_ids[valid_token_mask.bool()]).tolist()
    id2str = {
        tid: tokenizer.decode([tid]) for tid in unique_ids
    }  # token id to token string

    # Flatten valid (prompt_id, position) pairs.
    nonzero_idx = valid_token_mask.nonzero(
        as_tuple=False
    )  # (K, 2): [prompt_id, position]
    prompt_id = nonzero_idx[:, 0].numpy().astype(np.int32)
    position_id = nonzero_idx[:, 1].numpy().astype(np.int32)
    valid_input_ids = (
        input_ids[nonzero_idx[:, 0], nonzero_idx[:, 1]].numpy().astype(np.int32)
    )
    valid_token_str = [id2str[int(t)] for t in valid_input_ids]

    tokens_df = pd.DataFrame(
        {
            "prompt_id": prompt_id,
            "position": position_id,
            "token_id": valid_input_ids,
            "token_string": valid_token_str,
        }
    )

    return input_ids, valid_token_mask, prompts_df, tokens_df


def build_features_df(
    sae,
    W_E: torch.Tensor,
    tokenizer,
    device: torch.device,
) -> pd.DataFrame:
    """Build SAE feature alignment metadata dataframe. The SAE is for one LM layer.

    Parameters
    ----------
    sae : SAEModel
        Loaded SAE instance for a single transformer layer.
    W_E : torch.Tensor
        Language model token embedding matrix with shape (V, D), where V is
        vocabulary size and D is model hidden size.
    tokenizer : PreTrainedTokenizerBase
        Tokenizer used to decode aligned token ids into strings.
    device : torch.device
        Device used for geometric alignment computation.

    Returns
    -------
    pandas.DataFrame
        Per-feature table with columns:
        - feature_id : int32, contiguous feature index in [0, n_features)
        - feature_token_id : int32, top-1 aligned token id per feature
        - feature_token_string : str, decoded token text
        - alignment : float32, top-1 cosine similarity score
    """
    features = get_decoder_features(sae)
    geo = compute_geometric_alignment(features, W_E, top_k=1, device=device)
    n_features = int(geo.max_sims.shape[0])

    feature_token_ids = geo.topk_indices[:, 0].numpy().astype(np.int32)
    alignment = geo.max_sims.numpy().astype(np.float32)
    # Decode each unique token id once.
    id2str = {int(t): tokenizer.decode([int(t)]) for t in np.unique(feature_token_ids)}
    feature_token_strings = [id2str[int(t)] for t in feature_token_ids]

    return pd.DataFrame(
        {
            "feature_id": np.arange(n_features, dtype=np.int32),
            "feature_token_id": feature_token_ids,
            "feature_token_string": feature_token_strings,
            "alignment": alignment,
        }
    )


RECORDS_SCHEMA = pyarrow.schema(
    [
        ("prompt_id", pyarrow.int32()),
        ("position", pyarrow.int32()),
        ("feature_id", pyarrow.int32()),
        ("z", pyarrow.float32()),
        ("current_input_token_id", pyarrow.int32()),
    ]
)


@torch.no_grad()
def dump_all_layers(
    layers: list[int],
    saes: dict[int, SAEModel],
    writers: dict[int, pq.ParquetWriter],
    nn_model: NNsight,
    input_ids: torch.Tensor,
    token_mask: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[int, int]:
    """dump all layer sae results with writers.

    One LM forward per batch; each layer's SAE encoder and the z>0 selection
    run inside the nnsight trace, so only compact firing tensors (indices +
    values) are materialised on host memory — the dense activations and the
    dense z tensors never leave GPU.
    """
    n_prompts = input_ids.shape[0]
    totals = {li: 0 for li in layers}
    n_batches = (n_prompts + batch_size - 1) // batch_size

    for batch_idx, start in enumerate(range(0, n_prompts, batch_size)):
        # slice input data
        end = min(start + batch_size, n_prompts)
        input_ids_batch = input_ids[start:end].to(device)  # (B, S)
        token_mask_batch = token_mask[start:end].to(device)  # (B, S)

        saved: dict[int, dict] = {}
        with nn_model.trace(input_ids_batch):
            for layer_i in layers:
                h = get_layer_proxy(nn_model, layer_i).output
                _, z = saes[layer_i].encode(h)  # (B, S, F)
                z = (
                    z * token_mask_batch.unsqueeze(-1).float()
                )  # keep only non-pad and non-special tokens
                nonzero_ids = (
                    z > 0
                ).nonzero()  # (K, 3): K nonzeros and each coordinates with [B, S, F].
                batch_prompt_ids = nonzero_ids[:, 0]  # (K,)
                position_ids = nonzero_ids[:, 1]  # (K,)
                feature_ids = nonzero_ids[:, 2]  # (K,)
                saved[layer_i] = {
                    "batch_prompt_ids": batch_prompt_ids.to(torch.int32).save(),
                    "position_ids": position_ids.to(torch.int32).save(),
                    "feature_ids": feature_ids.to(torch.int32).save(),
                    "z": z[batch_prompt_ids, position_ids, feature_ids]
                    .to(torch.float32)
                    .save(),
                    "current_input_token_ids": input_ids_batch[
                        batch_prompt_ids, position_ids
                    ]
                    .to(torch.int32)
                    .save(),
                }

        batch_counts = {}
        for layer_i in layers:
            record_batch_dict = saved[layer_i]
            batch_prompt_ids = record_batch_dict["batch_prompt_ids"]
            if batch_prompt_ids.numel() == 0:
                # no activative z in this batch
                batch_counts[layer_i] = 0
                continue
            prompt_ids = (start + batch_prompt_ids).to(torch.int32).cpu().numpy()
            position_ids = record_batch_dict["position_ids"].cpu().numpy()
            record_batch_pyarrow = pyarrow.RecordBatch.from_arrays(
                [
                    pyarrow.array(prompt_ids, type=pyarrow.int32()),
                    pyarrow.array(position_ids, type=pyarrow.int32()),
                    pyarrow.array(
                        record_batch_dict["feature_ids"].cpu().numpy(),
                        type=pyarrow.int32(),
                    ),
                    pyarrow.array(
                        record_batch_dict["z"].cpu().numpy(), type=pyarrow.float32()
                    ),
                    pyarrow.array(
                        record_batch_dict["current_input_token_ids"].cpu().numpy(),
                        type=pyarrow.int32(),
                    ),
                ],
                schema=RECORDS_SCHEMA,
            )  # 增量写单文件用了 pyarrow
            writers[layer_i].write_batch(record_batch_pyarrow)
            totals[layer_i] += record_batch_pyarrow.num_rows
            batch_counts[layer_i] = record_batch_pyarrow.num_rows

        if batch_idx % 50 == 0:
            logger.info(
                f"batch {batch_idx}/{n_batches}: firings {batch_counts} (totals {totals})"
            )

        del saved
        if device.type == "cuda" and batch_idx % 50 == 0:
            torch.cuda.empty_cache()

    return totals


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Discover checkpoints ---
    if args.sae_paths:
        checkpoints = parse_sae_paths(args.sae_paths)
    else:
        checkpoints = discover_checkpoints(args.sae_dir, args.model_name, args.variant)
    if not checkpoints:
        raise SystemExit("No SAE checkpoints found.")
    layers = parse_layers_spec(args.layers, list(checkpoints.keys()))
    checkpoints = {i: checkpoints[i] for i in layers}
    logger.info(f"Processing layers: {layers}")

    # --- Load LM ---
    logger.info(f"Loading LM {args.model_name}...")
    lm_model, tokenizer = load_model(args.model_name, device=str(device))
    nn_model = NNsight(lm_model)
    W_E = get_embedding(lm_model).weight.data
    logger.info(f"vocab_size={W_E.shape[0]} dim_model={W_E.shape[1]}")

    # --- Dataset + prompt/token tables ---
    logger.info(
        f"Loading dataset {args.dataset}/{args.dataset_config} split={args.split}"
    )
    ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
    prompts = select_prompts(ds, args.n_samples)
    logger.info(f"selected {len(prompts)} non-empty prompts")

    input_ids, token_mask, prompts_df, tokens_df = build_prompt_tables(
        prompts, tokenizer, args.max_length
    )
    n_prompts, max_seq_len = input_ids.shape
    logger.info(
        f"tokenized: {n_prompts} prompts, padded or truncated to seqlen={max_seq_len}"
    )

    prompts_df.to_parquet(out_dir / "prompts.parquet", index=False)
    tokens_df.to_parquet(out_dir / "tokens.parquet", index=False)
    logger.info(
        f"wrote prompts.parquet ({len(prompts_df)} rows) and "
        f"tokens.parquet ({len(tokens_df)} rows)"
    )

    # --- Load all SAEs + write features.parquet per layer ---
    saes: dict[int, object] = {}
    n_features_per: dict[int, int] = {}
    for layer_idx in layers:
        layer_dir = out_dir / f"L{layer_idx}"
        layer_dir.mkdir(exist_ok=True)
        logger.info(f"Layer {layer_idx}: loading SAE from {checkpoints[layer_idx]}")
        sae = SAEModel.from_pretrained(checkpoints[layer_idx]).eval().to(device)
        saes[layer_idx] = sae
        n_features_per[layer_idx] = int(sae.config.dim_sparse)

        features_df = build_features_df(sae, W_E, tokenizer, device)
        features_df.to_parquet(layer_dir / "features.parquet", index=False)
        logger.info(f"L{layer_idx} features.parquet written ({len(features_df)} rows)")

    # Open one records writer per layer
    writers: dict[int, pq.ParquetWriter] = {
        li: pq.ParquetWriter(
            str(out_dir / f"L{li}" / "records.parquet"),
            RECORDS_SCHEMA,
            compression="zstd",
        )
        for li in layers
    }

    # --- Single-pass dump across all layers ---
    n_batches = (n_prompts + args.batch_size - 1) // args.batch_size
    try:
        totals = dump_all_layers(
            layers=layers,
            saes=saes,
            writers=writers,
            nn_model=nn_model,
            input_ids=input_ids,
            token_mask=token_mask,
            batch_size=args.batch_size,
            device=device,
        )
    finally:
        for w in writers.values():
            w.close()

    per_layer_stats = {
        li: {
            "n_features": n_features_per[li],
            "total_records": totals[li],
        }
        for li in layers
    }
    logger.info(f"All layers done: totals={totals} in {n_batches} batches")

    # --- Meta ---
    meta = {
        "args": vars(args),
        "layers": layers,
        "n_prompts": int(n_prompts),
        "max_length_padded": int(max_seq_len),
        "vocab_size": int(W_E.shape[0]),
        "tokenizer_name": tokenizer.__class__.__name__,
        "n_batches": n_batches,
        "per_layer": per_layer_stats,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info("Wrote meta.json.")


if __name__ == "__main__":
    main()
