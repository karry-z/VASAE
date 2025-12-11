#!/usr/bin/env python
import argparse

import torch
from accelerate import Accelerator
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute GPT-2 perplexity on Geralt-Targaryen/openwebtext2 "
        "using HF's strided sliding-window method (streamed)."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="gpt2",
        help="Model id (e.g., gpt2, openai-community/gpt2-large)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Geralt-Targaryen/openwebtext2",
        help="Dataset repo id",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train[:1%]",
        help="Which split or slice to use",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Stride size for sliding window",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="Text column name, auto-detected if None",
    )
    return parser.parse_args()


def compute_doc_stats(model, tokenizer, text, stride, device):
    """compute perplexity for one document (no global concatenation)
    https://huggingface.co/transformers/perplexity.html"""
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids  # (1, seq_len)

    seq_len = input_ids.size(1)
    max_length = model.config.n_positions

    nll_sum = 0.0
    n_tokens = 0
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end

        ids_slice = input_ids[:, begin:end].to(device)
        target_ids = ids_slice.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            out = model(ids_slice, labels=target_ids)
            loss = out.loss

        valid = (target_ids != -100).sum().item()
        nll_sum += loss.item() * (valid - 1)
        n_tokens += valid - 1

        prev_end = end
        if end == seq_len:
            break

    return nll_sum, n_tokens


def main():
    args = parse_args()
    logger = get_logger(__file__)

    accelerator = Accelerator()
    device = accelerator.device
    logger.info(f"device = {device}")

    # Load model + tokenizer
    logger.info(f"loading model: {args.model_id}")
    model = GPT2LMHeadModel.from_pretrained(args.model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_id)
    model.eval()
    logger.info(f"model loaded")

    # Load dataset (not concatenated)
    logger.info(f"loading dataset: {args.dataset_name} [{args.split}]")
    dataset = load_dataset(args.dataset_name, split=args.split)
    logger.info(f"dataset loaded with {len(dataset)} samples")

    # Detect text column
    if args.text_column:
        text_col = args.text_column
    else:
        text_col = "text" if "text" in dataset.column_names else dataset.column_names[0]
    logger.info(f"using text column: {text_col}")

    total_nll = 0.0
    total_tokens = 0

    logger.info("begin computing perplexity...")

    for idx, row in enumerate(tqdm(dataset, desc="processing docs")):
        text = row[text_col]
        if not text or not text.strip():
            continue

        nll, ntoks = compute_doc_stats(model, tokenizer, text, args.stride, device)
        total_nll += nll
        total_tokens += ntoks

        if accelerator.is_local_main_process and idx % 1000 == 0:
            logger.info(f"processed {idx} documents...")

    avg_nll = total_nll / total_tokens
    ppl = torch.exp(torch.tensor(avg_nll))

    logger.info(f"final average NLL/token = {avg_nll:.6f}")
    logger.info(f"final perplexity      = {ppl.item():.6f}")


if __name__ == "__main__":
    main()
