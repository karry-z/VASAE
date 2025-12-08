import os

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from utils import get_logger


class CFG:
    batch_size = 100
    device = "cuda"
    dataset = "openwebtext"  # "squad" or "openwebtext"
    max_length = 1024
    num_workers = 2
    out_path_template = "gpt2_{dataset_name}_ppl.txt"


def get_model(device: torch.device):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.config.pad_token_id = model.config.eos_token_id
    model.to(device)
    model.eval()

    return model, tokenizer


def get_dataset(dataset_name: str):
    if dataset_name == "squad":
        dataset = load_dataset("squad", split="validation")

        def build(e):
            ans = e["answers"]["text"][0] if e["answers"]["text"] else ""
            return {
                "text": (
                    "context: "
                    + e["context"].strip()
                    + "\nquestion: "
                    + e["question"].strip()
                    + "\nanswer: "
                    + ans.strip()
                )
            }

        dataset = dataset.map(build)
    elif dataset_name == "openwebtext":
        # large corpus, make sure you know what you're doing
        dataset = load_dataset("vietgpt/openwebtext_en", split="train")
        # assume dataset already has a "text" field
    else:
        raise ValueError(f"invalid dataset name: {dataset_name}")
    return dataset


def make_collate_fn(tokenizer, max_length: int):
    def collate_fn(batch):
        texts = [ex["text"] for ex in batch]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        return enc

    return collate_fn


def compute_batch_perplexity(model, batch, device: torch.device):
    """
    Compute per-sample perplexity for a batch.
    batch: dict with input_ids, attention_mask (on CPU)
    returns: 1D tensor of shape (batch_size,)
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)  # mask for padding

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, T, V)

    # Shift for language modeling: predict token t from tokens < t
    shift_logits = logits[
        ..., :-1, :
    ].contiguous()  # (B, T-1, V) no next token for the last one
    shift_labels = input_ids[
        ..., 1:
    ].contiguous()  # (B, T-1) no previous token to predict the first one
    shift_mask = attention_mask[..., 1:].contiguous()  # (B, T-1)

    # log softmax over vocab
    log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, T-1, V)

    # to avoid using invalid indices for padded positions, clamp labels but mask afterwards
    shift_labels_safe = shift_labels.clone()
    shift_labels_safe[shift_mask == 0] = 0

    # gather log-prob of true tokens, selects elements along a dimension using index lookup
    token_log_probs = log_probs.gather(
        dim=-1, index=shift_labels_safe.unsqueeze(-1)
    ).squeeze(
        -1
    )  # (B, T-1)

    # mask out pad tokens
    token_log_probs = token_log_probs * shift_mask

    # per-example negative log-likelihood (average over non-pad tokens)
    token_counts = shift_mask.sum(dim=-1)  # (B,)
    # avoid division by zero if any empty sequence
    token_counts = torch.clamp(token_counts, min=1)
    nll = -token_log_probs.sum(dim=-1) / token_counts  # (B,) Negative Log-Likelihood

    # perplexity = exp(nll)
    ppl = torch.exp(nll)  # (B,)
    return ppl.cpu()


def main():
    args = CFG()
    logger = get_logger(__file__)

    device = args.device
    logger.info(f"Using device: {device}")

    model, tokenizer = get_model(device)
    logger.info("GPT-2 loaded")

    dataset = get_dataset(args.dataset)
    logger.info(f"Dataset '{args.dataset}' loaded")
    logger.info(f"Total samples: {len(dataset)}")

    collate_fn = make_collate_fn(tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    os.makedirs("out", exist_ok=True)
    out_path = os.path.join(
        "out", args.out_path_template.format(dataset_name=args.dataset)
    )

    ppls = []
    logger.info("Starting evaluation")

    for batch_i, batch in enumerate(dataloader):
        batch_ppl = compute_batch_perplexity(model, batch, device)

        ppls.extend(batch_ppl)

        logger.info(
            f"{batch_i + 1} / {len(dataloader)} batches. batch mean ppl: {batch_ppl.mean().item():.4f}"
        )

    ppls = np.array(ppls, dtype=np.float64)
    mean_ppl = ppls.mean()
    # std_ppl = ppls_np.std()  # population std
    std_ppl_sample = ppls.std(ddof=1)  # sample std
    logger.info(f"mean ppl: {mean_ppl:.4f}, std_ppl: {std_ppl_sample}")

    np.savetxt(out_path, ppls)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
