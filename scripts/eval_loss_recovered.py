"""
Evaluate Loss Recovered for DecomposeSAEModel.

Loss Recovered = 1 - (CE_sae - CE_id) / (CE_zero - CE_id)

Requires full GPT-2 forward passes with hooks to replace layer outputs.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from vasae.models.decompose_sae import DecomposeSAEModel
from vasae.models.factory import (
    BlackBoxModelConfig,
    get_blackbox_model,
    load_embedding_layer,
)
from vasae.utils.log import get_logger
from vasae.utils.seed import set_seed


def make_hook(layer_idx, intervention_fn):
    """Create a forward hook that replaces the hidden state output of a transformer layer."""
    def hook(module, input, output):
        # Newer transformers may return a plain Tensor; older versions return a tuple
        if isinstance(output, torch.Tensor):
            return intervention_fn(output)
        h = output[0]
        h_new = intervention_fn(h)
        return (h_new,) + output[1:]
    return hook


@torch.no_grad()
def compute_ce_with_hook(gpt2_model, input_ids, attention_mask, layer_idx, intervention_fn):
    """Run GPT-2 forward pass with a hook on the specified layer, return per-token CE loss."""
    layer = gpt2_model.transformer.h[layer_idx]
    handle = layer.register_forward_hook(make_hook(layer_idx, intervention_fn))
    try:
        outputs = gpt2_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        # shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )
        loss = loss.view(shift_labels.shape)
        # mask and average
        loss = (loss * shift_mask).sum() / shift_mask.sum()
        return loss.item()
    finally:
        handle.remove()


@torch.no_grad()
def compute_ce_no_hook(gpt2_model, input_ids, attention_mask):
    """Run GPT-2 forward pass without any hook, return CE loss."""
    outputs = gpt2_model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    )
    loss = loss.view(shift_labels.shape)
    loss = (loss * shift_mask).sum() / shift_mask.sum()
    return loss.item()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-idx", type=int, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--pca-path", type=str, required=True)
    parser.add_argument("--d-pca", type=int, required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # data for re-tokenization
    parser.add_argument(
        "--data-dir", type=str,
        default=r"/scratch/b5bq/pu22650.b5bq/activations_gpt2_Geralt-Targaryen_openwebtext2",
    )

    # blackbox model
    parser.add_argument("--blackbox-model-name", type=str, default="gpt2")
    parser.add_argument(
        "--blackbox-model-dir", type=str,
        default=r"/scratch/b5bq/pu22650.b5bq/VASAE_out/BlackBoxModels/gpt2",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)
    logger = get_logger()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load GPT-2
    gpt2_model, tokenizer = get_blackbox_model(args.blackbox_model_name, device)

    # Load embedding for DecomposeSAEModel
    bb_cfg = BlackBoxModelConfig(
        name=args.blackbox_model_name,
        dir=Path(args.blackbox_model_dir),
    )
    emb = load_embedding_layer(bb_cfg)
    vocab_size, model_dim = emb.weight.shape

    # Load PCA
    pca_data = torch.load(args.pca_path, map_location=device, weights_only=True)
    W_full = pca_data["W_full"]

    # Build DecomposeSAEModel
    sae_model = DecomposeSAEModel(model_dim, vocab_size, args.d_pca, args.k).to(device)
    sae_model.attach_embedding(emb, freeze=True)
    sae_model.attach_pca(W_full[:, :args.d_pca].to(device))
    sae_model.load_state_dict(
        torch.load(args.model_path, map_location=device, weights_only=True),
        strict=False,
    )
    sae_model.eval()

    # Load text data for re-tokenization
    data_info_path = Path(args.data_dir) / "data_info.json"
    with open(data_info_path) as f:
        data_info = json.load(f)

    # Use test split indices (same seed, 70/20/10 split)
    n_total = len(data_info)
    train_size = int(0.7 * n_total)
    valid_size = int(0.2 * n_total)

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(n_total, generator=generator).tolist()
    test_indices = indices[train_size + valid_size:]

    # Limit samples
    test_indices = test_indices[:args.n_samples]

    # Collect texts and tokenize
    texts = []
    for idx in test_indices:
        tokens = data_info[idx]["display_text"]
        text = "".join(tokens)
        texts.append(text)

    layer_idx = args.layer_idx

    # Process in batches
    sum_ce_id = 0.0
    sum_ce_sae = 0.0
    sum_ce_zero = 0.0
    sum_ce_sparse = 0.0
    n_processed = 0

    for i in range(0, len(texts), args.batch_size):
        batch_texts = texts[i : i + args.batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # CE(id): no intervention
        ce_id = compute_ce_no_hook(gpt2_model, input_ids, attention_mask)

        # CE(SAE): full reconstruction
        def sae_intervention(h):
            shape = h.shape
            h_flat = h.view(-1, model_dim)
            out = sae_model(h_flat)
            return out.h_recon.view(shape)

        ce_sae = compute_ce_with_hook(gpt2_model, input_ids, attention_mask, layer_idx, sae_intervention)

        # CE(zero): zero ablation
        def zero_intervention(h):
            return torch.zeros_like(h)

        ce_zero = compute_ce_with_hook(gpt2_model, input_ids, attention_mask, layer_idx, zero_intervention)

        # CE(sparse): sparse-only reconstruction
        def sparse_intervention(h):
            shape = h.shape
            h_flat = h.view(-1, model_dim)
            out = sae_model(h_flat)
            return (out.h_sparse + sae_model.bias).view(shape)

        ce_sparse = compute_ce_with_hook(gpt2_model, input_ids, attention_mask, layer_idx, sparse_intervention)

        batch_n = len(batch_texts)
        sum_ce_id += ce_id * batch_n
        sum_ce_sae += ce_sae * batch_n
        sum_ce_zero += ce_zero * batch_n
        sum_ce_sparse += ce_sparse * batch_n
        n_processed += batch_n

        logger.info(
            f"Batch {i // args.batch_size + 1}: "
            f"CE_id={ce_id:.4f} CE_sae={ce_sae:.4f} CE_zero={ce_zero:.4f} CE_sparse={ce_sparse:.4f}"
        )

    avg_ce_id = sum_ce_id / n_processed
    avg_ce_sae = sum_ce_sae / n_processed
    avg_ce_zero = sum_ce_zero / n_processed
    avg_ce_sparse = sum_ce_sparse / n_processed

    loss_recovered = 1.0 - (avg_ce_sae - avg_ce_id) / (avg_ce_zero - avg_ce_id)
    loss_recovered_sparse = 1.0 - (avg_ce_sparse - avg_ce_id) / (avg_ce_zero - avg_ce_id)

    results = {
        "layer_idx": layer_idx,
        "d_pca": args.d_pca,
        "k": args.k,
        "n_samples": n_processed,
        "ce_id": avg_ce_id,
        "ce_sae": avg_ce_sae,
        "ce_zero": avg_ce_zero,
        "ce_sparse": avg_ce_sparse,
        "loss_recovered": loss_recovered,
        "loss_recovered_sparse": loss_recovered_sparse,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Loss recovered: {loss_recovered:.4f}")
    logger.info(f"Loss recovered (sparse-only): {loss_recovered_sparse:.4f}")
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
