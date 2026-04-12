"""Evaluate a pre-trained SAE model on wikitext test data.

Loads an SAE from HF format, extracts activations via nnsight,
and computes: MSE, variance explained, logit lens accuracy, CE metrics.

Example:
    python scripts/eval/eval_sae_online.py \
        --sae-path /scratch/.../009_online_gpt2_L11_k32_a0 \
        --model-name gpt2 --layer-idx 11
"""

import argparse
import json
import os
import re

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a pre-trained SAE model")
    p.add_argument("--sae-path", type=str, required=True,
                   help="Path to SAE model directory (config.json + model.safetensors)")
    p.add_argument("--model-name", type=str, default="gpt2")
    p.add_argument("--layer-idx", type=int, default=None,
                   help="Layer index (if None, parse from directory name)")
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dataset", type=str, default="wikitext")
    p.add_argument("--dataset-config", type=str, default="wikitext-103-raw-v1")
    return p.parse_args()


def parse_layer_from_dirname(dirname: str) -> int:
    """Extract layer index from directory name like '009_online_gpt2_L11_k32_a0'."""
    match = re.search(r"_L(\d+)_", dirname)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot parse layer index from directory name: {dirname}")


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    sae_path = Path(args.sae_path)

    # Determine layer index
    if args.layer_idx is not None:
        layer_idx = args.layer_idx
    else:
        layer_idx = parse_layer_from_dirname(sae_path.name)

    print(f"Evaluating SAE: {sae_path}")
    print(f"Layer: {layer_idx}, Model: {args.model_name}")

    # --- Lazy imports ---
    import datasets
    import transformers
    from datasets import load_dataset, load_from_disk
    from nnsight import NNsight

    datasets.disable_progress_bars()
    transformers.logging.set_verbosity_error()

    from vasae.engine.intervention import _get_layer_proxy
    from vasae.models.factory import get_embedding, get_layers, get_lm_head, load_model
    from vasae.models.sae import SAEConfig, SAEModel

    def extract_acts(nn_m, ids, lidx):
        with nn_m.trace(ids):
            layer = _get_layer_proxy(nn_m, lidx)
            h = layer.output.save()
        return h.detach()

    def cross_entropy(logits, labels, mask):
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_mask = mask[:, 1:].contiguous().bool()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                               shift_labels.view(-1), reduction="none")
        loss = loss.view(shift_labels.shape)
        return (loss * shift_mask).sum() / shift_mask.sum()

    # --- Load LLM ---
    print("Loading LLM...")
    llm, tokenizer = load_model(args.model_name, device=str(device))
    nn_model = NNsight(llm)
    emb = get_embedding(llm)
    lm_head = get_lm_head(llm)

    # --- Load SAE ---
    print("Loading SAE...")
    sae_model = SAEModel.from_pretrained(sae_path).to(device)
    sae_model.eval()

    # Attach embedding if tied decoder
    if sae_model.config.tied_decoder:
        sae_model.attach_embedding(emb, freeze=True)

    if sae_model.config.anchor_coeff > 0:
        sae_model.attach_anchor_embedding(emb)

    print(f"SAE config: dim_input={sae_model.config.dim_input}, "
          f"dim_sparse={sae_model.config.dim_sparse}, "
          f"tied={sae_model.config.tied_decoder}, k={sae_model.config.k}")

    # --- Load dataset ---
    print("Loading dataset...")
    # Use shared cache consistent with train_sae_online.py
    save_dir = sae_path.parent
    ds_cache_name = f"{args.dataset}_{args.dataset_config or 'default'}".replace("/", "_")
    data_cache_dir = save_dir / ".data_cache" / ds_cache_name

    if (data_cache_dir / "dataset_info.json").exists():
        print(f"Loading cached dataset from {data_cache_dir}")
        ds = load_from_disk(str(data_cache_dir))
    else:
        print(f"Loading dataset {args.dataset}...")
        ds = load_dataset(args.dataset, args.dataset_config, split="train")
        ds = ds.filter(lambda x: len(x["text"].strip()) > 50)

    # Use the same split offsets as training: skip train+eval, take test
    # train=4000, eval=1000, test=1000 (from 009 config)
    n_total = len(ds)
    n_skip = 5000  # train_samples + eval_samples from 009
    n_test = min(args.n_samples, n_total - n_skip)
    test_ds = ds.select(range(n_skip, n_skip + n_test))
    print(f"Test samples: {n_test}")

    # --- Tokenize ---
    def tokenize_batch(texts):
        return tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=args.max_length,
        )

    # --- Evaluation loop ---
    total_mse = 0.0
    total_ve_mse = 0.0
    total_ve_var = 0.0
    total_ll_correct = 0
    total_ll_total = 0
    total_ce_id = 0.0
    total_ce_sae = 0.0
    total_ce_zero = 0.0
    n_batches = 0

    from torch.utils.data import DataLoader

    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        return tokenize_batch(texts)

    dataloader = DataLoader(test_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn)

    for batch_i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Extract activations
        activations = extract_acts(nn_model, input_ids, layer_idx)

        # SAE forward
        output = sae_model(activations)
        recon = output.hidden_states_recon

        # MSE
        x_flat = activations.reshape(-1, activations.size(-1))
        xr_flat = recon.reshape(-1, recon.size(-1))
        mse = (x_flat - xr_flat).pow(2).mean().item()
        total_mse += mse

        # Variance explained components
        batch_mse_sum = (x_flat - xr_flat).pow(2).sum().item()
        batch_var_sum = (x_flat - x_flat.mean(dim=0, keepdim=True)).pow(2).sum().item()
        total_ve_mse += batch_mse_sum
        total_ve_var += batch_var_sum

        # Logit lens accuracy
        orig_logits = lm_head(activations)
        recon_logits = lm_head(recon)
        orig_tokens = orig_logits.argmax(dim=-1).flatten()
        recon_tokens = recon_logits.argmax(dim=-1).flatten()
        total_ll_correct += (orig_tokens == recon_tokens).sum().item()
        total_ll_total += orig_tokens.numel()

        # CE metrics using hooks
        target_layer = get_layers(llm)[layer_idx]

        # CE identity (no intervention)
        logits_id = llm(input_ids=input_ids, attention_mask=attention_mask).logits
        ce_id = cross_entropy(logits_id, input_ids, attention_mask).item()

        # CE SAE (replace with SAE reconstruction)
        recon_for_patch = recon.detach()
        def sae_hook(mod, inp, out, _recon=recon_for_patch):
            return (_recon,) + out[1:] if isinstance(out, tuple) else _recon
        handle = target_layer.register_forward_hook(sae_hook)
        logits_sae = llm(input_ids=input_ids, attention_mask=attention_mask).logits
        handle.remove()
        ce_sae = cross_entropy(logits_sae, input_ids, attention_mask).item()

        # CE zero (replace with zeros)
        def zero_hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            z = torch.zeros_like(h)
            return (z,) + out[1:] if isinstance(out, tuple) else z
        handle = target_layer.register_forward_hook(zero_hook)
        logits_zero = llm(input_ids=input_ids, attention_mask=attention_mask).logits
        handle.remove()
        ce_zero = cross_entropy(logits_zero, input_ids, attention_mask).item()

        total_ce_id += ce_id
        total_ce_sae += ce_sae
        total_ce_zero += ce_zero
        n_batches += 1

        if (batch_i + 1) % 5 == 0:
            print(f"  Batch {batch_i + 1}/{len(dataloader)}")

    # --- Aggregate ---
    avg_mse = total_mse / n_batches
    variance_explained = 1.0 - total_ve_mse / max(total_ve_var, 1e-8)
    logitlens_acc = total_ll_correct / max(total_ll_total, 1)
    avg_ce_id = total_ce_id / n_batches
    avg_ce_sae = total_ce_sae / n_batches
    avg_ce_zero = total_ce_zero / n_batches
    loss_recovered = 1.0 - (avg_ce_sae - avg_ce_id) / (avg_ce_zero - avg_ce_id + 1e-8)

    results = {
        "config": {
            "sae_path": str(sae_path),
            "model_name": args.model_name,
            "layer_idx": layer_idx,
            "n_samples": n_test,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
        },
        "test": {
            "loss": avg_mse,
            "variance_explained": variance_explained,
            "logitlens_acc": logitlens_acc,
            "ce_id": avg_ce_id,
            "ce_sae": avg_ce_sae,
            "ce_zero": avg_ce_zero,
            "loss_recovered": loss_recovered,
        },
    }

    # Save
    results_path = sae_path / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"  MSE:                {avg_mse:.6f}")
    print(f"  Variance Explained: {variance_explained:.4f}")
    print(f"  LogitLens Acc:      {logitlens_acc * 100:.2f}%")
    print(f"  CE(id):             {avg_ce_id:.4f}")
    print(f"  CE(sae):            {avg_ce_sae:.4f}")
    print(f"  CE(zero):           {avg_ce_zero:.4f}")
    print(f"  Loss Recovered:     {loss_recovered:.4f}")


if __name__ == "__main__":
    main()
