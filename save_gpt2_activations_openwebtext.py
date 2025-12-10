import json
import os

import numpy as np
import torch
from datasets import load_dataset
from librosa import ex
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from utils import get_logger


class HookCollector:
    def __init__(self, model, layers_to_hook):
        self.layers = layers_to_hook
        self.data = {}
        self.hooks = []

        for name, module in model.named_modules():
            if name in layers_to_hook:
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def _make_hook(self, name):
        def hook(_, __, output):
            self.data[name] = output.detach().cpu().to(torch.float32).numpy()

        return hook

    def clear(self):
        self.data = {}

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def probe_shapes(model, layers, vocab_size, max_len, device):
    collector = HookCollector(model, layers)
    dummy = torch.randint(0, vocab_size, (1, max_len)).to(device)

    with torch.no_grad():
        model(dummy)

    shapes = {layer: arr.shape for layer, arr in collector.data.items()}
    collector.remove()
    return shapes


class MemmapStore:
    def __init__(self, total_examples, layer_shapes, save_dir):
        self.save_dir = save_dir
        self.i = 0

        self.mem = {}
        self.meta = {}

        for layer, shape in layer_shapes.items():
            final_shape = (total_examples,) + shape[1:]
            fname = f"{layer.replace('.', '_')}.dat"
            path = os.path.join(save_dir, fname)

            self.mem[layer] = np.memmap(
                path, mode="w+", dtype=np.float32, shape=final_shape
            )

            self.meta[layer] = {
                "path": path,
                "shape": list(final_shape),
                "dtype": "float32",
            }

        with open(os.path.join(save_dir, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2)

    def append(self, activations):
        for layer, arr in activations.items():
            self.mem[layer][self.i] = arr.squeeze(0)
        self.i += 1

    def flush(self):
        for mm in self.mem.values():
            mm.flush()


def get_blackbox_model(model_name, device):
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device).eval()
    return model, tokenizer


def get_dataset(dataset):
    return load_dataset(dataset, split="train")


class CFG:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "/scratch/b5bq/pu22650.b5bq/activations_{model}_{dataset}"
    max_length = 64
    model_name = "gpt2"
    dataset = "Geralt-Targaryen/openwebtext2"


def main():
    args = CFG()
    logger = get_logger(__file__)
    args.save_dir = args.save_dir.format(
        model=args.model_name, dataset=args.dataset.replace("/", "_")
    )
    logger.info(f"the output dir: {args.save_dir}")
    os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    model, tokenizer = get_blackbox_model(args.model_name, args.device)
    logger.info(f"model {args.model_name} loaded")

    # Load dataset
    ds = get_dataset(args.dataset)
    total_examples = len(ds)
    logger.info(f"dataset {args.save_dir} loaded with {total_examples} examples")

    # Determine which layers to hook (MLP blocks)
    layers_to_hook = [
        name
        for name, _ in model.named_modules()
        if "mlp.c_fc" in name or "mlp.c_proj" in name
    ]  # TODO: 我目前分析的是mlp.c_fc，也就是说，这不是residule stream

    # Probe activation shapes
    logger.info(f"start to probe shapes ...")
    layer_shapes = probe_shapes(
        model, layers_to_hook, model.config.vocab_size, args.max_length, args.device
    )
    logger.info(
        f"hook layer:shape to collect activations {"; ".join([f"{layer}:{shape}" for layer, shape in layer_shapes.items()])}"
    )

    # Initialize memmap store + collector
    logger.info(f"start to create memmap store ...")
    store = MemmapStore(total_examples, layer_shapes, args.save_dir)
    logger.info(
        f"create memmap store. the meta file saved at {os.path.join(args.save_dir, "meta.json")}"
    )
    collector = HookCollector(model, layers_to_hook)

    # Main loop # TODO: batch process
    for example_i in range(total_examples):
        collector.clear()

        text = ds[example_i]["text"]
        tokens = tokenizer(
            text,
            return_tensors="pt",
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
        ).to(args.device)

        with torch.no_grad():
            model(**tokens)

        store.append(collector.data)

        if example_i % 100 == 0:
            logger.info(f"processed {example_i+1}/{total_examples}")

    # Finalize
    store.flush()
    collector.remove()
    logger.info(f"activations saved at {args.save_dir}")


if __name__ == "__main__":
    main()
