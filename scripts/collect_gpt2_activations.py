import argparse
import json
import os

import numpy as np
import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from vasae.utils.log import get_logger


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
            if isinstance(output, tuple):
                output = output[0]  # the output of a GPT2Block is a tuple
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


def get_display_tokens(text, enc, tokenizer):
    input_ids = enc["input_ids"][0]
    offsets = enc["offset_mapping"][0].tolist()

    display_tokens = []
    for i, (s, e) in enumerate(offsets):
        if s == e == 0:  # special tokens
            display_tokens.append(tokenizer.convert_ids_to_tokens(int(input_ids[i])))
        else:
            display_tokens.append(
                text[s:e]
            )  # 直接取原始位置的文本，避免可视化得到不干净的token
    return display_tokens


class MemmapStore:
    def __init__(self, total_examples, layer_shapes, save_dir):
        self.save_dir = save_dir
        self.memmap_arr_i = 0

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

        meta_path = os.path.join(save_dir, "meta.json")
        if os.path.exists(meta_path):
            raise FileExistsError(
                f"{meta_path} exists. please consider remove it or create a copy of it."
            )
        with open(meta_path, "w") as f:
            json.dump(self.meta, f, indent=2)

    def append(self, activations):
        for layer, arr in activations.items():
            self.mem[layer][self.memmap_arr_i] = arr.squeeze(0)
        self.memmap_arr_i += 1

    def flush(self):
        for mm in self.mem.values():
            mm.flush()


def get_blackbox_model(model_name, device):
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device).eval()
    return model, tokenizer


def get_dataset(dataset: str):
    if "openwebtext" in dataset:
        ds = load_dataset(dataset, split="train")
        ds = ds.add_column(
            "orig_idx", list(range(len(ds)))
        )  # keep orig_idx to track original text
        ds = ds.shuffle(seed=42).select(range(20000))

    elif "squad" in dataset:
        ds = load_dataset(dataset, split="validation")

        def build(ex):
            ans = ex["answers"]["text"][0] if ex["answers"]["text"] else ""
            return {
                "text": (
                    "context: "
                    + ex["context"].strip()
                    + "\nquestion: "
                    + ex["question"].strip()
                    + "\nanswer: "
                    + ans.strip()
                )
            }

        ds = ds.map(build)
    else:
        raise ValueError("invalid dataset")

    return ds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        help="device, cpu or cuda",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/scratch/b5bq/pu22650.b5bq/",
        help="a folder under this directory will be created to store activations",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=64,
        help="max length of context",
    )
    parser.add_argument(
        "--blackbox_model",
        type=str,
        default="gpt2",
        help="blackbox model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Geralt-Targaryen/openwebtext2",
        help="dataset name to collect activations on, support Geralt-Targaryen/openwebtext2, squad",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="logs/collect_gpt2_activations",
        help="log path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger(args.log)
    out_dir = os.path.join(
        args.save_dir,
        "activations_{blackbox_model}_{dataset}".format(
            blackbox_model=args.blackbox_model, dataset=args.dataset.replace("/", "_")
        ),
    )
    logger.info(f"the output dir: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    model, tokenizer = get_blackbox_model(args.blackbox_model, args.device)
    logger.info(f"model {args.blackbox_model} loaded")

    # Load dataset
    ds = get_dataset(args.dataset)
    total_examples = len(ds)
    logger.info(f"dataset {out_dir} loaded with {total_examples} examples")

    # Determine which layers to hook
    layers_to_hook = [
        name
        for name, module in model.named_modules()
        if name.startswith("transformer.h.") and name.count(".") == 2
    ]

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
    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_path):
        logger.warning(
            f"{meta_path} exists. a copy made and the original one will be covered"
        )
        os.rename(meta_path, meta_path + ".bak")

    store = MemmapStore(total_examples, layer_shapes, out_dir)
    logger.info(f"create memmap store.")

    collector = HookCollector(model, layers_to_hook)

    # Main loop
    data_info = []
    for example_i in range(total_examples):
        collector.clear()

        text = ds[example_i]["text"]
        tokens = tokenizer(
            text,
            return_tensors="pt",
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,  # 保存原始位置(start, end)
            add_special_tokens=True,
        ).to(args.device)

        with torch.no_grad():
            model(**tokens)

        store.append(collector.data)
        data_info.append(
            {
                "example_i": example_i,
                "orig_idx": ds[example_i]["orig_idx"],
                "display_text": get_display_tokens(text, tokens, tokenizer),
            }
        )

        if example_i % 100 == 0:
            logger.info(f"processed {example_i+1}/{total_examples}")

    # Finalize
    store.flush()
    meta_path = os.path.join(out_dir, "data_info.json")
    with open(meta_path, "w") as f:
        json.dump(data_info, f, indent=2)
    collector.remove()
    logger.info(f"activations saved at {out_dir}")


if __name__ == "__main__":
    main()
