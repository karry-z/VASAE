import argparse
import json
import os

import numpy as np
import torch
from datasets import load_dataset
from transformers import LlavaForConditionalGeneration, AutoProcessor

from vasae.configs.data import DataConfig
from vasae.data.data_schema import Meta
from vasae.data.dataset import load_meta
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
                output = output[0]
            self.data[name] = output.detach().cpu().to(torch.float32).numpy()

        return hook

    def clear(self):
        self.data = {}

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


class MemmapStore:
    def __init__(self, total_examples, layer_shapes, save_dir):
        self.save_dir = save_dir
        self.memmap_arr_i = 0

        self.mem = {}
        self.meta = {}

        data_dir = os.path.join(save_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        for layer, shape in layer_shapes.items():
            final_shape = (total_examples,) + shape[1:]
            fname = f"{layer.replace('.', '_')}.dat"
            path = os.path.join(data_dir, fname)

            self.mem[layer] = np.memmap(
                path, mode="w+", dtype=np.float32, shape=final_shape
            )

            # Use relative paths in meta (relative to save_dir)
            self.meta[layer] = {
                "path": f"data/{fname}",
                "shape": list(final_shape),
                "dtype": "float32",
                "mean": f"mean/{layer.replace('.', '_')}_mean.npy",
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


def probe_shapes(model, processor, layers, max_len, device):
    """Probe activation shapes with a dummy forward pass."""
    collector = HookCollector(model, layers)

    # Create a dummy input with a small blank image
    from PIL import Image
    dummy_image = Image.new("RGB", (336, 336), color=(128, 128, 128))
    dummy_text = "USER: <image>\nHello\nASSISTANT:"
    inputs = processor(
        text=dummy_text,
        images=dummy_image,
        return_tensors="pt",
        padding="max_length",
        max_length=max_len,
    ).to(device)

    with torch.no_grad():
        model(**inputs)

    shapes = {layer: arr.shape for layer, arr in collector.data.items()}
    collector.remove()
    return shapes


def save_means(meta: Meta, cfg: DataConfig):
    mean_dir = cfg.data_dir / "mean"
    mean_dir.mkdir(parents=True, exist_ok=True)
    for layer in meta:
        mm = np.memmap(
            cfg.data_dir / meta[layer].path,
            mode="r",
            dtype=meta[layer].dtype,
            shape=tuple(meta[layer].shape),
        )
        mean = mm.mean(axis=0, dtype=np.float64).astype(np.float32)
        path = mean_dir / f"{layer.replace('.', '_')}_mean.npy"
        np.save(path, mean)
        path.chmod(0o444)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/scratch/b5bq/pu22650.b5bq/",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--blackbox-model",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="lmms-lab/COCO-Caption",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=20000,
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to collect, e.g. '0,4,8,12,16,20,24,28,31'. Default: all layers.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger()

    out_dir = os.path.join(
        args.save_dir,
        "activations_{blackbox_model}_{dataset}".format(
            blackbox_model=args.blackbox_model.replace("/", "_"),
            dataset=args.dataset.replace("/", "_"),
        ),
    )
    logger.info(f"the output dir: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    logger.info(f"loading model {args.blackbox_model} ...")
    processor = AutoProcessor.from_pretrained(args.blackbox_model)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.blackbox_model, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    logger.info(f"model {args.blackbox_model} loaded")

    # Discover embed_tokens and lm_head in the model
    embed_tokens = None
    lm_head = None
    for name, module in model.named_modules():
        if name.endswith("embed_tokens") and isinstance(module, torch.nn.Embedding):
            embed_tokens = module
            logger.info(f"found embed_tokens at: {name}")
        if name.endswith("lm_head") and isinstance(module, torch.nn.Linear):
            lm_head = module
            logger.info(f"found lm_head at: {name}")
    assert embed_tokens is not None, "Could not find embed_tokens in model"
    assert lm_head is not None, "Could not find lm_head in model"

    # Save embedding and unembedding layers
    emb_unemb_dir = os.path.join(
        args.save_dir,
        "VASAE_out",
        "BlackBoxModels",
        args.blackbox_model.replace("/", "_"),
    )
    os.makedirs(emb_unemb_dir, exist_ok=True)
    torch.save(
        embed_tokens,
        os.path.join(emb_unemb_dir, "emb.pth"),
    )
    torch.save(
        lm_head,
        os.path.join(emb_unemb_dir, "unemb.pth"),
    )
    logger.info(f"saved emb.pth and unemb.pth to {emb_unemb_dir}")

    # Load dataset
    logger.info(f"loading dataset {args.dataset} ...")
    # Try train split first, fall back to val
    try:
        ds = load_dataset(args.dataset, split="train")
    except ValueError:
        logger.info("No train split found, using val split")
        ds = load_dataset(args.dataset, split="val")
    ds = ds.shuffle(seed=42).select(range(min(args.num_examples, len(ds))))
    total_examples = len(ds)
    logger.info(f"dataset loaded with {total_examples} examples")

    # Determine which layers to hook (only language model decoder layers)
    import re
    layers_to_hook = sorted(
        [
            name
            for name, _ in model.named_modules()
            if re.fullmatch(r".*language_model.*layers\.\d+", name)
        ],
        key=lambda n: int(n.rsplit(".", 1)[-1]),
    )
    if args.layers is not None:
        selected = set(int(x) for x in args.layers.split(","))
        layers_to_hook = [l for l in layers_to_hook if int(l.rsplit(".", 1)[-1]) in selected]
    logger.info(f"hooking {len(layers_to_hook)} decoder layers, e.g. {layers_to_hook[0]}")

    # Probe activation shapes
    logger.info("probing activation shapes ...")
    layer_shapes = probe_shapes(
        model, processor, layers_to_hook, args.max_length, args.device
    )
    logger.info(
        "hook layer:shape to collect activations "
        + "; ".join(f"{layer}:{shape}" for layer, shape in layer_shapes.items())
    )

    # Initialize memmap store + collector
    logger.info("creating memmap store ...")
    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_path):
        logger.warning(
            f"{meta_path} exists. a copy made and the original one will be covered"
        )
        os.rename(meta_path, meta_path + ".bak")

    store = MemmapStore(total_examples, layer_shapes, out_dir)
    logger.info("memmap store created.")

    collector = HookCollector(model, layers_to_hook)

    # Main loop
    data_info = []
    for example_i in range(total_examples):
        collector.clear()

        example = ds[example_i]
        image = example["image"]  # PIL Image from datasets
        # Support multiple dataset formats: answer (lmms-lab), captions (whyen-wang), sentences.raw (HuggingFaceM4)
        captions = (
            example.get("answer")
            or example.get("captions")
            or example.get("sentences", {}).get("raw")
        )
        caption = captions[0] if isinstance(captions, list) else captions
        prompt = f"USER: <image>\n{caption}\nASSISTANT:"

        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=args.max_length,
        ).to(args.device)

        with torch.no_grad():
            model(**inputs)

        store.append(collector.data)
        data_info.append(
            {
                "example_i": example_i,
                "caption": caption,
                "display_text": caption,
            }
        )

        if example_i % 100 == 0:
            logger.info(f"processed {example_i+1}/{total_examples}")

    # Finalize
    store.flush()

    # Save data_info
    info_path = os.path.join(out_dir, "data_info.json")
    with open(info_path, "w") as f:
        json.dump(data_info, f, indent=2)

    # Save mean values
    data_cfg = DataConfig(data_dir=out_dir)
    meta: Meta = load_meta(data_cfg.data_dir / "meta.json")
    save_means(meta, data_cfg)

    collector.remove()
    logger.info(f"activations saved at {out_dir}")


if __name__ == "__main__":
    main()
