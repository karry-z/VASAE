import os
import gc
import json
import torch
import numpy as np
import logging
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


class NumpyActivationCache:
    """
    Efficient activation capture using np.memmap (.dat files) with JSON metadata.
    Automatically pads/truncates to fixed sequence length.
    """
    def __init__(self, model, layers_to_hook, save_dir, total_examples,
                 max_length=512, vocab_size=50257):
        self.model = model
        self.layers_to_hook = layers_to_hook
        self.save_dir = save_dir
        self.total_examples = total_examples
        self.max_length = max_length
        self.vocab_size = vocab_size

        self.current_example_id = None
        self._hooks = []
        self._memmaps = {}
        self.meta = {}

        sample_shapes = self._get_first_sample_shapes()
        self._init_memmaps(sample_shapes)
        self.register_hooks()

    def _get_first_sample_shapes(self):
        """Run dummy forward pass to infer activation shapes."""
        activations = {}
        hooks = []

        def make_hook(name):
            def hook(_, __, output):
                activations[name] = output.detach().cpu()
            return hook

        for name, module in self.model.named_modules():
            if name in self.layers_to_hook:
                hooks.append(module.register_forward_hook(make_hook(name)))

        dummy_input = torch.randint(
            low=0, high=self.vocab_size, size=(1, self.max_length)
        ).to(next(self.model.parameters()).device)

        with torch.no_grad():
            self.model(dummy_input)

        for h in hooks:
            h.remove()

        shapes = {layer: act.shape for layer, act in activations.items()}
        logging.info(f"Sample activation shapes: {shapes}")
        return shapes

    def _init_memmaps(self, sample_shapes):
        """Initialize .dat memmaps and record metadata."""
        os.makedirs(self.save_dir, exist_ok=True)
        for layer, shape in sample_shapes.items():
            memmap_shape = (self.total_examples,) + shape[1:]  # drop batch dim
            path = os.path.join(self.save_dir, f"{layer.replace('.', '_')}.dat")
            logging.info(f"Allocating memmap for {layer} shape={memmap_shape}")

            self._memmaps[layer] = np.memmap(
                path, mode="w+", dtype=np.float32, shape=memmap_shape
            )

            # record metadata
            self.meta[layer] = {
                "path": path,
                "shape": list(memmap_shape),
                "dtype": "float32"
            }

        # save metadata JSON
        meta_path = os.path.join(self.save_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(self.meta, f, indent=2)
        logging.info(f"Metadata written to {meta_path}")

    def register_hooks(self):
        """Register hooks that write activations with padding/truncation."""
        def make_hook(layer_name):
            def hook(_, __, output):
                arr = output.detach().cpu().to(torch.float32).numpy().squeeze(0)

                self._memmaps[layer_name][self.current_example_id] = arr
            return hook

        for name, module in self.model.named_modules():
            if name in self.layers_to_hook:
                self._hooks.append(module.register_forward_hook(make_hook(name)))
        logging.info(f"Registered hooks for {len(self._hooks)} layers.")

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def flush(self):
        logging.info("start to save memmaps disk.")
        for mmap in self._memmaps.values():
            mmap.flush()
        logging.info("Memmaps flushed to disk.")





model_name = "gpt2"
save_dir = "/mnt/data/gpt2_activations"
max_length = 512
log_interval = 10

os.makedirs(save_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
model.eval()
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id


ds = load_dataset("squad", split="validation")  # smaller for test
total_examples = len(ds)
logging.info(f"Dataset size: {total_examples}")

layers_to_hook = [n for n, _ in model.named_modules() if "mlp.c_" in n]
logging.info(f"Hooking {len(layers_to_hook)} layers: {layers_to_hook}")

cache = NumpyActivationCache(model, layers_to_hook, save_dir, total_examples, max_length=max_length)

try:
    for i in range(total_examples):
        cache.current_example_id = i
        ex = ds[i]
        text = f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer:"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length).to(device)

        with torch.no_grad():
            _ = model(**inputs)

        if (i + 1) % log_interval == 0:
            logging.info(f"Processed {i+1}/{total_examples} examples.")

except KeyboardInterrupt:
    logging.warning("Interrupted! Flushing and removing hooks...")
    cache.flush()
    cache.remove_hooks()
    raise

cache.flush()
cache.remove_hooks()
logging.info("All activations saved successfully.")