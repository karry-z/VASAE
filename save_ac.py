import os
import torch
from transformers import GPT2Tokenizer, GPT2Model
import json
import pickle

class ActivationCache:
    def __init__(self, model, layers_to_hook=None):
        self.model = model
        self.layers_to_hook = layers_to_hook or []  # list of layer names or indices
        self.activations = {}  # dict: example_id → { layer_name → tensor }

    def _hook_fn(self, layer_name):
        def fn(module, input, output):
            # detach to move off graph, convert to CPU if needed
            self.activations[self.current_example_id][layer_name] = output.detach().cpu()
        return fn

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.layers_to_hook:
                module.register_forward_hook(self._hook_fn(name))

    def clear(self):
        self.activations = {}

    def save(self, path):
        # save as pickle (alternatively use torch.save)
        with open(path, 'wb') as f:
            pickle.dump(self.activations, f)

def collect_activations(
    texts,
    cache: ActivationCache,
    tokenizer,
    model,
    device=torch.device("cpu"),
    batch_size=1,
):
    model.eval()
    model.to(device)
    cache.register_hooks()
    for idx, text in enumerate(texts):
        cache.current_example_id = idx
        cache.activations[idx] = {}
        # tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # forward pass
        with torch.no_grad():
            _ = model(**inputs)
        # (activations stored via hooks)
    return cache.activations

def main():
    # config
    model_name = "gpt2"
    layers_to_hook = [
        "h.0.mlp.c_fc",
        "h.0.mlp.c_proj",
        # add other layer names or indices as desired
    ]
    save_path = "activation_cache.pkl"
    texts = [
        "This is the first example.",
        "Here is another sentence for activation extraction.",
        # add your corpus
    ]

    # load model & tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init cache
    cache = ActivationCache(model, layers_to_hook=layers_to_hook)

    # collect
    activations = collect_activations(texts, cache, tokenizer, model, device=device)

    # save
    cache.save(save_path)
    print(f"Saved activations for {len(texts)} examples to {save_path}")

if __name__ == "__main__":
    main()
