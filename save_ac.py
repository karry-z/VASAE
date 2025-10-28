import torch
import pickle
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
import os
import gc
from tqdm import tqdm

# ---------------- CONFIG ----------------
model_name = "gpt2"
save_dir = "/mnt/data/activation_cache"
os.makedirs(save_dir, exist_ok=True)
chunk_size = 100           # number of examples per save
max_length = 512           # tokenizer truncation limit
log_interval = 10          # log every N examples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- LOAD ----------------
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
model.eval()
ds = load_dataset("squad", split="validation")
total_files = (len(ds) + chunk_size - 1) // chunk_size
print(f"[INFO] Estimated output files: {total_files}")

# ---------------- FIND MLP LAYERS ----------------
layers_to_hook = [n for n, _ in model.named_modules() if "mlp.c_" in n]
print(f"Hooking {len(layers_to_hook)} MLP layers.")

# ---------------- ACT CACHE ----------------
class ActivationCache:
    def __init__(self, model, layers_to_hook):
        self.model = model
        self.layers_to_hook = layers_to_hook
        self.activations = {}
        self.current_example_id = None
        self._hooks = []
        self.register_hooks()

    def _hook_fn(self, layer_name):
        def fn(_, __, output):
            self.activations[self.current_example_id][layer_name] = output.detach().cpu()
        return fn

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.layers_to_hook:
                self._hooks.append(module.register_forward_hook(self._hook_fn(name)))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def save_chunk(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.activations, f)
        print(f"[INFO] Saved {len(self.activations)} examples to {path}")

    def clear(self):
        self.activations.clear()
        gc.collect()
        torch.cuda.empty_cache()

# ---------------- RESTART SUPPORT ----------------
def get_last_index():
    saved = [f for f in os.listdir(save_dir) if f.startswith("activations_")]
    if not saved:
        return 0
    last_file = sorted(saved)[-1]
    end = int(last_file.split("_")[-1].split(".")[0])
    print(f"[INFO] Resuming from index {end}")
    return end

# ---------------- MAIN ----------------
cache = ActivationCache(model, layers_to_hook)
start_idx = get_last_index()

try:
    for i in tqdm(range(start_idx, len(ds))):
        cache.current_example_id = i
        cache.activations[i] = {}

        ex = ds[i]
        text = f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer:"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            pred_ids = torch.argmax(outputs.logits, dim=-1)
            pred_text = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
            print(f"[PREDICT] GPT output: {pred_text[:150]}")  # truncate long outputs
            print(f"[GOLD] Answer: {ex['answers']['text'][0] if ex['answers']['text'] else 'N/A'}")

        if (i + 1) % log_interval == 0:
            print(f"[LOG] Processed {i+1}/{len(ds)} examples.")

        # Save periodically
        if (i + 1) % chunk_size == 0 or i == len(ds) - 1:
            save_path = os.path.join(save_dir, f"activations_{i+1-chunk_size:05d}_{i+1:05d}.pkl")
            cache.save_chunk(save_path)
            cache.clear()

except KeyboardInterrupt:
    print("\n[INTERRUPTED] Saving current progress...")
    if cache.activations:
        save_path = os.path.join(save_dir, f"activations_partial_{start_idx:05d}_{i:05d}.pkl")
        cache.save_chunk(save_path)
    cache.remove_hooks()
    raise

cache.remove_hooks()
print("All activations saved successfully.")
