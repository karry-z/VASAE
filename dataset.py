import json

import numpy as np
import torch
from torch.utils.data import Dataset


class GPT2LayerActivations(Dataset):
    """
    PyTorch Dataset for loading activations of a specific GPT-2 layer
    from a .dat memmap file defined in the metadata JSON.

    Meta JSON format:
    {
      "transformer.h.0.mlp.c_fc": {
        "path": "/path/to/layer.dat",
        "shape": [num_examples, seq_len, hidden_dim],
        "dtype": "float32"
      },
      ...
    }
    """

    def __init__(self, meta_path, layer_name, use_centralize=False):
        self.layer_name = layer_name
        self.meta_path = meta_path
        self.use_centralize = use_centralize
        # Load metadata
        with open(meta_path, "r") as f:
            meta = json.load(f)

        if layer_name not in meta:
            raise KeyError(
                f"Layer '{layer_name}' not found in metadata keys: {list(meta.keys())}"
            )

        info = meta[layer_name]
        self.path = info["path"]
        self.shape = tuple(info["shape"])
        self.dtype = info["dtype"]

        # Memory-map the activation file
        self.memmap = np.memmap(self.path, mode="r", dtype=self.dtype, shape=self.shape)
        self.num_examples = self.shape[0]

        if "mean" in info:
            self.mean = np.array(info["mean"], dtype=np.float32)
        else:
            self.__compute_and_store_mean()

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        """
        Returns:
            torch.Tensor: [seq_len, hidden_dim]
        """
        arr = self.memmap[idx]
        if self.use_centralize:
            arr = self.centralize(arr)
        return torch.from_numpy(arr)

    def centralize(self, x):
        # https://cdn.openai.com/papers/sparse-autoencoders.pdf the paper of OpenAI SAE pipeline include normalization on activations but it does not shown data preprocessing in the code repo, nor does the BatchTopKSAE repo.
        return x - self.mean

    def __compute_and_store_mean(self):
        mean = np.zeros(self.shape[1:], dtype=np.float64)

        for i in range(self.num_examples):
            mean += self.memmap[i]

        mean /= self.num_examples
        mean = mean.astype(np.float32)
        self.mean = mean

        with open(self.meta_path, "r") as f:
            meta = json.load(f)

        meta[self.layer_name]["mean"] = mean.tolist()

        with open(self.meta_path, "w") as f:
            json.dump(meta, f)
