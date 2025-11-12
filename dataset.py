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

    def __init__(self, meta_path, layer_name):
        self.layer_name = layer_name

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

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        """
        Returns:
            torch.Tensor: [seq_len, hidden_dim]
        """
        arr = self.memmap[idx]
        return torch.from_numpy(arr)
