import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from vasae.configs.data import DataConfig
from vasae.data.data_schema import LayerMeta, Meta


def load_meta(meta_path: Path) -> Meta:
    meta_path = Path(meta_path)
    with meta_path.open() as f:
        raw: dict = json.load(f)

    base = meta_path.parent

    return {
        name: LayerMeta(
            path=base / meta["path"],
            shape=meta["shape"],
            dtype=meta["dtype"],
            mean=base / meta["mean"],
        )
        for name, meta in raw.items()
    }


class GPT2LayerActivations(Dataset):
    """
    PyTorch Dataset for loading activations of a specific GPT-2 layer
    from a .dat memmap file defined in the metadata JSON.
    """

    def __init__(self, data_cfg: DataConfig):
        self.layer_name = data_cfg.layer_name
        self.meta_path = data_cfg.data_dir / "meta.json"
        self.use_centralize = data_cfg.use_centralize

        data_folder = data_cfg.data_dir
        # load meta
        meta = load_meta(Path(self.meta_path))

        info = meta[self.layer_name]
        self.path = info.path
        self.shape = info.shape
        self.dtype = info.dtype

        # Memory-map the activation file
        self.memmap = np.memmap(self.path, mode="r", dtype=self.dtype, shape=self.shape)
        self.num_examples = self.shape[0]

        # load mean
        self.mean = np.load(info.mean)

        # load data_info
        data_info_path = data_folder / "data_info.json"
        with data_info_path.open() as f:
            self.data_info = json.load(f)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, example_i):
        """
        Returns:
            torch.Tensor: [seq_len, hidden_dim]
        """
        arr = self.memmap[example_i]
        if self.use_centralize:
            arr = self.centralize(arr)

        return {
            "activations": torch.from_numpy(arr.copy()),
            "display_text": self.data_info[example_i]["display_text"],
        }

    def centralize(self, x):
        # https://cdn.openai.com/papers/sparse-autoencoders.pdf the paper of OpenAI SAE pipeline include normalization on activations but it does not shown data preprocessing in the code repo, nor does the BatchTopKSAE repo.
        return x - self.mean


def get_dataloader(data_cfg, seed):
    generator = torch.Generator().manual_seed(seed)
    dataset = GPT2LayerActivations(data_cfg=data_cfg)

    # split into train, valid, test 7:2:1
    train_size = int(0.7 * len(dataset))
    valid_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    train_dataset, valid_dataset, test_dataset = random_split(
        dataset,
        [train_size, valid_size, test_size],
        generator,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=data_cfg.train_batchsize, shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=data_cfg.valid_batchsize, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=data_cfg.test_batchsize, shuffle=False
    )

    return train_loader, valid_loader, test_loader
