import json
import logging
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

logging.basicConfig(
    format="[%(levelname)s] %(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


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


class VASAE(nn.Module):
    def __init__(self, topk=4, emb_weight=None):
        super().__init__()
        self.encoder = nn.Linear(emb_weight.size(1), emb_weight.size(0))
        self.decoder = nn.Linear(emb_weight.size(0), emb_weight.size(1))
        self.decoder.requires_grad_(False)
        self.decoder.weight = nn.Parameter(emb_weight.T)
        self.k = topk

    def k_sparse(self, x):
        # 实现k-sparse约束
        topk, indices = torch.topk(x, self.k, dim=1)
        mask = torch.zeros_like(x).scatter_(1, indices, 1)
        return x * mask

    def forward(self, x):
        x = self.encoder(x)
        z = self.k_sparse(x)
        x_rec = self.decoder(z)
        return x_rec, z


def train_model(model, train_loader, test_loader, args):
    device = args.device
    num_epochs = args.num_epochs
    # Per-sample loss (no reduction)
    criterion = nn.MSELoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_loss_means, train_loss_stds = [], []
    test_loss_means, test_loss_stds = [], []

    for epoch in range(num_epochs):
        model.train()
        sample_losses = []

        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            data = data.to(device)
            optimizer.zero_grad()

            decoded, _ = model(data)
            # Compute per-sample loss
            loss_per_sample = criterion(decoded, data).mean(
                dim=list(range(1, decoded.ndim))
            )  # mean over all dims except batch
            loss = loss_per_sample.mean()  # overall batch mean for backward pass

            loss.backward()
            optimizer.step()

            sample_losses.extend(loss_per_sample.detach().cpu().numpy())

        train_mean = np.mean(sample_losses)
        train_std = np.std(sample_losses)
        train_loss_means.append(train_mean)
        train_loss_stds.append(train_std)
        print(
            f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_mean:.6f} ± {train_std:.6f}"
        )

        # Evaluation
        model.eval()
        sample_losses = []
        with torch.no_grad():
            for data in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test]"):
                data = data.to(device)
                decoded, _ = model(data)
                loss_per_sample = criterion(decoded, data).mean(
                    dim=list(range(1, decoded.ndim))
                )
                sample_losses.extend(loss_per_sample.cpu().numpy())

        test_mean = np.mean(sample_losses)
        test_std = np.std(sample_losses)
        test_loss_means.append(test_mean)
        test_loss_stds.append(test_std)
        print(
            f"Epoch [{epoch+1}/{num_epochs}] Test Loss: {test_mean:.6f} ± {test_std:.6f}"
        )

    return train_loss_means, train_loss_stds, test_loss_means, test_loss_stds


class CFG:
    seed = 42
    k = 20  # topk sparsity for SAE

    meta_path = "/mnt/data/gpt2_activations/meta.json"
    layer_name = "transformer.h.5.mlp.c_proj"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "gpt2"
    save_dir = "out"
    save_filename = "loss_data.pkl"
    num_epochs = 20


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloader(meta_path, layer_name, train_bs=32, test_bs=32):
    dataset = GPT2LayerActivations(meta_path, layer_name)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_bs, shuffle=False)
    return train_loader, test_loader


def main():
    args = CFG()
    set_seed(args.seed)
    train_loader, test_loader = get_dataloader(
        args.meta_path, args.layer_name, train_bs=32, test_bs=32
    )

    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    gpt = GPT2LMHeadModel.from_pretrained(args.model_name)
    model = VASAE(
        topk=args.k,
        emb_weight=gpt.transformer.wte.weight,
    ).to(args.device)

    train_loss, train_loss_stds, test_loss, test_loss_stds = train_model(
        model, train_loader, test_loader, args
    )

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.save_filename)
    with open(save_path, "wb") as f:
        pickle.dump(
            {
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_loss_stds": train_loss_stds,
                "test_loss_stds": test_loss_stds,
            },
            f,
        )
    logging.info(f"save loss in {save_path}.")


main()
