import json
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

from dataset import GPT2LayerActivations
from vasae import VASAE

from utils import get_logger, set_seed

logger = get_logger(__file__)


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

    meta_path = "/scratch/b5bq/pu22650.b5bq/gpt2_activations/meta.json"
    layer_name = "transformer.h.5.mlp.c_proj"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "gpt2"
    save_dir = "out"
    save_filename = "loss_data.pkl"
    num_epochs = 20





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
    logger.info(f"save loss in {save_path}.")

    save_model_path = os.path.join(args.save_dir, "sae.pth")
    torch.save(model.state_dict(), save_model_path)
    logger.info(f"save model in {save_model_path}.")


main()
