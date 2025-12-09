import os
import pickle

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from dataset import GPT2LayerActivations
from models import VASAE
from utils import get_logger, set_seed


def train_model(model: VASAE, train_loader, test_loader, args, logger):
    device = args.device
    num_epochs = args.num_epochs
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_loss_means, train_loss_stds = [], []
    test_loss_means, test_loss_stds = [], []

    for epoch in range(num_epochs):
        model.train()
        sample_losses = []

        for batch_i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            decoded, _ = model(data)
            loss_dict = model.compute_loss(data, decoded, None)
            loss_dict["loss"].backward()
            optimizer.step()

            sample_losses.extend(loss_dict["loss_per_sample"].detach().cpu().numpy())
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} [Train] batch {batch_i+1}/{len(train_loader)} loss {loss_dict["loss"].item():.4f}"
            )
            if batch_i == 2:
                break

        train_mean = np.mean(sample_losses)
        train_std = np.std(sample_losses)
        train_loss_means.append(train_mean)
        train_loss_stds.append(train_std)
        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_mean:.6f} ± {train_std:.6f}"
        )

        # Evaluation
        model.eval()
        sample_losses = []
        with torch.no_grad():
            for batch_i, data in enumerate(test_loader):
                data = data.to(device)
                decoded, _ = model(data)
                loss_dict = model.compute_loss(data, decoded, None)
                sample_losses.extend(loss_dict["loss_per_sample"].cpu().numpy())
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} [Test] batch {batch_i+1}/{len(train_loader)} loss {loss_dict["loss"].item():.4f}"
                )
                if batch_i == 2:
                    break

        test_mean = np.mean(sample_losses)
        test_std = np.std(sample_losses)
        test_loss_means.append(test_mean)
        test_loss_stds.append(test_std)
        logger.info(
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
    train_batchsize = 32
    test_batchsize = 32


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
    logger = get_logger(__file__)
    set_seed(args.seed)
    train_loader, test_loader = get_dataloader(
        args.meta_path,
        args.layer_name,
        train_bs=args.train_batchsize,
        test_bs=args.test_batchsize,
    )

    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    gpt = GPT2LMHeadModel.from_pretrained(args.model_name)
    model = VASAE(
        k=args.k,
        embedding_weight=gpt.transformer.wte.weight,
    ).to(args.device)

    train_loss, train_loss_stds, test_loss, test_loss_stds = train_model(
        model, train_loader, test_loader, args, logger
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
