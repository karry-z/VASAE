import argparse
import os
import pickle

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import GPT2LayerActivations
from models import VASAE, get_blackbox_model, get_sae_model
from utils import get_logger, set_seed


def train_model(model: VASAE, train_loader, test_loader, args, logger):
    device = args.device
    num_epochs = args.num_epochs
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    logger.info(f"use Adam with lr={args.lr}")

    train_loss_means, train_loss_stds = [], []
    test_loss_means, test_loss_stds = [], []

    for epoch in range(num_epochs):
        model.train()
        sample_losses = []

        for batch_i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            decoded, sparse_code = model(data)
            loss_dict = model.compute_loss(data, decoded, sparse_code)
            loss_dict["loss"].backward()
            optimizer.step()

            sample_losses.extend(loss_dict["loss_per_sample"].detach().cpu().numpy())
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} [Train] batch {batch_i+1}/{len(train_loader)} loss {loss_dict["loss"].item():.4f}"
            )

            # TODO: for debugging
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
                decoded, sparse_code = model(data)
                loss_dict = model.compute_loss(data, decoded, sparse_code)
                sample_losses.extend(loss_dict["loss_per_sample"].cpu().numpy())
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} [Test] batch {batch_i+1}/{len(test_loader)} loss {loss_dict["loss"].item():.4f}"
                )

                # TODO: for debugging
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        help="device, cpu or cuda",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--k",
        type=int,
        default=32,
        help="topk sparsity for SAE",
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default="/scratch/b5bq/pu22650.b5bq/activations_gpt2_Geralt-Targaryen_openwebtext2/meta.json",
        help="",
    )
    parser.add_argument(
        "--layer_name",
        type=str,
        default="transformer.h.5",
        help="",
    )
    parser.add_argument(
        "--blackbox_model",
        type=str,
        default="gpt2",
        help="blackbox model",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="out",
        help="",
    )
    parser.add_argument(
        "--save_filename",
        type=str,
        default="loss_vasae_gpt2_transformer_h_5_openwebtext_wocen.pkl",
        help="",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="",
    )
    parser.add_argument(
        "--train_batchsize",
        type=int,
        default=32,
        help="",
    )
    parser.add_argument(
        "--test_batchsize",
        type=int,
        default=32,
        help="",
    )
    parser.add_argument("--use_centralize", action="store_true", help="")
    parser.add_argument(
        "--sae",
        type=str,
        default="VASAE_BatchKSparse",
        help="",
    )
    parser.add_argument(
        "--dim_input",
        type=int,
        default=768,
        help="",
    )
    parser.add_argument(
        "--dim_sparse",
        type=int,
        default=50257,
        help="",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=__file__,
        help="log path",
    )
    return parser.parse_args()


def get_dataloader(
    meta_path, layer_name, train_bs=32, test_bs=32, use_centralize=False
):
    dataset = GPT2LayerActivations(meta_path, layer_name, use_centralize)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_bs, shuffle=False)
    return train_loader, test_loader


def main():
    args = parse_args()
    logger = get_logger(args.log)
    set_seed(args.seed)
    train_loader, test_loader = get_dataloader(
        args.meta_path,
        args.layer_name,
        train_bs=args.train_batchsize,
        test_bs=args.test_batchsize,
        use_centralize=args.use_centralize,
    )
    blackbox_model, _ = get_blackbox_model(args.blackbox_model, args.device)
    model = get_sae_model(
        args.sae,
        k=args.k,
        dim_input=args.dim_input,
        dim_sparse=args.dim_sparse,
        embedding_weight=blackbox_model.transformer.wte.weight,
    ).to(args.device)
    logger.info(f"model {type(model)} loaded with k={args.k}")

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
    logger.info(f"save loss in {save_path}")

    save_model_path = os.path.join(
        args.save_dir, f"{args.sae}_k{args.k}_lr{args.lr}.pth"
    )
    torch.save(model.state_dict(), save_model_path)
    logger.info(f"save model in {save_model_path}")


if __name__ == "__main__":
    main()
