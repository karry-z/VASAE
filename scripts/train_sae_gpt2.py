import argparse
import os
import pickle

import numpy as np
import torch
import torch.optim as optim
from regex import R

from vasae.data.dataset import get_dataloader
from vasae.metrics.logitlens import LogitLens, LogitLensAccuracy
from vasae.models.factory import VASAE, get_blackbox_model, get_sae_model
from vasae.utils.log import get_logger
from vasae.utils.seed import set_seed


def train_model(
    model: VASAE,
    train_loader,
    test_loader,
    device,
    num_epochs,
    lr,
    max_batchsize,
    logger,
    logitlens: LogitLens,
    logitlens_acc: LogitLensAccuracy,
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    logger.info(f"use Adam with lr={lr}")

    train_loss_means, train_loss_stds = [], []
    test_loss_means, test_loss_stds = [], []
    train_acc_epoch = []
    train_acc_std_epoch = []
    test_acc_epoch = []
    test_acc_std_epoch = []

    for epoch in range(num_epochs):
        model.train()
        sample_losses = []
        data_ids_train_epoch, recons_ids_train_epoch = [], []

        for batch_i, data in enumerate(train_loader):
            data = data.to(device)  # TODO： data是全0
            optimizer.zero_grad()

            decoded, sparse_code = model(data)
            loss_dict = model.compute_loss(data, decoded, sparse_code)
            loss_dict["loss"].backward()
            optimizer.step()

            sample_losses.extend(loss_dict["loss_per_sample"].detach().cpu().numpy())

            # logitlen acc
            data_ids = logitlens.top1(data).cpu()
            recons_ids = logitlens.top1(decoded).cpu()
            acc, acc_std = logitlens_acc.compute(data_ids, recons_ids)
            data_ids_train_epoch.append(data_ids.flatten().tolist())
            recons_ids_train_epoch.append(recons_ids.flatten().tolist())

            logger.info(
                f"Epoch {epoch+1}/{num_epochs} [Train] "
                f"batch {batch_i+1}/{len(train_loader)} "
                f"loss {loss_dict['loss'].item():.4f} "
                f"acc: {acc * 100:.2f}% ± {acc_std * 100:.2f}% "
            )

            if max_batchsize > 0 and batch_i >= max_batchsize:
                logger.debug(f"break at batch {batch_i}")
                break

        train_mean = float(np.mean(sample_losses))
        train_std = float(np.std(sample_losses))
        train_loss_means.append(train_mean)
        train_loss_stds.append(train_std)

        acc, acc_std = logitlens_acc.compute(
            recons_ids_train_epoch, data_ids_train_epoch
        )
        train_acc_epoch.append(acc)
        train_acc_std_epoch.append(acc_std)

        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_mean:.4f} ± {train_std:.4f} "
            f"acc: {acc * 100:.2f}% ± {acc_std * 100:.2f}% "
        )

        model.eval()
        sample_losses = []
        data_ids_test_epoch, recons_ids_test_epoch = [], []

        with torch.no_grad():
            for batch_i, data in enumerate(test_loader):
                data = data.to(device)
                decoded, sparse_code = model(data)
                loss_dict = model.compute_loss(data, decoded, sparse_code)

                sample_losses.extend(loss_dict["loss_per_sample"].cpu().numpy())

                # logitlen acc
                data_ids = logitlens.top1(data).cpu()
                recons_ids = logitlens.top1(decoded).cpu()
                acc, acc_std = logitlens_acc.compute(data_ids, recons_ids)
                data_ids_test_epoch.extend(data_ids.flatten().tolist())
                recons_ids_test_epoch.extend(recons_ids.flatten().tolist())

                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} [Test] "
                    f"batch {batch_i+1}/{len(test_loader)} "
                    f"loss {loss_dict['loss'].item():.4f} "
                    f"acc: {acc * 100:.2f}% ± {acc_std * 100:.2f}% "
                )

                if max_batchsize > 0 and batch_i >= max_batchsize:
                    logger.debug(f"break at batch {batch_i}")
                    break

        test_mean = float(np.mean(sample_losses))
        test_std = float(np.std(sample_losses))
        test_loss_means.append(test_mean)
        test_loss_stds.append(test_std)

        acc, acc_std = logitlens_acc.compute(recons_ids_test_epoch, data_ids_test_epoch)
        test_acc_epoch.append(acc)
        test_acc_std_epoch.append(acc_std)

        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Test Loss: {test_mean:.4f} ± {test_std:.4f} "
            f"acc: {acc * 100:.2f}% ± {acc_std * 100:.2f}% "
        )

    return (
        train_loss_means,
        train_loss_stds,
        test_loss_means,
        test_loss_stds,
        train_acc_epoch,
        train_acc_std_epoch,
        test_acc_epoch,
        test_acc_std_epoch,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    # system
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # data
    parser.add_argument(
        "--meta_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--layer_name",
        type=str,
        default="transformer.h.5",
    )
    parser.add_argument(
        "--blackbox_model",
        type=str,
        default="gpt2",
    )
    parser.add_argument(
        "--use_centralize",
        action="store_true",
    )

    # model
    parser.add_argument(
        "--sae",
        type=str,
        default="VASAE_BatchKSparse",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--dim_input",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--dim_sparse",
        type=int,
        default=50257,
    )

    # training
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--train_batchsize",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--test_batchsize",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--max_batchsize",
        type=int,
        default=0,
        help="for debugging",
    )

    # logging / save
    parser.add_argument(
        "--save_dir",
        type=str,
        default="out",
    )
    parser.add_argument(
        "--save_filename",
        type=str,
        default="loss.pkl",
    )
    parser.add_argument(
        "--sae_save_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--log",
        type=str,
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    set_seed(args.seed)

    logger = get_logger(args.log)
    logger.info(vars(args))

    train_loader, test_loader = get_dataloader(
        args.meta_path,
        args.layer_name,
        train_bs=args.train_batchsize,
        test_bs=args.test_batchsize,
        use_centralize=args.use_centralize,
    )

    blackbox_model, _ = get_blackbox_model(
        args.blackbox_model,
        device,
    )

    model = get_sae_model(
        args.sae,
        k=args.k,
        dim_input=args.dim_input,
        dim_sparse=args.dim_sparse,
        embedding_weight=blackbox_model.transformer.wte.weight,
    ).to(device)

    logger.info(f"model {type(model)} loaded with k={args.k}")

    # train
    logitlens = LogitLens(blackbox_model.lm_head)
    logitlens_acc = LogitLensAccuracy()

    (
        train_loss,
        train_loss_stds,
        test_loss,
        test_loss_stds,
        train_acc_epoch,
        train_acc_std_epoch,
        test_acc_epoch,
        test_acc_std_epoch,
    ) = train_model(
        model,
        train_loader,
        test_loader,
        device=device,
        num_epochs=args.num_epochs,
        lr=args.lr,
        max_batchsize=args.max_batchsize,
        logger=logger,
        logitlens=logitlens,
        logitlens_acc=logitlens_acc,
    )

    # save results
    os.makedirs(args.save_dir, exist_ok=True)

    save_path = os.path.join(
        args.save_dir,
        args.save_filename,
    )
    with open(save_path, "wb") as f:
        pickle.dump(
            {
                "train_loss": train_loss,
                "train_loss_stds": train_loss_stds,
                "test_loss": test_loss,
                "test_loss_stds": test_loss_stds,
                "train_acc_epoch": train_acc_epoch,
                "train_acc_std_epoch": train_acc_std_epoch,
                "test_acc_epoch": test_acc_epoch,
                "test_acc_std_epoch": test_acc_std_epoch,
            },
            f,
        )

    logger.info(f"save loss in {save_path}")

    # save model
    model_path = args.sae_save_path
    torch.save(model.state_dict(), model_path)

    logger.info(f"save model in {model_path}")


if __name__ == "__main__":
    main()
