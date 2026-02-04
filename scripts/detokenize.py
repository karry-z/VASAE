import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from vasae.data.dataset import get_dataloader
from vasae.metrics.logitlens import LogitLens, LogitLensAccuracy
from vasae.models.factory import VASAE, get_blackbox_model, get_sae_model
from vasae.utils.log import get_logger
from vasae.utils.seed import set_seed


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
        default="out/logitlens",
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

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader, valid_loader, test_loader = get_dataloader(
        args.meta_path,
        args.layer_name,
        train_bs=args.train_batchsize,
        test_bs=args.test_batchsize,
        use_centralize=args.use_centralize,
    )

    blackbox_model, tokenizer = get_blackbox_model(
        args.blackbox_model,
        device,
    )

    logitlens = LogitLens(unembed_layer=blackbox_model.lm_head)

    rows = []
    example_offset = 0
    # layer_i = int(args.layer_name[-1])
    layer_i = 3
    with torch.no_grad():
        for batch_i, data in enumerate(test_loader):
            data = data.to(device)

            logitlens_output = logitlens.top1(data)
            token_ids, token_probs = (
                logitlens_output["token_ids"],
                logitlens_output["token_probs"],
            )

            B, S = token_ids.shape

            for b in range(B):
                example_i = example_offset + b
                for s in range(S):
                    token = tokenizer.decode(token_ids[b, s])
                    prob = float(token_probs[b, s])

                    rows.append(
                        {
                            "layer_i": layer_i,
                            "example_i": example_i,
                            "seq_i": s,
                            "token": token,
                            "prob": prob,
                        }
                    )

            example_offset += B

            if args.max_batchsize > 0 and batch_i >= args.max_batchsize:
                logger.debug(f"break at batch {batch_i}")
                break

        df = pd.DataFrame(rows)
        df["prob"] = df["prob"].astype("float32")

        out_path = save_dir / f"logitlens_layer_{layer_i}.parquet"
        df.to_parquet(out_path, index=False)

        logger.info(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
