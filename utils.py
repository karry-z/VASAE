import logging
import os
import random

import numpy as np
import torch


def get_logger(current_pyfile_path):
    current_pyfile_path = os.path.abspath(current_pyfile_path)
    logdir = os.path.join(os.path.dirname(current_pyfile_path), "logs")
    os.makedirs(logdir, exist_ok=True)
    current_pyfile_name = os.path.splitext(os.path.basename(current_pyfile_path))[0]
    logpath = os.path.join(logdir, f"{current_pyfile_name}.log")

    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(logpath)],
    )
    logger = logging.getLogger(current_pyfile_name)
    logger.info(f">>> start logging at {logpath}")
    return logger


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
